function void ug_mesh_optimize_partition_groups(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  B32 *is_boundary = arena_push_count(scratch.arena, B32, mesh->cells.len);
  U64 boundary_count = 0;
  for Iter_Index(it_cell, mesh->cells.len) {
    B32 boundary = 0;
    for Iter_Index(it_face, 4) {
      U32 adjacent = mesh->cells.faces[it_cell].adjacent[it_face];
      if (adjacent >= mesh->cells.len && adjacent < mesh->cells.len + mesh->halos.len) {
        boundary = 1;
        break;
      }
    }
    is_boundary[it_cell] = boundary;
    boundary_count      += boundary;
  }

  mesh->groups.boundary_len = boundary_count;
  mesh->groups.interior_len = mesh->cells.len - boundary_count;

  // NOTE(cmat): old_to_new: interior cells keep relative order at the front,
  // - boundary cells keep relative order at the back. Stable partition.
  U32 *old_to_new = arena_push_count(scratch.arena, U32, mesh->cells.len);
  U64 interior_at = 0;
  U64 boundary_at = mesh->groups.interior_len;
  for Iter_Index(it, mesh->cells.len) {
    old_to_new[it] = is_boundary[it] ? (U32)(boundary_at++) : (U32)(interior_at++);
  }

  // NOTE(cmat): Need an inverse index array (new->old = "y") shaped like
  // - morton_codes.y for array_reorder to consume; build it directly.
  U32 *new_to_old = arena_push_count(scratch.arena, U32, mesh->cells.len);
  for Iter_Index(it, mesh->cells.len) {
    new_to_old[old_to_new[it]] = (U32)it;
  }

  array_reorder(mesh->cells.len, sizeof(V3F),           (U08 *)mesh->cells.center, sizeof(U32), new_to_old);
  array_reorder(mesh->cells.len, sizeof(F32),           (U08 *)mesh->cells.volume, sizeof(U32), new_to_old);
  array_reorder(mesh->cells.len, sizeof(UG_Cell_Faces), (U08 *)mesh->cells.faces,  sizeof(U32), new_to_old);

  for Iter_Index(it_cell, mesh->cells.len) {
    for Iter_Index(it_face, 4) {
      U32  old_adjacent =  mesh->cells.faces[it_cell].adjacent[it_face];
      U32 *new_adjacent = &mesh->cells.faces[it_cell].adjacent[it_face];
      if (old_adjacent < mesh->cells.len) { *new_adjacent = old_to_new[old_adjacent]; }
    }
  }
  for Iter_Index(it, mesh->ghosts.len) { mesh->ghosts.parent_cell[it] = old_to_new[mesh->ghosts.parent_cell[it]]; }
  for Iter_Index(it, mesh->sends.len)  { mesh->sends.cell_send[it]    = old_to_new[mesh->sends.cell_send[it]];    }

  scratch_end(&scratch);
  profiler_end_function();
}


// ------------------------------------------------------------
// #-- Grouping optimization

// NOTE(cmat): We group cells by classifying interior and boundary cells.
// - This allows us to start solving the flux for the innermost cells,
// - while waiting for the halo cells to be exchanged.
function void ug_mesh_reorder_by_groups(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_info("Reordering by groups");

  // NOTE(cmat): Count boundary cells (cells touching halos).
  U64 *boundary_count_global  = 0;
  U64 *lane_cell_count_global = 0;
  if (lane_index() == 0) {
    boundary_count_global   = arena_push_count(scratch.arena, U64, lane_count());
    lane_cell_count_global  = arena_push_count(scratch.arena, U64, lane_count());
  }

  lane_broadcast_ptr(&boundary_count_global,  0);
  lane_broadcast_ptr(&lane_cell_count_global, 0);

  lane_cell_count_global[lane_index()] = range1_u64_len(lane_range(mesh->cells.len));

  // NOTE(cmat): is_boundary is reused below to build the permutation,
  // - so we keep it around instead of only counting.
  B32 *is_boundary = 0;
  if (lane_index() == 0) {
    is_boundary = arena_push_count(scratch.arena, B32, mesh->cells.len);
  }
  lane_broadcast_ptr(&is_boundary, 0);

  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    B32 boundary_cell = 0;
    for Iter_Index(it_face, 4) {
      U32 adjacent = mesh->cells.faces[it_cell].adjacent[it_face];
      if (adjacent >= mesh->cells.len && adjacent < mesh->cells.len + mesh->halos.len) {
        boundary_cell = 1;
        break;
      }
    }

    is_boundary[it_cell] = boundary_cell;
    if (boundary_cell) {
      boundary_count_global[lane_index()] += 1;
    }
  }

  // NOTE(cmat): Gather counts.
  lane_barrier();
  U64 boundary_count = 0;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      boundary_count += boundary_count_global[it];
    }
  }

  lane_broadcast_u64(&boundary_count, 0);

  // NOTE(cmat): We now have the interior and boundary ranges.
  U64 interior_count = mesh->cells.len - boundary_count;
  mesh->groups.cells_interior = range1_u64(0,               interior_count);
  mesh->groups.cells_boundary = range1_u64(interior_count,  mesh->cells.len);

  // NOTE(cmat): Compute per-lane prefix-sum offsets, same pattern as before:
  // - interior cells write starting at their prefix-summed interior offset,
  // - boundary cells write starting at interior_count + their prefix-summed boundary offset.
  lane_barrier();
  U64 boundary_offset = 0;
  U64 interior_offset = 0;
  for Iter_Index(it, lane_index()) {
    boundary_offset += boundary_count_global[it];
    interior_offset += lane_cell_count_global[it] - boundary_count_global[it];
  }

  // NOTE(cmat): Build old_to_new: stable partition, interior first then boundary.
  U32 *old_to_new = 0;
  if (lane_index() == 0) {
    old_to_new = arena_push_count(scratch.arena, U32, mesh->cells.len);
  }
  lane_broadcast_ptr(&old_to_new, 0);

  U64 boundary_at = interior_count + boundary_offset;
  U64 interior_at = interior_offset;
  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    if (is_boundary[it_cell]) {
      old_to_new[it_cell] = (U32)(boundary_at++);
    } else {
      old_to_new[it_cell] = (U32)(interior_at++);
    }
  }

  // NOTE(cmat): Build inverse map (new -> old) for array_reorder, which
  // - walks NEW slots and pulls from OLD ones. Safe to scatter in parallel
  // - since old_to_new is a permutation (each new index written exactly once).
  lane_barrier();
  U32 *new_to_old = 0;
  if (lane_index() == 0) {
    new_to_old = arena_push_count(scratch.arena, U32, mesh->cells.len);
  }
  lane_broadcast_ptr(&new_to_old, 0);

  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    new_to_old[old_to_new[it_cell]] = (U32)it_cell;
  }

  // NOTE(cmat): Reorder elements.
  lane_barrier();
  array_reorder(mesh->cells.len, sizeof(V3F),           (U08 *)mesh->cells.center, sizeof(U32), new_to_old);
  array_reorder(mesh->cells.len, sizeof(F32),           (U08 *)mesh->cells.volume, sizeof(U32), new_to_old);
  array_reorder(mesh->cells.len, sizeof(UG_Cell_Faces), (U08 *)mesh->cells.faces,  sizeof(U32), new_to_old);

  // NOTE(cmat): Remap adjacent cells (only inner ones).
  lane_barrier();
  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    for Iter_Index(it_face, 4) {
      U32  old_adjacent =  mesh->cells.faces[it_cell].adjacent[it_face];
      U32 *new_adjacent = &mesh->cells.faces[it_cell].adjacent[it_face];
      if (old_adjacent < mesh->cells.len) {
        *new_adjacent = old_to_new[old_adjacent];
      }
    }
  }

  // NOTE(cmat): Remap ghost parents.
  for Iter_Range(it, lane_range(mesh->ghosts.len)) {
    mesh->ghosts.parent_cell[it] = old_to_new[mesh->ghosts.parent_cell[it]];
  }

  // NOTE(cmat): Remap send cells.
  for Iter_Range(it, lane_range(mesh->sends.len)) {
    mesh->sends.cell_send[it] = old_to_new[mesh->sends.cell_send[it]];
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}
