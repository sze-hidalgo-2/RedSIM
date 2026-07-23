struct UG_Cell_Faces_Verts;
function void ug_mesh_compute_cells                   (UG_Mesh *mesh, UG_Grid *grid, Arena *arena);
function void ug_mesh_compute_cells_faces             (UG_Mesh *mesh, UG_Grid *grid, Arena *arena);
function void ug_mesh_compute_cells_faces_ghosts      (UG_Mesh *mesh, UG_Grid *grid, struct UG_Cell_Faces_Verts *faces_verts);

// ------------------------------------------------------------
// #-- Initialization

function void ug_mesh_init_from_grid(UG_Mesh *mesh, UG_Grid *grid, Arena *arena) {
  profiler_begin_function();
  Log_Zone_Scope("Computing mesh from grid") {
    ug_mesh_compute_cells               (mesh, grid, arena);  // NOTE(cmat): Compute cell geometric data.
    ug_mesh_compute_cells_faces         (mesh, grid, arena);  // NOTE(cmat): Compute cell adjacency.
  }

  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Compute Cells

function void ug_mesh_compute_cells(UG_Mesh *mesh, UG_Grid *grid, Arena *arena) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(arena);
  log_zone_start("Computing cells");

  Range3_F32 *bounds_global = 0;
  mesh->cells.len           = grid->elems.len;

  if (lane_index() == 0) {
    mesh->cells.center = arena_push_count(arena, V3F,    mesh->cells.len);
    mesh->cells.volume = arena_push_count(arena, F32,    mesh->cells.len);
    bounds_global      = arena_push_count(scratch.arena, Range3_F32, lane_count());
  }

  lane_broadcast_ptr(&mesh->cells.center,  0);
  lane_broadcast_ptr(&mesh->cells.volume,  0);
  lane_broadcast_ptr(&bounds_global,        0);

  Range3_F32 *bounds_local = bounds_global + lane_index();
  *bounds_local = range3_f32(v3f_f32(f32_limit_max), v3f_f32(f32_limit_min));

  // NOTE(cmat): Compute cell center, volume, bounding box.
  // TODO(cmat): Rewrite using SIMD.
  log_info("Computing cell data");
  for Iter_Range(it, lane_range(mesh->cells.len)) {

    // NOTE(cmat): Gather vectors.
    V4U v    = grid->elems.verts[it];
    V3F a    = v3f(grid->verts.x[v.x], grid->verts.y[v.x], grid->verts.z[v.x]);
    V3F b    = v3f(grid->verts.x[v.y], grid->verts.y[v.y], grid->verts.z[v.y]);
    V3F c    = v3f(grid->verts.x[v.z], grid->verts.y[v.z], grid->verts.z[v.z]);
    V3F d    = v3f(grid->verts.x[v.w], grid->verts.y[v.w], grid->verts.z[v.w]);

    // NOTE(cmat): Compute center, volume and bounds.
    mesh->cells.center[it] = v3f_mul(.25f, v3f_add(a, v3f_add(b, v3f_add(c, d))));
    mesh->cells.volume[it] = f32_abs(v3f_dot(v3f_sub(a, d), v3f_cross(v3f_sub(b, d), v3f_sub(c, d))) / 6.f);

    for Iter_Index(elem, 3) {
      bounds_local->min.dat[elem] = f32_min(bounds_local->min.dat[elem], a.dat[elem]);
      bounds_local->min.dat[elem] = f32_min(bounds_local->min.dat[elem], b.dat[elem]);
      bounds_local->min.dat[elem] = f32_min(bounds_local->min.dat[elem], c.dat[elem]);
      bounds_local->min.dat[elem] = f32_min(bounds_local->min.dat[elem], d.dat[elem]);

      bounds_local->max.dat[elem] = f32_max(bounds_local->max.dat[elem], a.dat[elem]);
      bounds_local->max.dat[elem] = f32_max(bounds_local->max.dat[elem], b.dat[elem]);
      bounds_local->max.dat[elem] = f32_max(bounds_local->max.dat[elem], c.dat[elem]);
      bounds_local->max.dat[elem] = f32_max(bounds_local->max.dat[elem], d.dat[elem]);
    }
  }

  // NOTE(cmat): Compute final bounds from each lane.
  lane_barrier();

  log_info("Synchronizing thread bounds");
  if (lane_index() == 0) {
    Range3_F32 bounds = range3_f32(v3f_f32(f32_limit_max), v3f_f32(f32_limit_min));
    for Iter_Index(it, lane_count()) {
      for Iter_Index(elem, 3) {
        bounds.min.dat[elem] = f32_min(bounds.min.dat[elem], bounds_global[it].min.dat[elem]);
        bounds.max.dat[elem] = f32_max(bounds.max.dat[elem], bounds_global[it].max.dat[elem]);
      }
    }

    // NOTE(cmat): Store the final bounds in the first slot.
    bounds_global[0] = bounds;
  }

  lane_barrier();

  // NOTE(cmat): Broadcast bounds.
  mesh->bounds = bounds_global[0];
  log_info("Mesh bounds: (%.2g, %.2g, %.2g), (%.2g, %.2g, %.2g)", V3_Expand(mesh->bounds.min), V3_Expand(mesh->bounds.max));

  lane_barrier();

  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}
// ------------------------------------------------------------
// #--  Compute Cell Faces.

#pragma pack(push, 1)
  typedef struct UG_Tetra_Face {
    V3U verts;  // NOTE(cmat): Sorted vertex indices
    U32 cell;   // NOTE(cmat): Cell index.
    U32 face;   // NOTE(cmat): Face index (0, 1, 2 or 3). Padded to 32 bit boundaty.
  } UG_Tetra_Face;
#pragma pack(pop)

typedef struct UG_Cell_Faces_Verts {
  V3U verts[4];
} UG_Cell_Faces_Verts;

Assert_Compiler(sizeof(UG_Tetra_Face) == 5 * sizeof(U32));

force_inline function B32 ug_tetra_face_match(UG_Tetra_Face *lhs, UG_Tetra_Face *rhs) {
  B32 match = memory_match(&lhs->verts, &rhs->verts, sizeof(V3U)); // NOTE(cmat): Only check vx, vy, vz.
  return match;
}

function void ug_mesh_compute_cells_faces(UG_Mesh *mesh, UG_Grid *grid, Arena *arena) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(arena);
  log_zone_start("Computing cell faces");

  // NOTE(cmat): Store all faces, including duplicates.
  U64            faces_all_len = 0; 
  UG_Tetra_Face *faces_all_dat = 0;

  if (lane_index() == 0) {
    faces_all_len = 4 * mesh->cells.len;
    faces_all_dat = arena_push_count(scratch.arena, UG_Tetra_Face, faces_all_len);
  }

  lane_broadcast_u64(&faces_all_len, 0);
  lane_broadcast_ptr(&faces_all_dat, 0);

  // NOTE(cmat): Add all faces.
  log_info("Sorting cell vertices");
  for Iter_Range(it, lane_range(grid->elems.len)) {
    
    // NOTE(cmat): Sort grid vertices, so that each face is unique in its description.
    V4U cell = grid->elems.verts[it];
    v4_u32_sort(&cell);

    U64 face_index                = 4 * it;
    faces_all_dat[face_index + 0] = (UG_Tetra_Face) { .verts = v3u(cell.x, cell.y, cell.z), .cell = it, .face = 0 };
    faces_all_dat[face_index + 1] = (UG_Tetra_Face) { .verts = v3u(cell.x, cell.y, cell.w), .cell = it, .face = 1 };
    faces_all_dat[face_index + 2] = (UG_Tetra_Face) { .verts = v3u(cell.x, cell.z, cell.w), .cell = it, .face = 2 };
    faces_all_dat[face_index + 3] = (UG_Tetra_Face) { .verts = v3u(cell.y, cell.z, cell.w), .cell = it, .face = 3 };
  }

  // NOTE(cmat): Now we radix-sort all faces resulting in duplicates being next to each other.
  lane_barrier();
  log_info("Sorting faces");

  array_sort_radix_u32(faces_all_len, 5, 2, (U32 *)faces_all_dat);
  array_sort_radix_u32(faces_all_len, 5, 1, (U32 *)faces_all_dat);
  array_sort_radix_u32(faces_all_len, 5, 0, (U32 *)faces_all_dat);
  
  // NOTE(cmat): Count how many duplicate faces we have on each lane.
  lane_barrier();
  log_info("Counting shared faces");

  U64 *duplicate_count_global = 0;
  if (lane_index() == 0) {
    duplicate_count_global = (U64 *)arena_push_count(scratch.arena, U64, lane_count());
  }

  lane_broadcast_ptr(&duplicate_count_global, 0);

  U64 *duplicate_count_local = duplicate_count_global + lane_index();
  *duplicate_count_local = 0;
  for Iter_Range(it, lane_range(faces_all_len - 1)) {
    *duplicate_count_local += ug_tetra_face_match(&faces_all_dat[it], &faces_all_dat[it + 1]);
  }

  // NOTE(cmat): Merge local duplicate counts.
  lane_barrier();
  U64 duplicate_count = 0;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      duplicate_count += duplicate_count_global[it];
    }
  }

  lane_broadcast_u64(&duplicate_count, 0);

  U32 faces_len = faces_all_len - duplicate_count;
  log_info("Face count: %'llu", faces_len);

  mesh->ghosts.len = faces_all_len - 2 * duplicate_count;
  log_info("Ghost count: %'llu", mesh->ghosts.len);
  
  UG_Cell_Faces_Verts *faces_verts = 0;

  // NOTE(cmat): Compute face information for each cell.
  if (lane_index() == 0) {
    mesh->cells.faces         = arena_push_count(arena, UG_Cell_Faces,       mesh->cells.len);
    mesh->ghosts.parent_cell  = arena_push_count(arena, U32,                 mesh->ghosts.len);
    mesh->ghosts.parent_face  = arena_push_count(arena, U08,                 mesh->ghosts.len);
    mesh->ghosts.marker_index = arena_push_count(arena, U32,                 mesh->ghosts.len);

    faces_verts               = arena_push_count(arena, UG_Cell_Faces_Verts, mesh->cells.len);
  }

  lane_broadcast_ptr(&mesh->cells.faces,          0);
  lane_broadcast_ptr(&mesh->ghosts.parent_cell,   0);
  lane_broadcast_ptr(&mesh->ghosts.parent_face,   0);
  lane_broadcast_ptr(&mesh->ghosts.marker_index,  0);

  lane_broadcast_ptr(&faces_verts,                0);

  Range1_U64 faces_all_range = lane_range(faces_all_len);

  // NOTE(cmat): Compute ghost count for each lane.
  U64 *global_ghost_len = 0;
  if (lane_index() == 0) {
    global_ghost_len = arena_push_count(scratch.arena, U64, lane_count());
  }

  lane_broadcast_ptr(&global_ghost_len, 0);
  U64 *local_ghost_len = global_ghost_len + lane_index();
  *local_ghost_len = 0;
  
  {
    U64 faces_it = faces_all_range.min;
    if (faces_it > 0) { faces_it += ug_tetra_face_match(&faces_all_dat[faces_it - 1], &faces_all_dat[faces_it]); }

    for (; faces_it < faces_all_range.max; faces_it += 1) {
      B32 match  = 0;
      if (faces_it + 1 < faces_all_len) {
        match = ug_tetra_face_match(&faces_all_dat[faces_it], &faces_all_dat[faces_it + 1]);
      }

      if (!match) {
        *local_ghost_len += 1;
      }

      faces_it += match ? 1 : 0;
    }
  }

  lane_barrier();

  // NOTE(cmat): Compute ghost offset for each lane.
  U64 local_ghost_offset = 0;
  for Iter_Index(it, lane_index()) {
    local_ghost_offset += global_ghost_len[it];
  }

  // NOTE(cmat): If we would start on the second pair of a duplicate skip the first face.
  U64 faces_it = faces_all_range.min;
  if (faces_it > 0) {
    faces_it += ug_tetra_face_match(&faces_all_dat[faces_it - 1], &faces_all_dat[faces_it]);
  }

  for (; faces_it < faces_all_range.max; faces_it += 1) {
    // NOTE(cmat): If there's no match in the last two, that means the last face does not have and adjancent cell.
    B32 match  = 0;
    if (faces_it + 1 < faces_all_len) {
      match = ug_tetra_face_match(&faces_all_dat[faces_it], &faces_all_dat[faces_it + 1]);
    }
   
    V3U verts = faces_all_dat[faces_it + 0].verts;
    U32 lhs_cell_index = faces_all_dat[faces_it + 0].cell;
    U32 lhs_face_index = faces_all_dat[faces_it + 0].face;
    U32 rhs_cell_index = 0;

    // NOTE(cmat): Compute face area and normal.
    V3F x0    = v3f(grid->verts.x[verts.dat[0]], grid->verts.y[verts.dat[0]], grid->verts.z[verts.dat[0]]);
    V3F x1    = v3f(grid->verts.x[verts.dat[1]], grid->verts.y[verts.dat[1]], grid->verts.z[verts.dat[1]]);
    V3F x2    = v3f(grid->verts.x[verts.dat[2]], grid->verts.y[verts.dat[2]], grid->verts.z[verts.dat[2]]);
    V3F fc    = v3f_mul(1.f/3.f, v3f_add(x0, v3f_add(x1, x2)));
    V3F cc    = mesh->cells.center[lhs_cell_index];
    V3F x01   = v3f_sub(x1, x0);
    V3F x02   = v3f_sub(x2, x0);
    V3F cross = v3f_cross(x01, x02);
    F32 area  = .5f * v3f_len(cross);
    V3F n     = v3f_noz(cross);
    n         = v3f_dot(n, v3f_sub(fc, cc)) >= 0 ? n : v3f_mul(-1.f, n);

    if (match) {
      rhs_cell_index = faces_all_dat[faces_it + 1].cell;
    } else {
      rhs_cell_index                                = mesh->cells.len + local_ghost_offset;
      mesh->ghosts.parent_cell[local_ghost_offset]  = lhs_cell_index;
      mesh->ghosts.parent_face[local_ghost_offset]  = (U08)lhs_face_index;
      local_ghost_offset                           += 1;
    }

    // NOTE(cmat): Sort before adding faces for future sorts.
    v3_u32_sort(&verts);

    UG_Cell_Faces_Verts *lhs_faces_verts    = faces_verts + lhs_cell_index;
    lhs_faces_verts->verts[lhs_face_index]  = verts;

    UG_Cell_Faces       *lhs_faces        = mesh->cells.faces       + lhs_cell_index;
    lhs_faces->adjacent [lhs_face_index]  = rhs_cell_index;
    lhs_faces->area     [lhs_face_index]  = area;
    lhs_faces->normal_x [lhs_face_index]  = n.x;
    lhs_faces->normal_y [lhs_face_index]  = n.y;
    lhs_faces->normal_z [lhs_face_index]  = n.z;

    if (match) {
      U32 rhs_face_index = faces_all_dat[faces_it + 1].face;

      UG_Cell_Faces_Verts *rhs_faces_verts    = faces_verts + rhs_cell_index;
      rhs_faces_verts->verts[rhs_face_index]  = verts;

      UG_Cell_Faces *rhs_faces             = mesh->cells.faces + rhs_cell_index;
      rhs_faces->adjacent [rhs_face_index] = lhs_cell_index;
      rhs_faces->area     [rhs_face_index] = area;
      rhs_faces->normal_x [rhs_face_index] = -n.x;
      rhs_faces->normal_y [rhs_face_index] = -n.y;
      rhs_faces->normal_z [rhs_face_index] = -n.z;
    }

    // NOTE(cmat): Skip next face if there's a match
    faces_it += match ? 1 : 0;
  }

  // NOTE(cmat): Assign ghost cells.
  lane_barrier();
  ug_mesh_compute_cells_faces_ghosts(mesh, grid, faces_verts);

  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

#pragma pack(push, 1)
typedef struct UG_Ghost_Face {
  V3U verts;
  U32 type;
  union { U32 ghost_index, marker_index; };
} UG_Ghost_Face;

#pragma pack(pop)

Assert_Compiler(sizeof(UG_Ghost_Face) == 5 * sizeof(U32));

force_inline function B32 ug_ghost_face_match(UG_Ghost_Face *lhs, UG_Ghost_Face *rhs) {
  B32 match = memory_match(&lhs->verts, &rhs->verts, sizeof(V3U)); // NOTE(cmat): Only check vx, vy, vz.
  return match;
}

function void ug_mesh_compute_cells_faces_ghosts(UG_Mesh *mesh, UG_Grid *grid, UG_Cell_Faces_Verts *faces_verts) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Assigning marks to ghost cells");

  U64            ghost_faces_all_len  = 0;
  UG_Ghost_Face *ghost_faces_all      = 0;

  if (lane_index() == 0) {
    U64 markers_len = 0;
    for Iter_Index(it, grid->markers.len) {
      markers_len += grid->markers.elems[it].len;
    }

    ghost_faces_all_len = mesh->ghosts.len + markers_len;
    ghost_faces_all     = arena_push_count(scratch.arena, UG_Ghost_Face, ghost_faces_all_len);
  }

  lane_broadcast_u64(&ghost_faces_all_len,  0);
  lane_broadcast_ptr(&ghost_faces_all,      0);

  // NOTE(cmat): Add ghost cell faces.
  log_info("Adding ghosts faces");
  for Iter_Range(it, lane_range(mesh->ghosts.len)) {
    U32 parent_cell  = mesh->ghosts.parent_cell[it];
    U08 parent_face  = mesh->ghosts.parent_face[it];
    V3U parent_verts = faces_verts[parent_cell].verts[parent_face];

    // NOTE(cmat): parent_verts is already sorted
    ghost_faces_all[it] = (UG_Ghost_Face) { .verts = parent_verts, .type = 0, .ghost_index = it };
  }

  // NOTE(cmat): Add marker faces.
  log_info("Adding marker faces");
  for Iter_Range(marker_it, lane_range(grid->markers.len)) {
    UG_Grid_Marker_Elems *elems = &grid->markers.elems[marker_it];
    U64 write_index             = mesh->ghosts.len;

    for Iter_Index(it, marker_it) {
      write_index += grid->markers.elems[it].len;
    }

    for Iter_Index(elem_it, elems->len) {
      V3U *marker_verts = &elems->verts[elem_it];
      v3_u32_sort(marker_verts);

      // NOTE(cmat): We store the marker ID using the face entry.
      ghost_faces_all[write_index + elem_it] = (UG_Ghost_Face) { .verts = *marker_verts, .type = 1, .marker_index = marker_it };
    }
  }

  // NOTE(cmat): Sort faces.
  log_info("Sorting faces");
  lane_barrier();
  array_sort_radix_u32(ghost_faces_all_len, 5, 2, (U32 *)ghost_faces_all);
  array_sort_radix_u32(ghost_faces_all_len, 5, 1, (U32 *)ghost_faces_all);
  array_sort_radix_u32(ghost_faces_all_len, 5, 0, (U32 *)ghost_faces_all);

  // NOTE(cmat): Based on duplicates, assign ghost cell types.
  lane_barrier();

  log_info("Assigning faces");
  U64 miss_count = 0;
  for Iter_Range(it, lane_range(ghost_faces_all_len - 1)) {
    UG_Ghost_Face *f1 = &ghost_faces_all[it + 0];
    UG_Ghost_Face *f2 = &ghost_faces_all[it + 1];
    if ((f1->type != f2->type) && ug_ghost_face_match(f1, f2)) {
      U32 ghost_index   = (f1->type == 0) ? f1->ghost_index  : f2->ghost_index;
      U32 marker_index  = (f1->type == 1) ? f1->marker_index : f2->marker_index;

      mesh->ghosts.marker_index[ghost_index] = marker_index;
      it += 1; // NOTE(cmat): Skip next face.
    } else {
      miss_count += 1;
    }
  }

  if (miss_count) {
    log_warning("There were %'llu misses while assigning markers", miss_count);
  }

  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Mesh Partitioning

#pragma pack(push, 1)
  typedef struct UG_Halo_key {
    U32 partition;
    U32 cell_global;
    U32 cell_local;
    U32 face_local;
  } UG_Halo_Key;
#pragma pack(pop)

function void ug_mesh_from_sub_mesh(UG_Mesh *mesh, UG_Mesh *mesh_global, UG_Partition *partition, U32 block_index, Arena *arena) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(arena);
  log_zone_start("Computing sub-mesh for partition %u", block_index);

  UG_Partition_Block *block = &partition->blocks_dat[block_index];
  mesh->cells.len           = block->cells_len;

  if (lane_index() == 0) {
    mesh->cells.center  = arena_push_count(arena, V3F,            mesh->cells.len);
    mesh->cells.volume  = arena_push_count(arena, F32,            mesh->cells.len);
    mesh->cells.faces   = arena_push_count(arena, UG_Cell_Faces,  mesh->cells.len);
  }

  lane_broadcast_ptr(&mesh->cells.center, 0);
  lane_broadcast_ptr(&mesh->cells.volume, 0);
  lane_broadcast_ptr(&mesh->cells.faces,  0);

  // NOTE(cmat): Gather values based from partitioning.
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    mesh->cells.center[it]  = mesh_global->cells.center [block->cells_dat[it]];
    mesh->cells.volume[it]  = mesh_global->cells.volume [block->cells_dat[it]];
    mesh->cells.faces[it]   = mesh_global->cells.faces  [block->cells_dat[it]];
  }

  // NOTE(cmat): Compute ghost cell count, halo cell count.
  lane_barrier();

  U64 *ghost_count  = 0;
  U64 *halo_count   = 0;
  if (lane_index() == 0) {
    ghost_count = arena_push_count(scratch.arena, U64, lane_count());
    halo_count  = arena_push_count(scratch.arena, U64, lane_count());
  }

  lane_broadcast_ptr(&ghost_count,  0);
  lane_broadcast_ptr(&halo_count,   0);

  // NOTE(cmat): Iterate through adjacency, count ghost cells and halo cells.
  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    for Iter_Index(it_face, 4) {
      U32 adjacent = mesh->cells.faces[it_cell].adjacent[it_face];
    
      if (adjacent >= mesh_global->cells.len) {
        ghost_count[lane_index()] += 1;
      } else if (partition->cells_block_index[adjacent] != block_index) {
        halo_count[lane_index()] += 1;
      }
    }
  }

  // NOTE(cmat): Compute total counts.
  U64 ghost_total_count = 0;
  U64 halo_total_count  = 0;

  lane_barrier();
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      ghost_total_count += ghost_count  [it];
      halo_total_count  += halo_count   [it];
    }
  }

  log_info("Ghost cell count %'llu", ghost_total_count);

  // NOTE(cmat): Compute prefix sums for ghost and halo offets.
  lane_barrier();
  U64 ghost_offset = 0;
  U64 halo_offset = 0;

  for Iter_Index(it, lane_index()) {
    ghost_offset += ghost_count [it];
    halo_offset  += halo_count  [it];
  }

  lane_broadcast_u64(&ghost_total_count,  0);
  lane_broadcast_u64(&halo_total_count,   0);

  // NOTE(cmat): Sort halo cells, remove duplicates.
  UG_Halo_Key *halo_keys = 0;
  if (lane_index() == 0) {
    halo_keys = arena_push_count(arena, UG_Halo_Key, halo_total_count);
  }

  lane_broadcast_ptr(&halo_keys, 0);

  // NOTE(cmat): Add halo cells.
  U64 halo_key_at = 0;
  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    for Iter_Index(it_face, 4) {
      U32 adjacent = mesh->cells.faces[it_cell].adjacent[it_face];

      if (adjacent >= mesh_global->cells.len) {
        // NOTE(cmat): Ghost.
      } else if (partition->cells_block_index[adjacent] != block_index) {
        halo_keys[halo_offset + (halo_key_at++)] = (UG_Halo_Key) {
          .partition    = partition->cells_block_index[adjacent],
          .cell_global  = adjacent,
          .cell_local   = it_cell,
          .face_local   = it_face,
        };
      }
    }
  }

  // NOTE(cmat): Sort halo keys.
  lane_barrier();

  // NOTE(cmat): Proper communication consistency is determined by the ordering (partition, globla_cell).
  // - Both neighboring partitions rely on this ordering, order is important!
  array_sort_radix_u32(halo_total_count, sizeof(UG_Halo_Key) / sizeof(U32), 1, (U32 *)halo_keys); // NOTE(cmat): Sort by global cell (identify duplicates).
  array_sort_radix_u32(halo_total_count, sizeof(UG_Halo_Key) / sizeof(U32), 0, (U32 *)halo_keys); // NOTE(cmat): Sort by partition.

  // NOTE(cmat): Count unique halo keys.
  lane_barrier();

  U64 *halo_unique_count = 0;
  if (lane_index() == 0) {
    halo_unique_count = arena_push_count(scratch.arena, U64, lane_count());
  }

  lane_broadcast_ptr(&halo_unique_count, 0);
  Range1_U64 halo_total_range = lane_range(halo_total_count);

  U64 halo_unique_begin = halo_total_range.min;
  if (halo_unique_begin > 0) {
    UG_Halo_Key *a = &halo_keys[halo_unique_begin - 1];
    UG_Halo_Key *b = &halo_keys[halo_unique_begin + 0];

    B32 match = a->partition == b->partition && a->cell_global == b->cell_global;
    halo_unique_begin += match;
  }

  // NOTE(cmat): Count duplicates on each lane.
  for (U64 it = halo_unique_begin; it < halo_total_range.max; ++it) {
    B32 unique = 0;
    if (it == 0) {
      unique = 1;
    } else {
      UG_Halo_Key *k1 = &halo_keys[it - 1];
      UG_Halo_Key *k2 = &halo_keys[it + 0];
      unique = (k1->partition != k2->partition) || (k1->cell_global != k2->cell_global);
    }

    if (unique) {
      halo_unique_count[lane_index()] += 1;
    }
  }

  // NOTE(cmat): Compute prefix sums.
  lane_barrier();

  U64 halo_unique_offset = 0;
  for Iter_Index(it, lane_index()) {
    halo_unique_offset += halo_unique_count[it];
  }

  // NOTE(cmat): Compute total unique halo count.
  lane_barrier();

  U64 halo_unique_count_global = 0;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      halo_unique_count_global += halo_unique_count[it];
    }
  }

  lane_broadcast_u64(&halo_unique_count_global, 0);
  log_info("Halo cell count: %'llu", halo_unique_count_global);

  // NOTE(cmat): Allocate ghost cells.
  lane_barrier();
  mesh->ghosts.len = ghost_total_count;
  if (lane_index() == 0) {
    mesh->ghosts.parent_cell   = arena_push_count(arena, U32, mesh->ghosts.len);
    mesh->ghosts.parent_face   = arena_push_count(arena, U08, mesh->ghosts.len);
    mesh->ghosts.marker_index  = arena_push_count(arena, U32, mesh->ghosts.len);
  }

  lane_broadcast_ptr(&mesh->ghosts.parent_cell,  0);
  lane_broadcast_ptr(&mesh->ghosts.parent_face,  0);
  lane_broadcast_ptr(&mesh->ghosts.marker_index, 0);

  // NOTE(cmat): Patch ghost faces and inner faces.
  U64 ghost_at = 0;
  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    for Iter_Index(it_face, 4) {
      U32  adjacent_global =  mesh->cells.faces[it_cell].adjacent[it_face];
      U32 *adjacent_local  = &mesh->cells.faces[it_cell].adjacent[it_face];

      if (adjacent_global >= mesh_global->cells.len) {
        // NOTE(cmat): Ghost.
        *adjacent_local = mesh->cells.len + halo_unique_count_global + ghost_offset + (ghost_at);
        mesh->ghosts.parent_cell  [ghost_offset + ghost_at] = it_cell;
        mesh->ghosts.parent_face  [ghost_offset + ghost_at] = it_face;
        mesh->ghosts.marker_index [ghost_offset + ghost_at] = mesh_global->ghosts.marker_index[adjacent_global - mesh_global->cells.len];
        ghost_at += 1;
      } else if (partition->cells_block_index[adjacent_global] != block_index) {
        // NOTE(cmat): Halo.
      } else {
        // NOTE(cmat): Inner.
        *adjacent_local = partition->cells_local_index[adjacent_global];
      }
    }
  }
  
  // NOTE(cmat): Allocate halo information.
  lane_barrier();
  mesh->halos.len       = halo_unique_count_global;
  mesh->halos.block_len = partition->blocks_len;

  U32 *halo_partition = 0;
  if (lane_index() == 0) {
    // NOTE(cmat): Allocations zero-ed by default, all ranges are empty, in [0, 0)
    mesh->halos.block_range = arena_push_count(arena,         Range1_U64, mesh->halos.block_len);
    mesh->halos.cell_global = arena_push_count(arena,         U32,        mesh->halos.len);
    halo_partition          = arena_push_count(scratch.arena, U32,        mesh->halos.len);
  }

  lane_broadcast_ptr(&mesh->halos.block_range,  0);
  lane_broadcast_ptr(&mesh->halos.cell_global,  0);
  lane_broadcast_ptr(&halo_partition,           0);

  // NOTE(cmat): Assign halo faces and sends.
  lane_barrier();

  // NOTE(cmat): Compute local starting halo index.
  U64 halo_unique_index = halo_unique_offset;

  if (lane_index() > 0) {
    UG_Halo_Key *k1 = &halo_keys[halo_total_range.min - 1];
    UG_Halo_Key *k2 = &halo_keys[halo_total_range.min + 0];
    B32 unique      = k1->partition != k2->partition || k1->cell_global != k2->cell_global;

    if (!unique) {
      halo_unique_index -= 1;
    }
  }

  // NOTE(cmat): Patch halo adjacency.
  for Iter_Range(it, halo_total_range) {
    B32 unique = 0;
  
    if (it == halo_total_range.min) {
      if (it == 0) {
        unique = 1;
      } else {
        UG_Halo_Key *k1 = &halo_keys[it - 1];
        UG_Halo_Key *k2 = &halo_keys[it + 0];
        unique          = k1->partition != k2->partition || k1->cell_global != k2->cell_global;
      }
    } else {
      UG_Halo_Key *k1    = &halo_keys[it - 1];
      UG_Halo_Key *k2    = &halo_keys[it + 0];
      unique             = k1->partition != k2->partition || k1->cell_global != k2->cell_global;
      halo_unique_index += unique;
    }

    UG_Halo_Key *key = &halo_keys[it];
    mesh->cells.faces[key->cell_local].adjacent[key->face_local] = mesh->cells.len + halo_unique_index;

    if (unique) {
      halo_partition          [halo_unique_index] = key->partition;   // NOTE(cmat): Halo index.
      mesh->halos.cell_global [halo_unique_index] = key->cell_global; // NOTE(cmat): Store global index for sends.
    }
  }

  // TODO(cmat): Sanity check, remove once stable.
  if (halo_unique_count[lane_index()] > 0) {
    U64 expected_end = halo_unique_offset + halo_unique_count[lane_index()] - 1;
    Assert(halo_unique_index == expected_end, "sanity check");
  }

  lane_barrier();
  if (lane_index() == 0) {
    for Iter_Index(it, mesh->halos.len) {
      U32         partition   = halo_partition[it];
      Range1_U64 *halo_range  = &mesh->halos.block_range[partition];

      // NOTE(cmat): First halo appearance, initialize.
      if (halo_range->min == halo_range->max) { halo_range->min = it; }
      halo_range->max = it + 1;
    }
  }
 
  // NOTE(cmat): Fill in sends.

  lane_barrier();
  scratch_end(&scratch);
  log_zone_end();
  profiler_end_function();
}

function void ug_mesh_array_init(UG_Mesh_Array *mesh_array, Arena *arena, U32 len) {
  Zero_Fill(mesh_array);

  mesh_array->len = len;
  if (lane_index() == 0) {
    mesh_array->dat = arena_push_count(arena, UG_Mesh, mesh_array->len);
  }

  lane_broadcast_ptr(&mesh_array->dat, 0);
}

function void ug_mesh_array_from_partition(UG_Mesh_Array *mesh_array, UG_Mesh *global_mesh, UG_Partition *partition, Range1_U64 partition_range, Arena *arena) {
  profiler_begin_function();
  log_zone_start("Constructing mesh array partitions for range [%'llu, %'llu)", partition_range.min, partition_range.max);

  for Iter_Range(it, partition_range) {
    Assert(it < mesh_array->len, "mesh array len overflow");
    ug_mesh_from_sub_mesh(&mesh_array->dat[it], global_mesh, partition, it, arena);
  }

  lane_barrier();
  log_zone_end();
  profiler_end_function();
}

function void ug_mesh_array_compute_sends(UG_Mesh_Array *mesh_array, struct UG_Partition *partition, Range1_U64 partition_range, Arena *arena) {
  profiler_begin_function();
  log_info("Computing mesh array sends for range [%'llu, %'llu)", partition_range.min, partition_range.max);

  for Iter_Range(block_index, partition_range) {
    UG_Mesh *mesh = &mesh_array->dat[block_index];

    // TODO(cmat): Multi lane.
    if (lane_index() == 0) {
      mesh->sends.block_len   = partition->blocks_len;
      mesh->sends.block_range = arena_push_count(arena, Range1_U64, partition->blocks_len);

      // NOTE(cmat): Compute send ranges.
      U64 send_at = 0;
      for Iter_Index(it_block, partition->blocks_len) {
        if (it_block != block_index) {
          UG_Mesh   *other_mesh = &mesh_array->dat[it_block];
          Range1_U64 range      = other_mesh->halos.block_range[block_index];
          U64        range_len  = range1_u64_len(range);

          mesh->sends.block_range[it_block] = range1_u64(send_at, send_at + range_len);
          send_at += range_len;
        } else {
          mesh->sends.block_range[it_block] = range1_u64(send_at, send_at);
        }
      }

      // NOTE(cmat): Allocate cell send.
      mesh->sends.len       = send_at;
      mesh->sends.cell_send = arena_push_count(arena, U32, send_at);
      log_info("Sends count: %'llu", mesh->sends.len);

      // NOTE(cmat): Fill cell send.
      send_at = 0;
      for Iter_Index(it_block, partition->blocks_len) {
        if (it_block != block_index) {
          UG_Mesh   *other_mesh = &mesh_array->dat[it_block];
          Range1_U64 range      = other_mesh->halos.block_range[block_index];
          U64        range_len  = range1_u64_len(range);

          for Iter_Index(it_cell, range_len) {
            U64 cell_global = other_mesh->halos.cell_global[range.min + it_cell];
            Assert(partition->cells_block_index[cell_global] == block_index, "invalid halo setup");
            U64 cell_local  = partition->cells_local_index[cell_global];

            mesh->sends.cell_send[send_at++] = cell_local;
          }
        }
      }

      Assert(send_at == mesh->sends.len, "invalid indexing");
    }
    
    lane_broadcast_type(&mesh->sends, 0);
  }

  profiler_end_function();
}

// ------------------------------------------------------------
// #-- IPC Commnuication

function void ug_mesh_ipc_distribute(UG_Mesh_Array *mesh_array) {
  profiler_begin_function();

  if (mesh_array->len > 1) {
    IPC_Sync_List sync_list = { };
    IPC_Sync_Scope(&sync_list) {
      log_info("Distributing mesh array to %u ranks", mesh_array->len - 1);
      for Iter_Index(it, (mesh_array->len - 1)) {
        U32 rank      = it + 1;
        UG_Mesh *mesh = mesh_array->dat + rank;
        
        // NOTE(cmat): Communicate lengths.
        ipc_rank_send(&sync_list, sizeof(UG_Mesh), mesh, rank, 0);
        
        // NOTE(cmat): UG_Cells
        ipc_rank_send(&sync_list, mesh->cells.len * sizeof(V3F),              mesh->cells.center,         rank, 0);
        ipc_rank_send(&sync_list, mesh->cells.len * sizeof(F32),              mesh->cells.volume,         rank, 0);
        ipc_rank_send(&sync_list, mesh->cells.len * sizeof(UG_Cell_Faces),    mesh->cells.faces,          rank, 0);
        
        // NOTE(cmat): UG_Halos
        ipc_rank_send(&sync_list, mesh->halos.block_len * sizeof(Range1_U64), mesh->halos.block_range,    rank, 0);
        ipc_rank_send(&sync_list, mesh->halos.len       * sizeof(U32),        mesh->halos.cell_global,    rank, 0);

        // NOTE(cmat): UG_Sends
        ipc_rank_send(&sync_list, mesh->sends.block_len   * sizeof(Range1_U64), mesh->sends.block_range,    rank, 0);
        ipc_rank_send(&sync_list, mesh->sends.len         * sizeof(U32),        mesh->sends.cell_send,      rank, 0);

        // NOTE(cmat): UG_Ghosts
        ipc_rank_send(&sync_list, mesh->ghosts.len * sizeof(U32),             mesh->ghosts.parent_cell,   rank, 0);
        ipc_rank_send(&sync_list, mesh->ghosts.len * sizeof(U08),             mesh->ghosts.parent_face,   rank, 0);
        ipc_rank_send(&sync_list, mesh->ghosts.len * sizeof(U32),             mesh->ghosts.marker_index,  rank, 0);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void ug_mesh_ipc_receive(Arena *arena, UG_Mesh *mesh, U32 rank) {
  profiler_begin_function();
  IPC_Sync_List sync_list = { };
  IPC_Sync_Scope(&sync_list) {

    // NOTE(cmat): Receive lengths, allocate.
    ipc_rank_receive(&sync_list, sizeof(UG_Mesh), mesh, rank, 0);
  }

  // NOTE(cmat): UG_Cells
  if (lane_index() == 0) {
    mesh->cells.center        = arena_push_count(arena, V3F,            mesh->cells.len);
    mesh->cells.volume        = arena_push_count(arena, F32,            mesh->cells.len);
    mesh->cells.faces         = arena_push_count(arena, UG_Cell_Faces,  mesh->cells.len);

    // NOTE(cmat): UG_Halos
    mesh->halos.block_range   = arena_push_count(arena, Range1_U64,     mesh->halos.block_len);
    mesh->halos.cell_global   = arena_push_count(arena, U32,            mesh->halos.len);

    // NOTE(cmat): UG_Sends
    mesh->sends.block_range   = arena_push_count(arena, Range1_U64,     mesh->sends.block_len);
    mesh->sends.cell_send     = arena_push_count(arena, U32,            mesh->sends.len);

    // NOTE(cmat): UG_Ghosts
    mesh->ghosts.parent_cell  = arena_push_count(arena, U32,            mesh->ghosts.len);
    mesh->ghosts.parent_face  = arena_push_count(arena, U08,            mesh->ghosts.len);
    mesh->ghosts.marker_index = arena_push_count(arena, U32,            mesh->ghosts.len);
  }

  // NOTE(cmat): Synchronize mesh.
  lane_broadcast_type(mesh, 0);

  IPC_Sync_Scope(&sync_list) {
    // NOTE(cmat): UG_Cells
    ipc_rank_receive(&sync_list, mesh->cells.len * sizeof(V3F),              mesh->cells.center,         rank, 0);
    ipc_rank_receive(&sync_list, mesh->cells.len * sizeof(F32),              mesh->cells.volume,         rank, 0);
    ipc_rank_receive(&sync_list, mesh->cells.len * sizeof(UG_Cell_Faces),    mesh->cells.faces,          rank, 0);
    
    // NOTE(cmat): UG_Halos
    ipc_rank_receive(&sync_list, mesh->halos.block_len * sizeof(Range1_U64), mesh->halos.block_range,    rank, 0);
    ipc_rank_receive(&sync_list, mesh->halos.len       * sizeof(U32),        mesh->halos.cell_global,    rank, 0);

    // NOTE(cmat): UG_Sends
    ipc_rank_receive(&sync_list, mesh->sends.block_len * sizeof(Range1_U64), mesh->sends.block_range,    rank, 0);
    ipc_rank_receive(&sync_list, mesh->sends.len       * sizeof(U32),        mesh->sends.cell_send,      rank, 0);

    // NOTE(cmat): UG_Ghosts
    ipc_rank_receive(&sync_list, mesh->ghosts.len * sizeof(U32),             mesh->ghosts.parent_cell,   rank, 0);
    ipc_rank_receive(&sync_list, mesh->ghosts.len * sizeof(U08),             mesh->ghosts.parent_face,   rank, 0);
    ipc_rank_receive(&sync_list, mesh->ghosts.len * sizeof(U32),             mesh->ghosts.marker_index,  rank, 0);
  }

  lane_barrier();
  profiler_end_function();
}
