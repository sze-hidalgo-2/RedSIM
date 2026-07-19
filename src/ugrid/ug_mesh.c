// ------------------------------------------------------------
// #-- Initialization

function void ug_mesh_init(UG_Mesh *mesh) {
  Zero_Fill(mesh);
  arena_init(&mesh->arena);
}

// ------------------------------------------------------------
// #-- Compute Cell Faces

#pragma pack(push, 1)
  typedef struct UG_Tetra_Face {
    V3U verts;  // NOTE(cmat): Sorted vertex indices
    U32 cell;   // NOTE(cmat): Cell index.
    U32 face;   // NOTE(cmat): Face index (0, 1, 2 or 3). Padded to 32 bit boundaty.
  } UG_Tetra_Face;
#pragma pack(pop)

Assert_Compiler(sizeof(UG_Tetra_Face) == 5 * sizeof(U32));

function B32 ug_tetra_face_match(UG_Tetra_Face *lhs, UG_Tetra_Face *rhs) {
  B32 match = memory_match(&lhs->verts, &rhs->verts, sizeof(V3U)); // NOTE(cmat): Only check vx, vy, vz.
  return match;
}

function void ug_mesh_compute_cells_faces(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Computing faces");

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
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    
    // NOTE(cmat): Sort cell vertices, so that each face is unique in its description.
    V4U *cell = &mesh->cells.verts[it];
    v4_u32_sort(cell);

    U64 face_index                = 4 * it;
    faces_all_dat[face_index + 0] = (UG_Tetra_Face) { .verts = v3u(cell->x, cell->y, cell->z), .cell = it, .face = 0 };
    faces_all_dat[face_index + 1] = (UG_Tetra_Face) { .verts = v3u(cell->x, cell->y, cell->w), .cell = it, .face = 1 };
    faces_all_dat[face_index + 2] = (UG_Tetra_Face) { .verts = v3u(cell->x, cell->z, cell->w), .cell = it, .face = 2 };
    faces_all_dat[face_index + 3] = (UG_Tetra_Face) { .verts = v3u(cell->y, cell->z, cell->w), .cell = it, .face = 3 };
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
  
  // NOTE(cmat): Compute face information for each cell.
  if (lane_index() == 0) {
    mesh->cells.faces         = arena_push_count(&mesh->arena, UG_Cell_Faces,       mesh->cells.len);
    mesh->cells.faces_verts   = arena_push_count(&mesh->arena, UG_Cell_Faces_Verts, mesh->cells.len);
    mesh->ghosts.parent_cell  = arena_push_count(&mesh->arena, U32,                 mesh->ghosts.len);
    mesh->ghosts.parent_face  = arena_push_count(&mesh->arena, U08,                 mesh->ghosts.len);
    mesh->ghosts.marker_index = arena_push_count(&mesh->arena, U32,                 mesh->ghosts.len);
  }

  lane_broadcast_ptr(&mesh->cells.faces,          0);
  lane_broadcast_ptr(&mesh->ghosts.parent_cell,   0);
  lane_broadcast_ptr(&mesh->ghosts.parent_face,   0);
  lane_broadcast_ptr(&mesh->ghosts.marker_index,  0);

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
   
    V3U face_verts = faces_all_dat[faces_it + 0].verts;
    U32 lhs_cell_index = faces_all_dat[faces_it + 0].cell;
    U32 lhs_face_index = faces_all_dat[faces_it + 0].face;
    U32 rhs_cell_index = 0;

    // NOTE(cmat): Compute face area and normal.
    V3F x0    = v3f(mesh->verts.x[face_verts.dat[0]], mesh->verts.y[face_verts.dat[0]], mesh->verts.z[face_verts.dat[0]]);
    V3F x1    = v3f(mesh->verts.x[face_verts.dat[1]], mesh->verts.y[face_verts.dat[1]], mesh->verts.z[face_verts.dat[1]]);
    V3F x2    = v3f(mesh->verts.x[face_verts.dat[2]], mesh->verts.y[face_verts.dat[2]], mesh->verts.z[face_verts.dat[2]]);
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
    v3_u32_sort(&face_verts);

    UG_Cell_Faces_Verts *lhs_faces_verts    = mesh->cells.faces_verts + lhs_cell_index;
    lhs_faces_verts->verts[lhs_face_index]  = face_verts;

    UG_Cell_Faces       *lhs_faces        = mesh->cells.faces       + lhs_cell_index;
    lhs_faces->adjacent [lhs_face_index]  = rhs_cell_index;
    lhs_faces->area     [lhs_face_index]  = area;
    lhs_faces->normal_x [lhs_face_index]  = n.x;
    lhs_faces->normal_y [lhs_face_index]  = n.y;
    lhs_faces->normal_z [lhs_face_index]  = n.z;

    if (match) {
      U32 rhs_face_index = faces_all_dat[faces_it + 1].face;

      UG_Cell_Faces_Verts *rhs_faces_verts    = mesh->cells.faces_verts + rhs_cell_index;
      rhs_faces_verts->verts[rhs_face_index]  = face_verts;

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

  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Assign Ghost Cells.

#pragma pack(push, 1)
typedef struct UG_Ghost_Face {
  V3U verts;
  U32 type;
  union { U32 ghost_index, marker_index; };
} UG_Ghost_Face;

#pragma pack(pop)

Assert_Compiler(sizeof(UG_Ghost_Face) == 5 * sizeof(U32));

function B32 ug_ghost_face_match(UG_Ghost_Face *lhs, UG_Ghost_Face *rhs) {
  B32 match = memory_match(&lhs->verts, &rhs->verts, sizeof(V3U)); // NOTE(cmat): Only check vx, vy, vz.
  return match;
}

function void ug_mesh_assign_ghosts(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Assigning marks to ghost cells");

  U64            ghost_faces_all_len  = 0;
  UG_Ghost_Face *ghost_faces_all      = 0;

  if (lane_index() == 0) {
    U64 markers_len = 0;
    for Iter_Index(it, mesh->markers.len) {
      markers_len += mesh->markers.elems[it].len;
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
    V3U parent_verts = mesh->cells.faces_verts[parent_cell].verts[parent_face];

    // NOTE(cmat): parent_verts is already sorted
    ghost_faces_all[it] = (UG_Ghost_Face) { .verts = parent_verts, .type = 0, .ghost_index = it };
  }

  // NOTE(cmat): Add marker faces.
  log_info("Adding marker faces");
  for Iter_Range(marker_it, lane_range(mesh->markers.len)) {
    UG_Marker_Elems *elems = &mesh->markers.elems[marker_it];
    U64 write_index        = mesh->ghosts.len;

    for Iter_Index(it, marker_it) {
      write_index += mesh->markers.elems[it].len;
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
// #-- Compute Cells.

function void ug_mesh_compute_cells(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Computing cells");

  Range3_F32 *bounds_global = 0;
  if (lane_index() == 0) {
    mesh->cells.center = arena_push_count(&mesh->arena, V3F,         mesh->cells.len);
    mesh->cells.volume = arena_push_count(&mesh->arena, F32,         mesh->cells.len);
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
    V4U v    = mesh->cells.verts[it];
    V3F a    = v3f(mesh->verts.x[v.x], mesh->verts.y[v.x], mesh->verts.z[v.x]);
    V3F b    = v3f(mesh->verts.x[v.y], mesh->verts.y[v.y], mesh->verts.z[v.y]);
    V3F c    = v3f(mesh->verts.x[v.z], mesh->verts.y[v.z], mesh->verts.z[v.z]);
    V3F d    = v3f(mesh->verts.x[v.w], mesh->verts.y[v.w], mesh->verts.z[v.w]);

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

  lane_barrier();

  log_info("Mesh bounds: (%.2g, %.2g, %.2g), (%.2g, %.2g, %.2g)", V3_Expand(mesh->bounds.min), V3_Expand(mesh->bounds.max));

  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Reorder Cells.

function void ug_mesh_reorder_cells(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Reodering cells");

  // NOTE(cmat): Compute morton codes for each cell center
  log_info("Computing morton codes");

  V2_U64 *morton_codes = 0;
  if (lane_index() == 0) {
    morton_codes = arena_push_count(scratch.arena, V2_U64, mesh->cells.len);
  }

  lane_broadcast_ptr(&morton_codes, 0);

  V3F bounds_len_rcp = v3f_rcp(range3_f32_len(mesh->bounds));
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    V3F center_normed = v3f_had(v3f_sub(mesh->cells.center[it], mesh->bounds.min), bounds_len_rcp);

    V2_U64 *entry     = morton_codes + it;
    entry->x          = morton64_encode_v3f(center_normed);
    entry->y          = it;
  }

  // NOTE(cmat): Sort morton codes.
  lane_barrier();
  log_info("Sorting cells by morton code");
  array_sort_radix_u64(mesh->cells.len, 2, 0, (U64 *)morton_codes);

  // NOTE(cmat): Reorder cell data using the order of the sorted morton codes.
  lane_barrier();

  // NOTE(cmat): Convert index references first.
  array_reorder(mesh->cells.len, sizeof(V4U), (U08 *)mesh->cells.verts,  sizeof(V2_U64), &morton_codes->y);
  array_reorder(mesh->cells.len, sizeof(V3F), (U08 *)mesh->cells.center, sizeof(V2_U64), &morton_codes->y);
  array_reorder(mesh->cells.len, sizeof(F32), (U08 *)mesh->cells.volume, sizeof(V2_U64), &morton_codes->y);

  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}
