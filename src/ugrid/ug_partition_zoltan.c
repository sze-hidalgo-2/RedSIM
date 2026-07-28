// ------------------------------------------------------------
// #-- Zoltan General Callbacks

// NOTE(cmat): Callback for the number of cells in mesh.
function I32 ug_zoltan_callback_num_obj_fn(void *user_ptr, I32 *ierr) {
  UG_Mesh *mesh           = (UG_Mesh *)user_ptr;
  *ierr                   = ZOLTAN_OK;
  I32 result              = (I32)mesh->cells.len;
  return result;
}

function void ug_zoltan_callback_obj_list_fn(void *user_ptr, I32 num_grid_entries, I32 num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, I32 wgt_dim, F32 *obj_wgts, I32 *ierr) {
  UG_Mesh *mesh = (UG_Mesh *)user_ptr;
  for Iter_Index(it, mesh->cells.len) {
    global_ids[it * num_grid_entries] = (ZOLTAN_ID_TYPE)it;
    if (num_lid_entries) {
      local_ids[it * num_lid_entries] = (ZOLTAN_ID_TYPE)it;
    }

    // NOTE(cmat): Uniform weights everywhere
    if (wgt_dim > 0) {
      obj_wgts[it * wgt_dim] = 1.f;
    }
  }

  *ierr = ZOLTAN_OK;
}

// ------------------------------------------------------------
// #-- Zoltan Geometric Callbacks

function I32 ug_zoltan_callback_num_geom_fn(void *user_ptr, I32 *ierr) {
  *ierr = ZOLTAN_OK;
  return 3;
}

function void ug_zoltan_callback_geom_multi_fn(void *user_ptr, I32 num_gid_entries, I32 num_lid_entries, I32 num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, I32 num_dim, F64 *geom_vec, I32 *ierr) {
  UG_Mesh *mesh = (UG_Mesh *)user_ptr;

  for Iter_Index(it, num_obj) {
    U32 cell   = (U32)global_ids[it * num_gid_entries];
    V3F center = mesh->cells.center[cell];

    geom_vec[it * 3 + 0] = (F64)center.x;
    geom_vec[it * 3 + 1] = (F64)center.y;
    geom_vec[it * 3 + 2] = (F64)center.z;
  }

  *ierr = ZOLTAN_OK;
}

function void ug_zoltan_initialize_once(void) {
  static B32 initialized = 0;
  if (!initialized) {
    F32    version = 0;
    int    argc    = 0;
    char **argv    = 0;
    Zoltan_Initialize(argc, argv, &version);
    initialized = 1;
  }
}

// NOTE(cmat): Shared by both partitioners -- scatters Zoltan's flat
// - (cell, target_block) export list into the same UG_Partition layout
// - ug_partition_rcb_split builds (blocks_dat / cells_block_index / cells_local_index).
function void ug_partition_zoltan_finalize(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count,
                                          int num_export, ZOLTAN_ID_PTR export_global_ids, int *export_to_part) {
  Arena_Temp scratch = scratch_start(arena);

  partition->blocks_dat         = arena_push_count(arena, UG_Partition_Block, partition_count);
  partition->cells_block_index  = arena_push_count(arena, U32,                mesh->cells.len);
  partition->cells_local_index  = arena_push_count(arena, U32,                mesh->cells.len);

  // NOTE(cmat): Count cells assigned to each block.
  U64 *block_counts = arena_push_count(scratch.arena, U64, partition_count);
  for (int it = 0; it < num_export; it += 1) {
    U32 block = (U32)export_to_part[it];
    block_counts[block] += 1;
  }

  for Iter_Index(block, partition_count) {
    partition->blocks_dat[block].cells_len = block_counts[block];
    partition->blocks_dat[block].cells_dat = arena_push_count(arena, U32, block_counts[block]);
  }

  // NOTE(cmat): Scatter cells into their block, filling both index maps.
  U64 *block_write_at = arena_push_count(scratch.arena, U64, partition_count);
  for (int it = 0; it < num_export; it += 1) {
    U32 cell  = (U32)export_global_ids[it];
    U32 block = (U32)export_to_part[it];
    U32 local = (U32)(block_write_at[block]++);

    partition->blocks_dat[block].cells_dat[local] = cell;
    partition->cells_block_index[cell]            = block;
    partition->cells_local_index[cell]            = local;
  }

  scratch_end(&scratch);
}

// ------------------------------------------------------------
// #-- Graph callbacks
//
// NOTE(cmat): Edges are built from cell face adjacency. Only "inner"
// - adjacency (adjacent < cells.len) counts as a graph edge -- ghost /
// - boundary faces have no counterpart cell and are skipped, matching the
// - same `adjacent < cells.len` convention used throughout ug_mesh_*.

static void ug_zoltan_callback_num_edges_multi_fn(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                          int *num_edges, int *ierr) {
  UG_Mesh *mesh = (UG_Mesh *)data;

  for (int it = 0; it < num_obj; it += 1) {
    U32            cell  = (U32)global_ids[it * num_gid_entries];
    UG_Cell_Faces *faces = &mesh->cells.faces[cell];

    int count = 0;
    for Iter_Index(face, 4) {
      count += (faces->adjacent[face] < mesh->cells.len);
    }

    num_edges[it] = count;
  }

  *ierr = ZOLTAN_OK;
}

static void ug_zoltan_callback_edge_list_multi_fn(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                          int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
                                          int wgt_dim, float *ewgts, int *ierr) {
  UG_Mesh *mesh = (UG_Mesh *)data;
  U64                edge_at = 0;

  for (int it = 0; it < num_obj; it += 1) {
    U32            cell  = (U32)global_ids[it * num_gid_entries];
    UG_Cell_Faces *faces = &mesh->cells.faces[cell];

    for Iter_Index(face, 4) {
      U32 adjacent = faces->adjacent[face];
      if (adjacent < mesh->cells.len) {
        nbor_global_id[edge_at * num_gid_entries] = (ZOLTAN_ID_TYPE)adjacent;
        nbor_procs[edge_at]                       = 0; // NOTE(cmat): Serial partition -- every cell "lives" on rank 0.
        if (wgt_dim > 0) { ewgts[edge_at * wgt_dim] = 1.f; }
        edge_at += 1;
      }
    }
  }

  *ierr = ZOLTAN_OK;
}



// ------------------------------------------------------------
// #-- Partitioning Implementation

function void ug_partition_zoltan_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan RCB");

  partition->blocks_len = partition_count;

  if (lane_index() == 0) {
    ug_zoltan_initialize_once();

    // REFACTORED(cmat): Use MPI_COMM_WORLD instead of MPI_COMM_SELF, so that
    // - Zoltan_LB_Partition is a genuine collective call across every MPI rank,
    // - not a serial call confined to rank 0. Callers must now invoke this
    // - function on *every* rank; only rank 0 is expected to pass a mesh with
    // - cells.len > 0, every other rank should pass a zeroed UG_Mesh, which
    // - contributes zero objects to the partitioning problem but still lets
    // - the rank participate correctly in the collective call.
    struct Zoltan_Struct *zz  = Zoltan_Create(MPI_COMM_WORLD);

    Zoltan_Set_Param(zz, "LB_METHOD",        "RCB");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM",   "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS",     "PARTS");
    Zoltan_Set_Param(zz, "DEBUG_LEVEL",      "0");

    char partition_count_str[32];
    snprintf(partition_count_str, sizeof(partition_count_str), "%u", partition_count);
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", partition_count_str);

    Zoltan_Set_Num_Obj_Fn    (zz, ug_zoltan_callback_num_obj_fn,    mesh);
    Zoltan_Set_Obj_List_Fn   (zz, ug_zoltan_callback_obj_list_fn,   mesh);
    Zoltan_Set_Num_Geom_Fn   (zz, ug_zoltan_callback_num_geom_fn,   mesh);
    Zoltan_Set_Geom_Multi_Fn (zz, ug_zoltan_callback_geom_multi_fn, mesh);

    int           changes, num_gid_entries, num_lid_entries;
    int           num_import, num_export;
    ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
    int          *import_procs, *import_to_part, *export_procs, *export_to_part;

    int result = Zoltan_LB_Partition(zz, &changes, &num_gid_entries, &num_lid_entries,
                                      &num_import, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                      &num_export, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);

    Assert(result == ZOLTAN_OK, "Zoltan RCB partition failed");
    log_info("Zoltan RCB produced %u blocks from %d exported cells", partition_count, num_export);

    // NOTE(cmat): num_export/export_to_part is only meaningful on the rank(s) that
    // - actually own cells (rank 0, in our setup) -- other ranks will see num_export == 0,
    // - so finalize() is a safe no-op for them. Finalize/mesh-array construction/IPC
    // - distribution stay rank-0-only, as before.
    ug_partition_zoltan_finalize(partition, arena, mesh, partition_count, num_export, export_global_ids, export_to_part);

    Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
    Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
    Zoltan_Destroy(&zz);
  }

  lane_broadcast_ptr(&partition->blocks_dat,         0);
  lane_broadcast_ptr(&partition->cells_block_index,  0);
  lane_broadcast_ptr(&partition->cells_local_index,  0);
  lane_barrier();

  log_zone_end();
  profiler_end_function();
}

function void ug_partition_zoltan_graph(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan Graph");

  partition->blocks_len = partition_count;

  if (lane_index() == 0) {
    ug_zoltan_initialize_once();

    // REFACTORED(cmat): MPI_COMM_WORLD (was MPI_COMM_SELF) -- see ug_partition_zoltan_rcb
    // - above for the full rationale. This is what actually lets the PHG/ParMETIS graph
    // - package run its parallel algorithm across ranks, instead of executing serially
    // - inside whichever single process happened to call it.
    struct Zoltan_Struct *zz  = Zoltan_Create(MPI_COMM_WORLD);

    Zoltan_Set_Param(zz, "LB_METHOD",        "GRAPH");
    Zoltan_Set_Param(zz, "GRAPH_PACKAGE",    "PHG"); // NOTE(cmat): Switch to "PHG" if ParMETIS isn't linked in.
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM",   "1");
    Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM",  "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS",     "PARTS");
    Zoltan_Set_Param(zz, "DEBUG_LEVEL",      "0");

    char partition_count_str[32];
    snprintf(partition_count_str, sizeof(partition_count_str), "%u", partition_count);
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", partition_count_str);

    Zoltan_Set_Num_Obj_Fn         (zz, ug_zoltan_callback_num_obj_fn,         mesh);
    Zoltan_Set_Obj_List_Fn        (zz, ug_zoltan_callback_obj_list_fn,        mesh);
    Zoltan_Set_Num_Edges_Multi_Fn (zz, ug_zoltan_callback_num_edges_multi_fn, mesh);
    Zoltan_Set_Edge_List_Multi_Fn (zz, ug_zoltan_callback_edge_list_multi_fn, mesh);

    int           changes, num_gid_entries, num_lid_entries;
    int           num_import, num_export;
    ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
    int          *import_procs, *import_to_part, *export_procs, *export_to_part;

    int result = Zoltan_LB_Partition(zz, &changes, &num_gid_entries, &num_lid_entries,
                                      &num_import, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                      &num_export, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);

    Assert(result == ZOLTAN_OK, "Zoltan graph partition failed");
    log_info("Zoltan graph produced %u blocks from %d exported cells", partition_count, num_export);

    // NOTE(cmat): See ug_partition_zoltan_rcb -- num_export == 0 on every rank that
    // - contributed no cells, so finalize is a rank-0-only no-op elsewhere.
    ug_partition_zoltan_finalize(partition, arena, mesh, partition_count, num_export, export_global_ids, export_to_part);

    Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
    Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
    Zoltan_Destroy(&zz);
  }

  lane_broadcast_ptr(&partition->blocks_dat,         0);
  lane_broadcast_ptr(&partition->cells_block_index,  0);
  lane_broadcast_ptr(&partition->cells_local_index,  0);
  lane_barrier();

  log_zone_end();
  profiler_end_function();
}
