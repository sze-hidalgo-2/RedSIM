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
// #-- Partitioning Implementation

function void ug_partition_zoltan_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan RCB");

  partition->blocks_len = partition_count;

  if (lane_index() == 0) {
    ug_zoltan_initialize_once();

    struct Zoltan_Struct *zz  = Zoltan_Create(MPI_COMM_SELF);

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

