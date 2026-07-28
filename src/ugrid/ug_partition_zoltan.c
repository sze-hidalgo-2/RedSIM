// ------------------------------------------------------------
// #-- Zoltan General Callbacks

typedef struct UG_Zoltan_Context {
  UG_Mesh *mesh;
} UG_Zoltan_Context;

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

    // NOTE(cmat): Uniform weights everywhere. // NOTE(cmat): Uniform weights everywhere.
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

function void ug_zoltan_geom_multi_fn(void *user_ptr, I32 num_gid_entries, I32 num_lid_entries, I32 num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, I32 num_dim, F64 *geom_vec, I32 *ierr) {
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

// ------------------------------------------------------------
// #-- Partitioning Implementation


function void ug_partition_zoltan_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan RCB");

  log_zone_end();
  profiler_end_function();
}
