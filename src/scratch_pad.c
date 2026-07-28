// ================================================================
// #-- Mesh Partitioning: Zoltan (RCB + Graph)
// ================================================================
//
// NOTE(cmat): Mirrors the interface of ug_partition_rcb, but delegates the
// - actual decomposition to Zoltan. RCB and graph partitioning share the
// - same object query callbacks; graph partitioning additionally registers
// - edge callbacks built from cell face adjacency.
//
// NOTE(cmat): Runs on a single lane-group (lane_index() == 0) against the
// - full global mesh, same as ug_partition_rcb -- Zoltan is therefore driven
// - on MPI_COMM_SELF. If this ever needs to run collectively across ranks
// - (e.g. re-partitioning an already-distributed mesh), swap MPI_COMM_SELF
// - for the real communicator and have the callbacks report only
// - locally-owned cells instead of the whole mesh.
//
// NOTE(cmat): Verify these callback signatures against your installed
// - zoltan.h -- the shape below matches the stable Zoltan C API, but a few
// - fields (e.g. exact float vs double weight types) can shift slightly
// - between Zoltan versions/build configs.

#include <stdio.h>
#include <mpi.h>
#include <zoltan.h>

typedef struct UG_Zoltan_Context {
  UG_Mesh *mesh;
} UG_Zoltan_Context;

// ------------------------------------------------------------
// #-- Shared query callbacks

static int ug_zoltan_num_obj_fn(void *data, int *ierr) {
  UG_Zoltan_Context *ctx = (UG_Zoltan_Context *)data;
  *ierr = ZOLTAN_OK;
  return (int)ctx->mesh->cells.len;
}

static void ug_zoltan_obj_list_fn(void *data, int num_gid_entries, int num_lid_entries,
                                   ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                   int wgt_dim, float *obj_wgts, int *ierr) {
  UG_Zoltan_Context *ctx = (UG_Zoltan_Context *)data;
  U64                len = ctx->mesh->cells.len;

  for (U64 it = 0; it < len; it += 1) {
    global_ids[it * num_gid_entries] = (ZOLTAN_ID_TYPE)it;
    if (num_lid_entries) { local_ids[it * num_lid_entries] = (ZOLTAN_ID_TYPE)it; }
    if (wgt_dim > 0)     { obj_wgts[it * wgt_dim] = 1.f; } // NOTE(cmat): Uniform weight; swap for cells.volume[it] to balance by cell size instead.
  }

  *ierr = ZOLTAN_OK;
}

// ------------------------------------------------------------
// #-- RCB (geometric) callbacks

static int ug_zoltan_num_geom_fn(void *data, int *ierr) {
  *ierr = ZOLTAN_OK;
  return 3;
}

static void ug_zoltan_geom_multi_fn(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                     ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                     int num_dim, double *geom_vec, int *ierr) {
  UG_Zoltan_Context *ctx = (UG_Zoltan_Context *)data;

  for (int it = 0; it < num_obj; it += 1) {
    U32 cell   = (U32)global_ids[it * num_gid_entries];
    V3F center = ctx->mesh->cells.center[cell];

    geom_vec[it * 3 + 0] = (double)center.x;
    geom_vec[it * 3 + 1] = (double)center.y;
    geom_vec[it * 3 + 2] = (double)center.z;
  }

  *ierr = ZOLTAN_OK;
}

// ------------------------------------------------------------
// #-- Graph callbacks
//
// NOTE(cmat): Edges are built from cell face adjacency. Only "inner"
// - adjacency (adjacent < cells.len) counts as a graph edge -- ghost /
// - boundary faces have no counterpart cell and are skipped, matching the
// - same `adjacent < cells.len` convention used throughout ug_mesh_*.

static void ug_zoltan_num_edges_multi_fn(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                          int *num_edges, int *ierr) {
  UG_Zoltan_Context *ctx = (UG_Zoltan_Context *)data;

  for (int it = 0; it < num_obj; it += 1) {
    U32            cell  = (U32)global_ids[it * num_gid_entries];
    UG_Cell_Faces *faces = &ctx->mesh->cells.faces[cell];

    int count = 0;
    for Iter_Index(face, 4) {
      count += (faces->adjacent[face] < ctx->mesh->cells.len);
    }

    num_edges[it] = count;
  }

  *ierr = ZOLTAN_OK;
}

static void ug_zoltan_edge_list_multi_fn(void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                          int *num_edges, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
                                          int wgt_dim, float *ewgts, int *ierr) {
  UG_Zoltan_Context *ctx     = (UG_Zoltan_Context *)data;
  U64                edge_at = 0;

  for (int it = 0; it < num_obj; it += 1) {
    U32            cell  = (U32)global_ids[it * num_gid_entries];
    UG_Cell_Faces *faces = &ctx->mesh->cells.faces[cell];

    for Iter_Index(face, 4) {
      U32 adjacent = faces->adjacent[face];
      if (adjacent < ctx->mesh->cells.len) {
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
// #-- Shared setup / result gathering

static void ug_zoltan_initialize_once(void) {
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
static void ug_partition_zoltan_finalize(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count,
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
// #-- RCB entry point

function void ug_partition_zoltan_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan RCB");

  partition->blocks_len = partition_count;

  if (lane_index() == 0) {
    ug_zoltan_initialize_once();

    struct Zoltan_Struct *zz  = Zoltan_Create(MPI_COMM_SELF);
    UG_Zoltan_Context     ctx = { .mesh = mesh };

    Zoltan_Set_Param(zz, "LB_METHOD",        "RCB");
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM",   "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS",     "PARTS");
    Zoltan_Set_Param(zz, "DEBUG_LEVEL",      "0");

    char partition_count_str[32];
    snprintf(partition_count_str, sizeof(partition_count_str), "%u", partition_count);
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", partition_count_str);

    Zoltan_Set_Num_Obj_Fn    (zz, ug_zoltan_num_obj_fn,    &ctx);
    Zoltan_Set_Obj_List_Fn   (zz, ug_zoltan_obj_list_fn,   &ctx);
    Zoltan_Set_Num_Geom_Fn   (zz, ug_zoltan_num_geom_fn,   &ctx);
    Zoltan_Set_Geom_Multi_Fn (zz, ug_zoltan_geom_multi_fn, &ctx);

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

// ------------------------------------------------------------
// #-- Graph entry point

function void ug_partition_zoltan_graph(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  log_zone_start("Partitioning mesh: Zoltan Graph");

  partition->blocks_len = partition_count;

  if (lane_index() == 0) {
    ug_zoltan_initialize_once();

    struct Zoltan_Struct *zz  = Zoltan_Create(MPI_COMM_SELF);
    UG_Zoltan_Context     ctx = { .mesh = mesh };

    Zoltan_Set_Param(zz, "LB_METHOD",        "GRAPH");
    Zoltan_Set_Param(zz, "GRAPH_PACKAGE",    "PARMETIS"); // NOTE(cmat): Switch to "PHG" if ParMETIS isn't linked in.
    Zoltan_Set_Param(zz, "NUM_GID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "NUM_LID_ENTRIES",  "1");
    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM",   "1");
    Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM",  "1");
    Zoltan_Set_Param(zz, "RETURN_LISTS",     "PARTS");
    Zoltan_Set_Param(zz, "DEBUG_LEVEL",      "0");

    char partition_count_str[32];
    snprintf(partition_count_str, sizeof(partition_count_str), "%u", partition_count);
    Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", partition_count_str);

    Zoltan_Set_Num_Obj_Fn         (zz, ug_zoltan_num_obj_fn,         &ctx);
    Zoltan_Set_Obj_List_Fn        (zz, ug_zoltan_obj_list_fn,        &ctx);
    Zoltan_Set_Num_Edges_Multi_Fn (zz, ug_zoltan_num_edges_multi_fn, &ctx);
    Zoltan_Set_Edge_List_Multi_Fn (zz, ug_zoltan_edge_list_multi_fn, &ctx);

    int           changes, num_gid_entries, num_lid_entries;
    int           num_import, num_export;
    ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
    int          *import_procs, *import_to_part, *export_procs, *export_to_part;

    int result = Zoltan_LB_Partition(zz, &changes, &num_gid_entries, &num_lid_entries,
                                      &num_import, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                      &num_export, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);

    Assert(result == ZOLTAN_OK, "Zoltan graph partition failed");
    log_info("Zoltan graph produced %u blocks from %d exported cells", partition_count, num_export);

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

// ------------------------------------------------------------
// #-- Header declarations to add alongside ug_partition_rcb:
//
// function void ug_partition_zoltan_rcb   (UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);
// function void ug_partition_zoltan_graph (UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);
