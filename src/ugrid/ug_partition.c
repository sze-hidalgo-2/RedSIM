#pragma pack(push, 1)
typedef struct UG_Partition_RCB_Key {
  V3U center;
  U32 cell;
} UG_Partition_RCB_Key;
#pragma pack(pop)

Assert_Compiler(sizeof(UG_Partition_RCB_Key) == 4 * sizeof(U32));


function void
ug_partition_rcb_child_bounds(Range1_U64 left_range, Range1_U64 right_range, UG_Partition_RCB_Key *rcb_keys, Range3_F32 *left_bounds, Range3_F32 *right_bounds) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  Range3_F32 *left_bounds_global  = 0;
  Range3_F32 *right_bounds_global = 0;
  if (lane_index() == 0){ 
    left_bounds_global  = arena_push_count(scratch.arena, Range3_F32, lane_count());
    right_bounds_global = arena_push_count(scratch.arena, Range3_F32, lane_count());
  }

  lane_broadcast_ptr(&left_bounds_global,   0);
  lane_broadcast_ptr(&right_bounds_global,  0);

  Range3_F32 *left_bounds_local   = left_bounds_global  + lane_index();
  Range3_F32 *right_bounds_local  = right_bounds_global + lane_index();

  left_bounds_local->min  = v3f(f32_limit_max, f32_limit_max, f32_limit_max);
  left_bounds_local->max  = v3f(f32_limit_min, f32_limit_min, f32_limit_min);
  right_bounds_local->min = v3f(f32_limit_max, f32_limit_max, f32_limit_max);
  right_bounds_local->max = v3f(f32_limit_min, f32_limit_min, f32_limit_min);

  for Iter_Range(it, lane_range(range1_u64_len(left_range))) {
    U64 index = it + left_range.min;
    V3F p = v3f(f32_from_radix_key(rcb_keys[index].center.x),
                f32_from_radix_key(rcb_keys[index].center.y),
                f32_from_radix_key(rcb_keys[index].center.z));

    left_bounds_local->min.x = f32_min(left_bounds_local->min.x, p.x);
    left_bounds_local->min.y = f32_min(left_bounds_local->min.y, p.y);
    left_bounds_local->min.z = f32_min(left_bounds_local->min.z, p.z);
    left_bounds_local->max.x = f32_max(left_bounds_local->max.x, p.x);
    left_bounds_local->max.y = f32_max(left_bounds_local->max.y, p.y);
    left_bounds_local->max.z = f32_max(left_bounds_local->max.z, p.z);
  }
 
  for Iter_Range(it, lane_range(range1_u64_len(right_range))) {
    U64 index = it + right_range.min;
    V3F p = v3f(f32_from_radix_key(rcb_keys[index].center.x),
                f32_from_radix_key(rcb_keys[index].center.y),
                f32_from_radix_key(rcb_keys[index].center.z));

    right_bounds_local->min.x = f32_min(right_bounds_local->min.x, p.x);
    right_bounds_local->min.y = f32_min(right_bounds_local->min.y, p.y);
    right_bounds_local->min.z = f32_min(right_bounds_local->min.z, p.z);
    right_bounds_local->max.x = f32_max(right_bounds_local->max.x, p.x);
    right_bounds_local->max.y = f32_max(right_bounds_local->max.y, p.y);
    right_bounds_local->max.z = f32_max(right_bounds_local->max.z, p.z);
  }

  // NOTE(cmat): Reduce bounds on lane 0.
  lane_barrier();

  if (lane_index() == 0) {
    left_bounds->min  = v3f(f32_limit_max, f32_limit_max, f32_limit_max);
    left_bounds->max  = v3f(f32_limit_min, f32_limit_min, f32_limit_min);
    right_bounds->min = v3f(f32_limit_max, f32_limit_max, f32_limit_max);
    right_bounds->max = v3f(f32_limit_min, f32_limit_min, f32_limit_min);

    for Iter_Index(it, lane_count()) {
      left_bounds->min.x  = f32_min(left_bounds->min.x, left_bounds_global[it].min.x);
      left_bounds->min.y  = f32_min(left_bounds->min.y, left_bounds_global[it].min.y);
      left_bounds->min.z  = f32_min(left_bounds->min.z, left_bounds_global[it].min.z);
      left_bounds->max.x  = f32_max(left_bounds->max.x, left_bounds_global[it].max.x);
      left_bounds->max.y  = f32_max(left_bounds->max.y, left_bounds_global[it].max.y);
      left_bounds->max.z  = f32_max(left_bounds->max.z, left_bounds_global[it].max.z);

      right_bounds->min.x = f32_min(right_bounds->min.x, right_bounds_global[it].min.x);
      right_bounds->min.y = f32_min(right_bounds->min.y, right_bounds_global[it].min.y);
      right_bounds->min.z = f32_min(right_bounds->min.z, right_bounds_global[it].min.z);
      right_bounds->max.x = f32_max(right_bounds->max.x, right_bounds_global[it].max.x);
      right_bounds->max.y = f32_max(right_bounds->max.y, right_bounds_global[it].max.y);
      right_bounds->max.z = f32_max(right_bounds->max.z, right_bounds_global[it].max.z);
    }
  }

  lane_broadcast_type(left_bounds, 0);
  lane_broadcast_type(right_bounds, 0);
  
  scratch_end(&scratch);
  lane_barrier();
  profiler_end_function();
}


function void ug_partition_rcb_split(UG_Partition *partition, Arena *arena, Range3_F32 bounds, U32 partition_begin, U32 partition_count, Range1_U64 range, UG_Partition_RCB_Key *rcb_keys, U32 depth) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);

  // NOTE(cmat): Leaf node.
  if (partition_count == 1) {
    log_info("RCB leaf #%u: %'llu elements", partition_begin, range_len);

    UG_Partition_Block *block = &partition->blocks_dat[partition_begin];

    block->cells_len = range_len;
    if (lane_index() == 0) {
      block->cells_dat = arena_push_count(arena, U32, range_len);
    }

    lane_broadcast_ptr(&block->cells_dat, 0);

    for Iter_Range(it, lane_range(range_len)) {
      U32 index = it + range.min;

      // NOTE(cmat): Assign cells to block.
      block->cells_dat[it]                                = rcb_keys[index].cell;
      partition->cells_block_index[rcb_keys[index].cell]  = partition_begin;
      partition->cells_local_index[rcb_keys[index].cell]  = it;
    }
    
    lane_barrier();

  } else {
    U32 split_axis = 0;
    range3_f32_largest_axis (bounds, &split_axis);
    array_sort_radix_u32    (range_len, sizeof(UG_Partition_RCB_Key) / sizeof(U32), split_axis, (U32 *)(rcb_keys + range.min));

    U32 left_partition_count  = partition_count / 2;
    U32 right_partition_count = partition_count - left_partition_count;

    // NOTE(cmat): Find weighted center.
    U64 center_index = range.min + (range_len * left_partition_count) / partition_count;
    // V3U center_key   = rcb_keys[center_index].center;
    // F32 split_coord  = f32_from_radix_key(center_key.dat[split_axis]);
    
    Range1_U64 left_range  = range1_u64(range.min, center_index);
    Range1_U64 right_range = range1_u64(center_index, range.max);

    Range3_F32 bounds_left    = { };
    Range3_F32 bounds_right   = { };

    ug_partition_rcb_child_bounds(left_range, right_range, rcb_keys, &bounds_left, &bounds_right);

    // TODO(cmat): We should recompute bounds.
    // bounds_left.max.dat   [split_axis] = split_coord;
    // bounds_right.min.dat  [split_axis] = split_coord;
    ug_partition_rcb_split(partition, arena, bounds_left,  partition_begin,                         left_partition_count,  left_range,  rcb_keys, depth + 1);
    ug_partition_rcb_split(partition, arena, bounds_right, partition_begin + left_partition_count,  right_partition_count, right_range, rcb_keys, depth + 1);
  }

  profiler_end_function();
}

function void ug_partition_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(arena);
  log_zone_start("Partitioning mesh: RCB");

  UG_Partition_RCB_Key *rcb_keys = 0;
  if (lane_index() == 0) {
    rcb_keys = arena_push_count(scratch.arena, UG_Partition_RCB_Key, mesh->cells.len);
  }

  lane_broadcast_ptr(&rcb_keys, 0);

  // NOTE(cmat): Assining RCB keys.
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    V3F center          = mesh->cells.center[it];
    rcb_keys[it].center = v3u(radix_key_from_f32(center.x), radix_key_from_f32(center.y), radix_key_from_f32(center.z));
    rcb_keys[it].cell   = it;
  }

  lane_barrier();

  log_info("Computing partitions for %u blocks", partition_count);
  Range3_F32 bounds = mesh->bounds;

  partition->blocks_len = partition_count;
  if (lane_index() == 0) {
    partition->blocks_dat         = arena_push_count(arena, UG_Partition_Block, partition_count);
    partition->cells_block_index  = arena_push_count(arena, U32,                mesh->cells.len);
    partition->cells_local_index  = arena_push_count(arena, U32,                mesh->cells.len);
  }

  lane_broadcast_ptr(&partition->blocks_dat,          0);
  lane_broadcast_ptr(&partition->cells_block_index,   0);
  lane_broadcast_ptr(&partition->cells_local_index,   0);
  ug_partition_rcb_split(partition, arena, bounds, 0, partition_count, range1_u64(0, mesh->cells.len), rcb_keys, 0);

  lane_barrier();

  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}
