typedef struct UG_Partition {
} UG_Partition;

#pragma pack(push, 1)
typedef struct UG_Partition_RCB_Key {
  V3U center;
  U32 cell;
  U32 partition;
} UG_Partition_RCB_Key;

#pragma pack(pop)

Assert_Compiler(sizeof(UG_Partition_RCB_Key) == 5 * sizeof(U32));

function void ug_partition_rcb_split_recursive(UG_Grid *grid, Range3_F32 bounds, U32 partition_begin, U32 partition_count, Range1_U64 range, UG_Partition_RCB_Key *rcb_keys) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);

  // NOTE(cmat): Leaf node.
  if (partition_count == 1) {
    log_info("RCB leaf #%u: %'llu elements", partition_begin, range_len);
    for Iter_Range(it, lane_range(range_len)) {
      U32 index = it + range.min;

      // NOTE(cmat): Assign partition.
      rcb_keys[index].partition = partition_begin;
    }
  } else {
    U32 split_axis = 0;
    range3_f32_largest_axis (bounds, &split_axis);
    array_sort_radix_u32    (range_len, sizeof(UG_Partition_RCB_Key) / sizeof(U32), split_axis, (U32 *)(rcb_keys + range.min));

    U32 left_partition_count  = partition_count / 2;
    U32 right_partition_count = partition_count - left_partition_count;

    // NOTE(cmat): Find weighted center.
    U32 center_index  = range.min + (range_len * left_partition_count) / partition_count;
    V3U center_key    = rcb_keys[center_index].center;
    V3F center        = v3f(f32_from_radix_key(center_key.x), f32_from_radix_key(center_key.y), f32_from_radix_key(center_key.z));

    // log_info("Split at axis %u: %.2g", split_axis, center.dat[split_axis]);

    Range3_F32 bounds_left = bounds;
    Range3_F32 bounds_right = bounds;

    bounds_left.max.dat   [split_axis] = center.dat[split_axis];
    bounds_right.min.dat  [split_axis] = center.dat[split_axis];
    ug_partition_rcb_split_recursive(grid, bounds_left,  partition_begin,                         left_partition_count,  range1_u64(range.min,    center_index),  rcb_keys);
    ug_partition_rcb_split_recursive(grid, bounds_right, partition_begin + left_partition_count,  right_partition_count, range1_u64(center_index, range.max),     rcb_keys);
  }

  profiler_end_function();
}

function void ug_partition_rcb(UG_Grid *grid) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Partitioning grid: RCB");

  Assert(grid->elems.len != 0, "invalid grid");

  // NOTE(cmat): Compute centroids.
  UG_Partition_RCB_Key  *rcb_key       = 0;
  Range3_F32            *bounds_global = 0;
  if (lane_index() == 0) {
    rcb_key       = arena_push_count(scratch.arena, UG_Partition_RCB_Key, grid->elems.len);
    bounds_global = arena_push_count(scratch.arena, Range3_F32,           lane_count());
  }

  lane_broadcast_ptr(&rcb_key,        0);
  lane_broadcast_ptr(&bounds_global,  0);

  Range3_F32 *bounds_local  = bounds_global + lane_index();
  *bounds_local             = range3_f32(v3f_f32(f32_limit_max), v3f_f32(f32_limit_min));

  log_info("Computing centroids and bounds");
  for Iter_Range(it, lane_range(grid->elems.len)) {
    V4U verts          = grid->elems.verts[it];
    V3F x1             = v3f(grid->verts.x[verts.x], grid->verts.y[verts.x], grid->verts.z[verts.x]);
    V3F x2             = v3f(grid->verts.x[verts.y], grid->verts.y[verts.y], grid->verts.z[verts.y]);
    V3F x3             = v3f(grid->verts.x[verts.z], grid->verts.y[verts.z], grid->verts.z[verts.z]);
    V3F x4             = v3f(grid->verts.x[verts.w], grid->verts.y[verts.w], grid->verts.z[verts.w]);
    V3F center         = v3f_mul(.25f, v3f_add(x1, v3f_add(x2, v3f_add(x3, x4))));

    rcb_key[it].center    = v3u(radix_key_from_f32(center.x), radix_key_from_f32(center.y), radix_key_from_f32(center.z));
    rcb_key[it].cell      = it;
    rcb_key[it].partition = 0;

    for Iter_Index(elem, 3) {
      bounds_local->min.dat[elem] = f32_min(bounds_local->min.dat[elem], center.dat[elem]);
      bounds_local->max.dat[elem] = f32_max(bounds_local->max.dat[elem], center.dat[elem]);
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

  Range3_F32 bounds = bounds_global[0];
  ug_partition_rcb_split_recursive(grid, bounds, 0, 64, range1_u64(0, grid->elems.len), rcb_key);

  lane_barrier();

  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}
