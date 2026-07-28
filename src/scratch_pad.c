#pragma pack(push, 1)
typedef struct UG_Partition_RCB_Key {
  V3U center;
  U32 cell;
} UG_Partition_RCB_Key;
#pragma pack(pop)

Assert_Compiler(sizeof(UG_Partition_RCB_Key) == 4 * sizeof(U32));

// ============================================================
// CHANGE 1 & 2: bounds-update bug fix + restored adaptive axis
// ============================================================
// Added `root_bounds` param so we can classify each leaf's
// boundary-touching faces later (see CHANGE 4/5). Doesn't affect
// existing signature semantics otherwise.
function void ug_partition_rcb_split(UG_Partition *partition, Arena *arena, Range3_F32 root_bounds, Range3_F32 bounds, U32 partition_begin, U32 partition_count, Range1_U64 range, UG_Partition_RCB_Key *rcb_keys, U32 depth) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);

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
      block->cells_dat[it]                                = rcb_keys[index].cell;
      partition->cells_block_index[rcb_keys[index].cell]  = partition_begin;
      partition->cells_local_index[rcb_keys[index].cell]  = it;
    }

    lane_barrier();

    // CHANGE 4: record each leaf's final geometric bounds so we can
    // classify boundary-touching faces after the fact, without any
    // extra tree walk. Requires `bounds_dat` added to UG_Partition (see below).
    if (lane_index() == 0) {
      partition->bounds_dat[partition_begin] = bounds;
    }

  } else {
    // CHANGE 2: adaptive largest-axis split is now safe to re-enable,
    // because CHANGE 1 (below) makes `bounds` shrink exactly and
    // symmetrically regardless of cell density. Previously this branch
    // made things *worse* because bounds were density-warped, causing
    // sibling nodes to diverge onto inconsistent axis sequences.
    U32 split_axis = 0;
    range3_f32_largest_axis(bounds, &split_axis);

    // CHANGE 2b: stability guard. Near-cubic sub-boxes (common right after
    // a split) can flip which axis is "largest" from F32 rounding noise
    // alone, causing sibling branches to pick different axes for what
    // should be a symmetric cut. Fall back to depth-cycling only when the
    // two largest axes are within a tight tolerance of each other.
    {
      V3F extent = v3f_sub(bounds.max, bounds.min);
      F32 sorted[3] = { extent.x, extent.y, extent.z };
      // simple 3-element sort
      if (sorted[0] < sorted[1]) { F32 t = sorted[0]; sorted[0] = sorted[1]; sorted[1] = t; }
      if (sorted[1] < sorted[2]) { F32 t = sorted[1]; sorted[1] = sorted[2]; sorted[2] = t; }
      if (sorted[0] < sorted[1]) { F32 t = sorted[0]; sorted[0] = sorted[1]; sorted[1] = t; }
      F32 relative_gap = (sorted[0] - sorted[1]) / f32_max(sorted[0], 1e-6f);
      if (relative_gap < 0.01f) { // within 1% — treat as tied
        split_axis = depth % 3;
      }
    }

    array_sort_radix_u32(range_len, sizeof(UG_Partition_RCB_Key) / sizeof(U32), split_axis, (U32 *)(rcb_keys + range.min));

    U32 left_partition_count  = partition_count / 2;
    U32 right_partition_count = partition_count - left_partition_count;

    U64 center_index = range.min + (range_len * left_partition_count) / partition_count;

    // CHANGE 1: THE BUG. Previously we set the child bounds to the
    // *coordinate of the median cell* (center.dat[split_axis]), which
    // conflates "where the cell-count-balanced cut falls" with "what the
    // geometric bound of each child is." On non-uniform density meshes
    // (boundary-layer clustering, refinement near walls) these are NOT
    // the same point, so bounds_left/bounds_right became asymmetric in a
    // density-dependent way — corrupting every subsequent recursive split
    // and any downstream axis-selection logic.
    //
    // Fix: use the geometric midpoint of the CURRENT bounds for the split
    // plane. Cell-count balance is still achieved via center_index (used
    // below to cut the sorted array) — we've just decoupled it from the
    // geometric bound, which is what it should always have been.
    F32 split_coord = 0.5f * (bounds.min.dat[split_axis] + bounds.max.dat[split_axis]);

    Range3_F32 bounds_left  = bounds;
    Range3_F32 bounds_right = bounds;
    bounds_left.max.dat[split_axis]  = split_coord;
    bounds_right.min.dat[split_axis] = split_coord;

    ug_partition_rcb_split(partition, arena, root_bounds, bounds_left,  partition_begin,                        left_partition_count,  range1_u64(range.min,    center_index), rcb_keys, depth + 1);
    ug_partition_rcb_split(partition, arena, root_bounds, bounds_right, partition_begin + left_partition_count, right_partition_count, range1_u64(center_index, range.max),    rcb_keys, depth + 1);
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
    // CHANGE 4: storage for each leaf's final geometric bounds.
    partition->bounds_dat         = arena_push_count(arena, Range3_F32,         partition_count);
  }

  lane_broadcast_ptr(&partition->blocks_dat,          0);
  lane_broadcast_ptr(&partition->cells_block_index,   0);
  lane_broadcast_ptr(&partition->cells_local_index,   0);
  lane_broadcast_ptr(&partition->bounds_dat,          0);

  ug_partition_rcb_split(partition, arena, bounds, bounds, 0, partition_count, range1_u64(0, mesh->cells.len), rcb_keys, 0);

  lane_barrier();

  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

// ============================================================
// CHANGE 5: cost-aware rank assignment (new)
// ============================================================
// This is the part that actually targets your measured 2x per-block
// halo-count spread. Geometric RCB balances cell count, not comm cost —
// corner/edge/face/interior blocks are structurally unequal in halo size
// no matter how well the axis/bounds logic works (see prior message).
// Rather than trying to force equal-cost geometric partitions (hard to
// get right, easy to get wrong), we balance at RANK assignment time:
// greedily bin-pack the 64 already-computed blocks onto N ranks so each
// rank's TOTAL halo-exchange cost is balanced, instead of assigning
// blocks to ranks contiguously (block_id / blocks_per_rank) or by
// round-robin, either of which can accidentally concentrate all the
// expensive "interior" blocks onto a few ranks.
//
// Uses LPT (Longest Processing Time first): sort blocks by cost
// descending, repeatedly assign the next-heaviest block to the
// currently-lightest rank. Simple, well-understood, gives a bound of
// within ~4/3 of optimal for this kind of scheduling problem.
typedef struct UG_Partition_Rank_Assignment {
  U32  len;          // == partition_count
  U32 *block_rank;   // block_rank[block_index] = rank
} UG_Partition_Rank_Assignment;

function void ug_partition_assign_ranks_by_cost(
  UG_Partition_Rank_Assignment *out,
  Arena *arena,
  U32   *block_halo_count,   // halo count per block, e.g. from ug_mesh_array_from_partition
  U32   *block_ghost_count,  // ghost count per block
  U32    block_count,
  U32    rank_count,
  F32    halo_weight,        // tune empirically; start at 1.0
  F32    ghost_weight        // start lower than halo_weight — ghost cells
                              // are typically cheaper than a full halo
                              // exchange (no MPI round-trip), adjust to taste
) {
  out->len        = block_count;
  out->block_rank = arena_push_count(arena, U32, block_count);

  // Compute per-block cost.
  Arena_Temp scratch = scratch_start(arena);
  F32 *cost       = arena_push_count(scratch.arena, F32, block_count);
  U32 *sorted_idx = arena_push_count(scratch.arena, U32, block_count);
  for (U32 i = 0; i < block_count; i += 1) {
    cost[i]       = halo_weight * (F32)block_halo_count[i] + ghost_weight * (F32)block_ghost_count[i];
    sorted_idx[i] = i;
  }

  // Sort block indices by cost descending (simple insertion sort is fine —
  // block_count is small, e.g. 64; swap for array_sort_radix if this ever
  // needs to scale to thousands of blocks).
  for (U32 i = 1; i < block_count; i += 1) {
    U32 key_idx = sorted_idx[i];
    F32 key_cost = cost[key_idx];
    I32 j = (I32)i - 1;
    while (j >= 0 && cost[sorted_idx[j]] < key_cost) {
      sorted_idx[j + 1] = sorted_idx[j];
      j -= 1;
    }
    sorted_idx[j + 1] = key_idx;
  }

  // Greedy LPT bin-pack onto ranks.
  F32 *rank_load = arena_push_count(scratch.arena, F32, rank_count);
  for (U32 r = 0; r < rank_count; r += 1) rank_load[r] = 0.0f;

  for (U32 i = 0; i < block_count; i += 1) {
    U32 block = sorted_idx[i];

    U32 lightest_rank = 0;
    F32 lightest_load  = rank_load[0];
    for (U32 r = 1; r < rank_count; r += 1) {
      if (rank_load[r] < lightest_load) {
        lightest_load  = rank_load[r];
        lightest_rank  = r;
      }
    }

    out->block_rank[block]  = lightest_rank;
    rank_load[lightest_rank] += cost[block];
  }

  // Sanity log — check this after switching over. Max/min per-rank load
  // ratio should drop from ~2.0x (your current contiguous grouping,
  // assuming it happens to line up with octants) toward ~1.0-1.1x.
  {
    F32 min_load = rank_load[0], max_load = rank_load[0];
    for (U32 r = 1; r < rank_count; r += 1) {
      min_load = f32_min(min_load, rank_load[r]);
      max_load = f32_max(max_load, rank_load[r]);
    }
    log_info("Rank cost balance: min %.1f, max %.1f, ratio %.3f", min_load, max_load, max_load / f32_max(min_load, 1.0f));
  }

  scratch_end(&scratch);
}
