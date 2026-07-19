// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Reordering

function void array_reorder(U64 array_len, U64 type_size, U08 *array_dat, U64 key_size, U64 *key_dat) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  if (array_dat) {

    U64 array_bytes = type_size * array_len;
    U08 *array_copy = 0;
    if (lane_index() == 0) {
      array_copy = arena_push_size(scratch.arena, array_bytes);
    }

    lane_broadcast_ptr(&array_copy, 0);

    // NOTE(cmat): Copy array.
    Range1_U64 copy_range     = lane_range(array_len);
    U64        copy_bytes_off = type_size * copy_range.min;
    U64        copy_bytes_len = type_size * range1_u64_len(copy_range);
    memory_copy(array_copy + copy_bytes_off, array_dat + copy_bytes_off, copy_bytes_len);

    // NOTE(cmat): Fill original by permutating copy.
    lane_barrier();

    for Iter_Range(it, lane_range(array_len)) {
      U64 key = *(U64 *)((U08 *)key_dat + it * key_size);
      memory_copy(array_dat + type_size * it, array_copy + type_size * key, type_size);
    }
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}


// ------------------------------------------------------------
// #-- Radix Sort

function void array_sort_radix_u32(U64 array_len, U64 array_stride, U64 array_offset, U32 *array_dat) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  U64  stride_bytes  = sizeof(U32) * array_stride;
  U32 *temp_array    = 0;

  enum {
    // NOTE(cmat): 16 bits seem optimal, this leads to 2 passses and fits nicely into L1 cache.
    Radix_Bits  = 16,
    Bucket_Size = 1 << Radix_Bits,
    Radix_Mask  = Bucket_Size - 1,
  };

  U64  bucket_size        = Bucket_Size; // 256;
  I64  global_count_len   = bucket_size;
  I64  local_count_len    = bucket_size * lane_count();
  U64 *global_count_dat   = 0;
  U64 *local_count_dat    = 0;

  if (lane_index() == 0) {
    temp_array        = (U32 *)arena_push_size(scratch.arena, array_len         * stride_bytes);
    global_count_dat  = (U64 *)arena_push_size(scratch.arena, global_count_len  * sizeof(U64));
    local_count_dat   = (U64 *)arena_push_size(scratch.arena, local_count_len   * sizeof(U64));
  }

  lane_barrier();
  lane_broadcast_ptr(&temp_array,       0);
  lane_broadcast_ptr(&global_count_dat, 0);
  lane_broadcast_ptr(&local_count_dat,  0);

  U64 pass_count = (32 + Radix_Bits - 1) / Radix_Bits;
  // for Iter_Index(it_byte, 8) {
  for Iter_Index(pass, pass_count) {
    U32 shift = (U32)(pass * Radix_Bits);

    // NOTE(cmat): Clear histrogram for each lane.
    U64 *local_lane_count = local_count_dat + lane_index() * bucket_size;
    memory_fill(local_lane_count, 0, bucket_size * sizeof(U64));
    
    // NOTE(cmat): Build histogram for each lane.
    for Iter_Range(it, lane_range(array_len)) {
      U64 index                 = it * array_stride + array_offset;
      U32 bucket                = (array_dat[index] >> shift) & Radix_Mask;
      local_lane_count[bucket] += 1;
    }

    lane_barrier();

    if (lane_index() == 0) {
      // NOTE(cmat): Merge histograms.
      for Iter_Index(bucket, bucket_size) {
        U64 sum = 0;
        for Iter_Index(lane, lane_count()) {
          sum += local_count_dat[lane * bucket_size + bucket];
        }

        global_count_dat[bucket] = sum;
      }

      // NOTE(cmat): Convert to offsets (prefix sum)
      U64 prefix = 0;
      for Iter_Index(bucket, bucket_size) {
        U64 c                     = global_count_dat[bucket];
        global_count_dat[bucket]  = prefix;
        prefix                   += c;
      }

      // NOTE(cmat): Compute and store per-lane offsets in local_count_dat.
      for Iter_Index(bucket, bucket_size) {
        U64 running = global_count_dat[bucket];

        for Iter_Index(lane, lane_count()) {
          U64 *count    = local_count_dat + lane * bucket_size;
          U64  c        = count[bucket];
          count[bucket] = running;
          running      += c;
        }
      }
    }

    lane_barrier();

    // NOTE(cmat): Scatter
    U64 *lane_offsets = local_count_dat + lane_index() * bucket_size;
    for Iter_Range(it, lane_range(array_len)) {
      U64 src_index = it * array_stride;
      U32 value     = array_dat[src_index + array_offset];
      U32 bucket    = (value >> shift) & Radix_Mask;
      U64 dst_index = (lane_offsets[bucket]++) * array_stride;

      //!
      Assert(dst_index < array_len * array_stride, "error here");
      memory_copy(temp_array + dst_index, array_dat + src_index, stride_bytes);
    }

    lane_barrier();

    // NOTE(cmat): Instead of copying temp into the original array, we swap them instead.
    // Since the number of iterations is even (n=2), we actually end up with the original
    // array getting copied into during the final pass, so no final copy is needed.
    if (lane_index() == 0) {
      Macro_Swap(U32 *, array_dat, temp_array);
    }

    lane_barrier      ();
    lane_broadcast_ptr(&array_dat,   0);
    lane_broadcast_ptr(&temp_array,  0);
  }
  
  lane_barrier();

  scratch_end(&scratch);
  profiler_end_function();
}

function void array_sort_radix_u64(U64 array_len, U64 array_stride, U64 array_offset, U64 *array_dat) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  U64  stride_bytes       = sizeof(U64) * array_stride;
  U64 *temp_array         = 0;

  enum {
    // NOTE(cmat): 16 bits seem optimal, this leads to 4 passses and fits nicely into L1 cache.
    Radix_Bits  = 16,
    Bucket_Size = 1 << Radix_Bits,
    Radix_Mask  = Bucket_Size - 1,
  };

  U64  bucket_size        = Bucket_Size; // 256;
  I64  global_count_len   = bucket_size;
  I64  local_count_len    = bucket_size * lane_count();
  U64 *global_count_dat   = 0;
  U64 *local_count_dat    = 0;

  if (lane_index() == 0) {
    temp_array        = (U64 *)arena_push_size(scratch.arena, array_len         * stride_bytes);
    global_count_dat  = (U64 *)arena_push_size(scratch.arena, global_count_len  * sizeof(U64));
    local_count_dat   = (U64 *)arena_push_size(scratch.arena, local_count_len   * sizeof(U64));
  }

  lane_barrier();
  lane_broadcast_ptr(&temp_array,       0);
  lane_broadcast_ptr(&global_count_dat, 0);
  lane_broadcast_ptr(&local_count_dat,  0);

  U64 pass_count = (64 + Radix_Bits - 1) / Radix_Bits;
  for Iter_Index(pass, pass_count) {
    U64 shift = pass * Radix_Bits;

    // NOTE(cmat): Clear histrogram for each lane.
    U64 *local_lane_count = local_count_dat + lane_index() * bucket_size;
    memory_fill(local_lane_count, 0, bucket_size * sizeof(U64));
    
    // NOTE(cmat): Build histogram for each lane.
    for Iter_Range(it, lane_range(array_len)) {
      U64 index                 = it * array_stride + array_offset;
      U32 bucket                = (array_dat[index] >> shift) & Radix_Mask;
      local_lane_count[bucket] += 1;
    }

    lane_barrier();

    if (lane_index() == 0) {
      // NOTE(cmat): Merge histograms.
      for Iter_Index(bucket, bucket_size) {
        U64 sum = 0;
        for Iter_Index(lane, lane_count()) {
          sum += local_count_dat[lane * bucket_size + bucket];
        }

        global_count_dat[bucket] = sum;
      }

      // NOTE(cmat): Convert to offsets (prefix sum)
      U64 prefix = 0;
      for Iter_Index(bucket, bucket_size) {
        U64 c                     = global_count_dat[bucket];
        global_count_dat[bucket]  = prefix;
        prefix                   += c;
      }

      // NOTE(cmat): Compute and store per-lane offsets in local_count_dat.
      for Iter_Index(bucket, bucket_size) {
        U64 running = global_count_dat[bucket];

        for Iter_Index(lane, lane_count()) {
          U64 *count    = local_count_dat + lane * bucket_size;
          U64  c        = count[bucket];
          count[bucket] = running;
          running      += c;
        }
      }
    }

    lane_barrier();

    // NOTE(cmat): Scatter
    U64 *lane_offsets = local_count_dat + lane_index() * bucket_size;
    for Iter_Range(it, lane_range(array_len)) {
      U64 src_index = it * array_stride;
      U64 value     = array_dat[src_index + array_offset];
      // U08 byte      = (value >> (it_byte * 8)) & 0xFF;
      U32 bucket    = (value >> shift) & Radix_Mask;
      U64 dst_index = (lane_offsets[bucket]++) * array_stride;

      //!
      Assert(dst_index < array_len * array_stride, "error here");
      memory_copy(temp_array + dst_index, array_dat + src_index, stride_bytes);
    }

    lane_barrier();

    // NOTE(cmat): Instead of copying temp into the original array, we swap them instead.
    // Since the number of iterations is even (n=4), we actually end up with the original
    // array getting copied into during the final pass, so no final copy is needed.
    if (lane_index() == 0) {
      Macro_Swap(U64 *, array_dat, temp_array);
    }

    lane_barrier      ();
    lane_broadcast_ptr(&array_dat,   0);
    lane_broadcast_ptr(&temp_array,  0);
  }
  
  lane_barrier();

  scratch_end(&scratch);
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Space Filling Curves

function U64 morton64_encode_gap_3(U32 x) {
  // NOTE(cmat): Add gaps to a 21 bit integer so that there are two zero-s between every bit.
  U64 result = (U64)x & 0x1FFFFFULL;
  result     = (result | (result << 32)) & 0x1F00000000FFFFULL;
  result     = (result | (result << 16)) & 0x1F0000FF0000FFULL;
  result     = (result | (result <<  8)) & 0x100F00F00F00F00FULL;
  result     = (result | (result <<  4)) & 0x10C30C30C30C30C3ULL;
  result     = (result | (result <<  2)) & 0x1249249249249249ULL;

  return result;
}

function Morton64 morton64_encode_v3f(V3F v_unit) {
  // NOTE(cmat): Clamp input
  v_unit.x = f32_clamp(v_unit.x, 0, 1);
  v_unit.y = f32_clamp(v_unit.y, 0, 1);
  v_unit.z = f32_clamp(v_unit.z, 0, 1);

  // NOTE(cmat): First, we quantize each component to 21 bits.
  F32    q_21        = (F32)(0x1FFFFF);
  V3_F32 q_position  = v3f_mul  (q_21, v_unit);
  V3_U32 u_position  = V3_Cast  (q_position, U32);
  U64    morton      = 0;

  // NOTE(cmat): Now that we have three 21 bit integers, we interleave their bits.
  morton = morton | (morton64_encode_gap_3(u_position.x) << 0);
  morton = morton | (morton64_encode_gap_3(u_position.y) << 1);
  morton = morton | (morton64_encode_gap_3(u_position.z) << 2);

  return morton;
}

