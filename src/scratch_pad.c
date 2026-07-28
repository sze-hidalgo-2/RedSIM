
function void array_reorder_key_u64_strided(U64 array_len, U64 elem_stride, U64 copy_size, U08 *array_dat, U64 key_size, U64 *key_dat, Array_Reorder_Mode mode) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  if (array_dat) {
    U08 *array_copy = 0;
    if (lane_index() == 0) {
      array_copy = arena_push_size(scratch.arena, copy_size * array_len);
    }
    lane_broadcast_ptr(&array_copy, 0);

    // NOTE(cmat): Copy array.
    for Iter_Range(it, lane_range(array_len)) {
      memory_copy(array_copy + copy_size * it, array_dat + elem_stride * it, copy_size);
    }

    lane_barrier();

    if (mode == Array_Reorder_Mode_New_To_Old) {
      for Iter_Range(it, lane_range(array_len)) {
        U64 key = *(U64 *)((U08 *)key_dat + it * key_size);
        memory_copy(array_dat + elem_stride * it, array_copy + copy_size * key, copy_size);
      }
    } else {
      for Iter_Range(it, lane_range(array_len)) {
        U64 key = *(U64 *)((U08 *)key_dat + it * key_size);
        memory_copy(array_dat + elem_stride * key, array_copy + copy_size * it, copy_size);
      }
    }
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}
