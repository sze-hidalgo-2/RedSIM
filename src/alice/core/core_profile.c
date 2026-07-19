// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

global                SpallProfile  Spall_Context         = { };
global thread_storage SpallBuffer   Spall_Thread_Buffer   = { };
global thread_storage Arena         Spall_Thread_Arena    = { };

function void profiler_startup(void) {
  F64 clock_multiplier = sys_performance_clock_to_nanoseconds();
  spall_init_file("spall_trace.spall", clock_multiplier, &Spall_Context);
}

function void profiler_shutdown(void) {
  spall_quit(&Spall_Context);
}

function void profiler_init_for_thread(void) {
  Zero_Fill(&Spall_Thread_Buffer);
  Zero_Fill(&Spall_Thread_Arena);

  arena_init(&Spall_Thread_Arena);

  U64   backing_len   = u64_megabytes(128);
  U08  *backing_dat   = arena_push_size(&Spall_Thread_Arena, backing_len);

  U64 thread_id   = sys_thread_id();
  Spall_Thread_Buffer = (SpallBuffer) {
    .pid    = 0,
    .tid    = thread_id,
    .length = backing_len,
    .data   = backing_dat,
  };

  // NOTE(cmat): Touch pages to pre-fault all of them.
  memory_fill(Spall_Thread_Buffer.data, 0x00, Spall_Thread_Buffer.length);

  spall_buffer_init(&Spall_Context, &Spall_Thread_Buffer);

  // Str08 thread_name = str08_lit("Channel 0");
  Arena_Temp scratch = { };
  Scratch_Scope(&scratch, 0) {
    Str08 thread_name = str08_format(scratch.arena, "%S: %u", thread_context()->group_name, thread_context()->group_index);
    spall_buffer_name_thread(&Spall_Context, &Spall_Thread_Buffer, (char *)thread_name.txt, thread_name.len);

    Str08 process_name = str08_lit("Process: 0");
    spall_buffer_name_process(&Spall_Context, &Spall_Thread_Buffer, (char *)process_name.txt, process_name.len);
  }
}

function void profiler_quit_for_thread(void) {
  spall_buffer_quit(&Spall_Context, &Spall_Thread_Buffer);
}

function void profiler_thread_flush(void) {
  spall_buffer_flush(&Spall_Context, &Spall_Thread_Buffer);
}

function force_inline void profiler_begin_function_ext(Str08 function_name, U64 clock) {
  spall_buffer_begin(&Spall_Context, &Spall_Thread_Buffer, (char *)function_name.txt, function_name.len, clock);
}

function force_inline void profiler_end_function_ext(U64 clock) {
  spall_buffer_end(&Spall_Context, &Spall_Thread_Buffer, clock);
}
