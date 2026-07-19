// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

#if BUILD_PROFILE

  function              void profiler_startup             (void);
  function              void profiler_shutdown            (void);
  function              void profiler_init_for_thread     (void);
  function              void profiler_quit_for_thread     (void);
  function              void profiler_thread_flush        (void);
  function force_inline void profiler_begin_function_ext  (Str08 function_name, U64 clock);
  function force_inline void profiler_end_function_ext    (U64 clock);

# define profiler_begin_function() profiler_begin_function_ext(str08_lit(__FUNCTION__), sys_performance_clock_now())
# define profiler_end_function()   profiler_end_function_ext(sys_performance_clock_now())
# define Profiler_Scope            Defer_Scope(profiler_begin_function(), profiler_end_function())

#else

  force_inline function void profiler_startup             (void)                           { }
  force_inline function void profiler_shutdown            (void)                           { }
  force_inline function void profiler_init_for_thread     (void)                           { }
  force_inline function void profiler_quit_for_thread     (void)                           { }
  force_inline function void profiler_thread_flush        (void)                           { }
  force_inline function void profiler_begin_function_ext  (Str08 function_name, U64 clock) { }
  force_inline function void profiler_end_function_ext    (U64 clock)                      { }

# define profiler_begin_function()
# define profiler_end_function()
# define Profiler_Scope

#endif
