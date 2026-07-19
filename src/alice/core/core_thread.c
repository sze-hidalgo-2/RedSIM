// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Thread Context

global thread_storage Thread_Context_Storage Thread_Context = { };

function void thread_context_init(SYS_Barrier group_barrier, U64 *group_storage, Str08 group_name, U32 group_index, U32 group_count) {
  Zero_Fill(&Thread_Context);

  // TODO(cmat): Should we push this onto an arena?
  Thread_Context.group_barrier = group_barrier;
  Thread_Context.group_name    = group_name;
  Thread_Context.group_storage = group_storage;

  Thread_Context.group_index   = group_index;
  Thread_Context.group_count   = group_count;

  Thread_Context.lane_index    = group_index;
  Thread_Context.lane_count    = group_count;

  // NOTE(cmat): Set thread-name.
  Scratch scratch = { };
  Scratch_Scope(&scratch, 0) {
    Str08 thread_name = str08_format(scratch.arena, "%S: %u", group_name, group_index);
    sys_thread_set_name(thread_name);
  }
}

function void thread_context_destroy(void) {
  // NOTE(cmat): Cleanup scratch memory.
  if (Thread_Context.scratch_1.initialized) { arena_destroy(&Thread_Context.scratch_1); }
  if (Thread_Context.scratch_2.initialized) { arena_destroy(&Thread_Context.scratch_2); }
}

function Thread_Context_Storage *thread_context(void) {
  return &Thread_Context;
}

// ------------------------------------------------------------
// #-- Thread Groups

function void thread_group_init(Thread_Group *thread_group, Str08 group_name, U32 group_count) {
  arena_init(&thread_group->arena);
  thread_group->group_barrier  = (group_count > 1) ? sys_barrier_init(group_count) : SYS_Barrier_None;
  thread_group->group_name     = group_name;
  thread_group->group_count    = group_count;
  thread_group->group_threads  = arena_push_count(&thread_group->arena, SYS_Thread, group_count);
  thread_group->group_storage  = 0;
  thread_group->group_infos    = arena_push_count(&thread_group->arena, Thread_Group_Info, group_count);
}

function void thread_group_destroy(Thread_Group *thread_group) {
  arena_destroy(&thread_group->arena);
  Zero_Fill(thread_group);

  if (thread_group->group_barrier.value != SYS_Barrier_None.value) {
    sys_barrier_destroy(&thread_group->group_barrier);
  }
}

function void *thread_group_entry_setup(void *user_data) {
  Thread_Group_Info *info = (Thread_Group_Info *)user_data;

  thread_context_init(info->group_barrier, info->group_storage, info->group_name, info->group_index, info->group_count);
  profiler_init_for_thread();

  lane_barrier();
  info->user_entry(info->user_data);
  
  profiler_quit_for_thread();
  thread_context_destroy();

  return 0;
}

function void thread_group_launch(Thread_Group *thread_group, Thread_Group_Entry *user_entry, void *user_data) {
  for Iter_Index(it, thread_group->group_count) {
    thread_group->group_infos[it] = (Thread_Group_Info) {
      .group_barrier = thread_group->group_barrier,
      .group_name    = thread_group->group_name,
      .group_index   = it,
      .group_count   = thread_group->group_count,
      .group_storage = &thread_group->group_storage,
      .user_entry    = user_entry,
      .user_data     = user_data,
    };

    thread_group->group_threads[it] = sys_thread_launch(thread_group_entry_setup, &thread_group->group_infos[it]);
  }
}

function void thread_group_wait_all(Thread_Group *thread_group) {
  for Iter_Index(it, thread_group->group_count) {
    sys_thread_join(&thread_group->group_threads[it]);
  }
}

// ------------------------------------------------------------
// #-- Wavefronts, Lanes

function U32 lane_index(void) {
  return Thread_Context.lane_index;
}

function U32 lane_count(void) {
  return Thread_Context.lane_count;
}

function Range1_U64 lane_range(U64 count) {
  U64         work_quotient    = count / lane_count(); 
  U64         work_remainder   = count % lane_count();
  U64         remainder_before = u64_min(lane_index(), work_remainder);
  U64         range_start      = (work_quotient * lane_index() + remainder_before);
  U64         range_end        = (range_start + work_quotient + (lane_index() < work_remainder));
  Range1_U64  range            = range1_u64(range_start, range_end);

  return range;
}

function void lane_barrier(void) {
  profiler_begin_function();
  sys_barrier_wait(&Thread_Context.group_barrier);
  profiler_end_function();
}

function void lane_broadcast_u64(U64 *value, U32 broadcast_lane) {
  if (lane_index() == broadcast_lane) { memory_copy(Thread_Context.group_storage, value, sizeof(U64)); }
  lane_barrier();
  if (lane_index() != broadcast_lane) { memory_copy(value, Thread_Context.group_storage, sizeof(U64)); }
  lane_barrier();
}

function void lane_broadcast_ptr(void *value, U32 broadcast_lane) {
  lane_broadcast_u64((U64 *)value, broadcast_lane);
}

// ------------------------------------------------------------
// #-- Scratch Storage

function Arena *scratch_get_for_thread(Arena *conflict) {
  Arena *scratch = &Thread_Context.scratch_1;
  if(conflict == &Thread_Context.scratch_1) {
    scratch = &Thread_Context.scratch_2;
  }

  // NOTE(cmat): If not initialized, initialize given scratch storage.
  If_Unlikely (!scratch->initialized) {
    arena_init(scratch);
  }
  
  return scratch;
}

function Scratch scratch_start(Arena *conflict) { 
  Arena       *arena  = scratch_get_for_thread(conflict);
  Arena_Temp   temp   = arena_temp_start(arena);
  return temp;
}

function void scratch_end(Scratch *scratch) {
  arena_temp_end(scratch);
}

// ------------------------------------------------------------
// #-- Mutex

function void mutex_start(Mutex *mutex) {
  U32 ticket = atomic_increment_u32(&mutex->next) - 1;
  while (atomic_read_u32(&mutex->serving) != ticket) {
    spin_lock();
  }
}

function void mutex_end(Mutex *ticket) {
  atomic_increment_u32(&ticket->serving);
}

