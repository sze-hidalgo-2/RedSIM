// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// NOTE(cmat): For more about the concepts of lanes and how this
// multi-threading system is setup, see:
// - [ https://www.dgtlgrove.com/p/multi-core-by-default ]

// ------------------------------------------------------------
// #-- Thread Context

// NOTE(cmat): Idea of an implicit thread context adopted from Jonathan Blow's talks,
// Ryan Fleury's blogs, and Odin's builtin "context" mechanic.

// NOTE(cmat): Every thread has a context attached to it,
// containing scratch arenas, thread metadata and more.

typedef struct Thread_Context_Storage {
  Str08           group_name;
  U32             group_index;
  U32             group_count;
  SYS_Barrier     group_barrier;
  U64            *group_storage;

  U32             lane_index;
  U32             lane_count;

  Arena           scratch_1;
  Arena           scratch_2;
} Thread_Context_Storage;


// TODO(cmat): Don't love the idea of hiding thread_context initialization from the user,
// - Either the user should perhaps call init/destroy manually, or make this explicit in the wrappers somehow.
// - Same goes for profiling.
function void                     thread_context_init     (SYS_Barrier group_barrier, U64 *group_storage, Str08 group_name, U32 group_index, U32 group_count);
function void                     thread_context_destroy  (void);
function Thread_Context_Storage * thread_context          (void);

// ------------------------------------------------------------
// #-- Thread Groups

#define Thread_Group_Entry_Hook(name_) void name_(void *user_data)
typedef Thread_Group_Entry_Hook(Thread_Group_Entry);

typedef struct Thread_Group_Info {
  SYS_Barrier         group_barrier;
  Str08               group_name;
  U32                 group_index;
  U32                 group_count;
  U64                *group_storage;
  Thread_Group_Entry *user_entry;
  void               *user_data;
} Thread_Group_Info;

typedef struct Thread_Group {
  Arena               arena;
  Str08               group_name;
  U32                 group_count;
  SYS_Barrier         group_barrier;
  SYS_Thread         *group_threads;
  Thread_Group_Info  *group_infos;
  U64                 group_storage;
} Thread_Group;

function void thread_group_init       (Thread_Group *thread_group, Str08 group_name, U32 group_count);
function void thread_group_destroy    (Thread_Group *thread_group);
function void thread_group_launch     (Thread_Group *thread_group, Thread_Group_Entry *user_entry, void *user_data);
function void thread_group_wait_all   (Thread_Group *thread_group);

// ------------------------------------------------------------
// #-- Wavefronts, Lanes

function U32        lane_index         (void);
function U32        lane_count         (void);
function Range1_U64 lane_range         (U64 count);
function void       lane_barrier       (void);
function void       lane_broadcast_u64 (U64  *value,  U32 broadcast_lane);
function void       lane_broadcast_ptr (void *value,  U32 broadcast_lane);

// ------------------------------------------------------------
// #-- Scratch Storage

// NOTE(cmat): Idea from Ryan Fleury's memory managment talk:
// - [  https://www.youtube.com/watch?v=TZ5a3gCCZYo ]
// - The idea is whenever we pass a scratch arena to a function,
// - the function itself might also use the global scratch arena, causing aliasing.
// - This can be referred to as "arena aliasing", similar to pointer aliasing issues
// - well known in C. To fix this, we have two scratch arenas, and we ping pong between the two,
// - so when we ask for a new scrath arena, we're making sure it's not the arena that's been
// - passed to the function.

typedef Arena_Temp Scratch;
function Scratch scratch_start (Arena *conflict);
function void    scratch_end   (Scratch *scratch);

#define Scratch_Scope(scratch_, conflict_) \
  Defer_Scope(*(scratch_) = scratch_start(conflict_), scratch_end(scratch_))

// ------------------------------------------------------------
// #-- Mutex

typedef struct Mutex {
  volatile U32 next;
  volatile U32 serving;
} Mutex;

function void mutex_start (Mutex *mutex);
function void mutex_end   (Mutex *ticket);

#define Mutex_Scope(mutex) Defer_Scope(mutex_start(mutex), mutex_end(mutex))
