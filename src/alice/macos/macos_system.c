// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- macOS State

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <sys/param.h>

#include <pthread.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#endif

typedef struct MacOS_Barrier {
  pthread_mutex_t mutex;
  pthread_cond_t  cond;
  U32             count;
  U32             threshold;
  U32             generation;
} MacOS_Barrier;

// ------------------------------------------------------------
// #-- macOS Handle

typedef struct MacOS_Handle {
  SYS_Handle_Type type;
  B32             file_is_append;   // NEW: tracks O_APPEND for sys_file_write
  union {
    I32           file;
    pthread_t     thread;
    MacOS_Barrier barrier;   // was: pthread_barrier_t barrier;
  };
} MacOS_Handle;

typedef struct MacOS_Handle_Node {
  struct MacOS_Handle_Node *next;
  struct MacOS_Handle_Node *prev;
  MacOS_Handle              value;
} MacOS_Handle_Node;

typedef struct MacOS_Handle_List {
  MacOS_Handle_Node *first;
  MacOS_Handle_Node *last;
} MacOS_Handle_List;

global struct {
  SYS_Context        context;
  SYS_NUMA_Layout    numa_layout;
  Arena              arena;
  MacOS_Handle_List  handle_list;
  MacOS_Handle_List  handle_free_list;
} MacOS_State = { };

global F64 MacOS_Clock_To_Nanosec = 0;

// ------------------------------------------------------------
// #-- Handle Helpers

function SYS_Handle sys_handle_from_macos_handle(MacOS_Handle_Node *macos_handle) {
  SYS_Handle sys_handle = {
    .type  = macos_handle->value.type,
    .value = (U64)(void *)macos_handle,
  };
  return sys_handle;
}

function MacOS_Handle_Node *macos_handle_from_sys_handle(SYS_Handle *sys_handle,
                                                         SYS_Handle_Type type_check) {
  MacOS_Handle_Node *macos_handle = (MacOS_Handle_Node *)(void *)sys_handle->value;
  Assert(macos_handle->value.type == type_check, "invalid handle passed");
  return macos_handle;
}

function MacOS_Handle_Node *macos_handle_allocate(void) {
  MacOS_Handle_Node *node = 0;

  if (MacOS_State.handle_free_list.first) {
    node = MacOS_State.handle_free_list.first;
    DLL_Remove_First(MacOS_State.handle_free_list.first,
                     MacOS_State.handle_free_list.last);

    Zero_Fill(node);

    DLL_Push_Back(MacOS_State.handle_list.first,
                  MacOS_State.handle_list.last,
                  node);
  } else {
    node = arena_push_type(&MacOS_State.arena, MacOS_Handle_Node);

    DLL_Push_Back(MacOS_State.handle_list.first,
                  MacOS_State.handle_list.last,
                  node);
  }

  Zero_Fill(&node->value);
  return node;
}

function void macos_handle_free(MacOS_Handle_Node *node) {
  DLL_Remove(MacOS_State.handle_list.first,
             MacOS_State.handle_list.last,
             node);

  Zero_Fill(node);

  DLL_Push_Back(MacOS_State.handle_free_list.first,
                MacOS_State.handle_free_list.last,
                node);
}

// ------------------------------------------------------------
// #-- System API Implementation

link_function SYS_Context *sys_context(void) {
  return &MacOS_State.context;
}

link_function SYS_NUMA_Layout *sys_numa_layout(void) {
  return &MacOS_State.numa_layout;
}

// ------------------------------------------------------------
// #-- Output

link_function void sys_stream_write(Str08 buffer, SYS_Stream stream) {
  I32 unix_handle = 0;

  switch (stream) {
    case SYS_Stream_Standard_Output: {
      unix_handle = STDOUT_FILENO;
    } break;

    case SYS_Stream_Standard_Error: {
      unix_handle = STDERR_FILENO;
    } break;

    Invalid_Default;
  }

  write(unix_handle, buffer.txt, buffer.len);
}

link_function void sys_panic(Str08 reason) {
  sys_stream_write(
    str08_lit("## PANIC -- Aborting Execution ##\n"),
    SYS_Stream_Standard_Error
  );

  sys_stream_write(reason, SYS_Stream_Standard_Error);
  sys_stream_write(str08_lit("\n"), SYS_Stream_Standard_Error);

  abort();
}

// ------------------------------------------------------------
// #-- Time

link_function Local_Time sys_local_time_utc(void) {
  struct timeval tv;
  gettimeofday(&tv, 0);

  Local_Time result =
    local_time_from_unix_time((U64)tv.tv_sec, (U64)tv.tv_usec);

  return result;
}

// ------------------------------------------------------------
// #-- Virtual Memory

link_function U08 *sys_memory_reserve(U64 bytes) {
  void *address = mmap(
    0,
    bytes,
    PROT_NONE,
    MAP_PRIVATE | MAP_ANON,
    -1,
    0
  );

  if (address == MAP_FAILED) {
    sys_panic(str08_lit("virtual memory reserve failed"));
  }

  return (U08 *)address;
}

link_function void sys_memory_unreserve(void *virtual_base, U64 bytes) {
  if (munmap(virtual_base, bytes) != 0) {
    sys_panic(str08_lit("virtual memory unreserve failed"));
  }
}

link_function void sys_memory_commit(void *virtual_base,
                                     U64 bytes,
                                     SYS_Commit_Flag mode) {
  I32 prot = PROT_NONE;

  if (mode & SYS_Commit_Flag_Read) {
    prot |= PROT_READ;
  }

  if (mode & SYS_Commit_Flag_Write) {
    prot |= PROT_WRITE;
  }

  if (mode & SYS_Commit_Flag_Executable) {
    prot |= PROT_EXEC;
  }

  if (mprotect(virtual_base, bytes, prot) != 0) {
    sys_panic(str08_lit("virtual memory commit failed"));
  }
}

link_function void sys_memory_uncommit(void *virtual_base, U64 bytes) {
  if (mprotect(virtual_base, bytes, PROT_NONE) != 0) {
    sys_panic(str08_lit("virtual memory uncommit failed"));
  }
}

// ------------------------------------------------------------
// #-- Directory

link_function B32 sys_directory_create(Str08 folder_path) {
  I08 buffer[4096 + 1];

  U64 len = u64_min(folder_path.len, 4096);

  memory_copy(buffer, folder_path.txt, len);
  buffer[len] = 0;

  return mkdir((const char *)buffer, 0755) == 0;
}

link_function B32 sys_directory_delete(Str08 folder_path) {
  I08 buffer[4096 + 1];

  U64 len = u64_min(folder_path.len, 4096);

  memory_copy(buffer, folder_path.txt, len);
  buffer[len] = 0;

  return rmdir((const char *)buffer) == 0;
}

// ------------------------------------------------------------
// #-- Files

link_function SYS_File sys_file_open(Str08 file_path, SYS_File_Access_Flag flags) {
  I08 buffer[4096 + 1];

  U64 len = u64_min(file_path.len, 4096);

  memory_copy(buffer, file_path.txt, len);
  buffer[len] = 0;

  I32 mode = 0;

  if ((flags & SYS_File_Access_Flag_Read) &&
      (flags & SYS_File_Access_Flag_Write)) {
    mode |= O_RDWR;
  } else if (flags & SYS_File_Access_Flag_Read) {
    mode |= O_RDONLY;
  } else if (flags & SYS_File_Access_Flag_Write) {
    mode |= O_WRONLY;
  }

  if (flags & SYS_File_Access_Flag_Create) {
    mode |= O_CREAT;
  }

  if (flags & SYS_File_Access_Flag_Truncate) {
    mode |= O_TRUNC;
  }

  if (flags & SYS_File_Access_Flag_Append) {
    mode |= O_APPEND;
  }

  I32 fd = open((const char *)buffer, mode, 0644);

  if (fd < 0) {
    SYS_File result = {0};
    return result;
  }

  MacOS_Handle_Node *macos_handle = macos_handle_allocate();

  macos_handle->value.type           = SYS_Handle_Type_File;
  macos_handle->value.file           = fd;
  macos_handle->value.file_is_append = (flags & SYS_File_Access_Flag_Append) != 0; // NEW

  return sys_handle_from_macos_handle(macos_handle);
}

link_function U64 sys_file_size(SYS_File *file) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(file, SYS_Handle_Type_File);

  I32 file_handle = handle->value.file;

  struct stat st;
  if (fstat(file_handle, &st) == 0) {
    return (U64)st.st_size;
  }

  return 0;
}

link_function void sys_file_write(SYS_File *file,
                                  Range1_U64 range,
                                  void *data) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(file, SYS_Handle_Type_File);

  I32 file_handle = handle->value.file;
  U64 bytes       = range1_u64_len(range);

  if (handle->value.file_is_append) {
    // NOTE(cmat): Unlike Linux, macOS's pwrite() does NOT honor O_APPEND —
    // it always writes at the given offset. Use write() here instead so
    // append-opened files behave the same as they do on Linux (offset is
    // ignored, data is appended to the current end of file).
    write(file_handle, data, bytes);
  } else {
    U64 offset = range.min;
    pwrite(file_handle, data, bytes, (off_t)offset);
  }
}

link_function void sys_file_read(SYS_File *file,
                                 Range1_U64 range,
                                 void *data) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(file, SYS_Handle_Type_File);

  I32 file_handle = handle->value.file;

  U64 bytes  = range1_u64_len(range);
  U64 offset = range.min;

  pread(file_handle, data, bytes, (off_t)offset);
}

link_function void sys_file_close(SYS_File *file) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(file, SYS_Handle_Type_File);

  close(handle->value.file);
  macos_handle_free(handle);
}

// ------------------------------------------------------------
// #-- File Mapping

link_function SYS_File_Map sys_file_map(SYS_File *file,
                                        Range1_U64 range) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(file, SYS_Handle_Type_File);

  I32 file_handle = handle->value.file;

  U64 bytes  = range1_u64_len(range);
  U64 offset = range.min;

  U64 page_size = sys_context()->mmu_page_bytes;

  U64 offset_aligned =
    (offset / page_size) * page_size;

  U64 offset_remainder =
    offset - offset_aligned;

  U64 map_bytes =
    bytes + offset_remainder;

  U08 *address = mmap(
    0,
    map_bytes,
    PROT_READ,
    MAP_PRIVATE,
    file_handle,
    (off_t)offset_aligned
  );

  SYS_File_Map map = {
    .map_full  = str08(map_bytes, address),
    .map_range = str08(bytes, address + offset_remainder),
  };

  return map;
}

link_function void sys_file_unmap(SYS_File_Map *map) {
  if (map->map_full.txt) {
    munmap(map->map_full.txt, map->map_full.len);
  }
}

// ------------------------------------------------------------
// #-- Threads

link_function SYS_Thread sys_thread_launch(
  SYS_Thread_Entry_Point *entry_point,
  void *user_data) {

  pthread_t thread_handle = 0;

  pthread_create(
    &thread_handle,
    0,
    entry_point,
    user_data
  );

  MacOS_Handle_Node *macos_handle =
    macos_handle_allocate();

  macos_handle->value.type   = SYS_Handle_Type_Thread;
  macos_handle->value.thread = thread_handle;

  return sys_handle_from_macos_handle(macos_handle);
}

link_function void sys_thread_join(SYS_Thread *thread) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(thread, SYS_Handle_Type_Thread);

  pthread_join(handle->value.thread, 0);

  macos_handle_free(handle);
}

link_function U64 sys_thread_id(void) {
  uint64_t thread_id = 0;

  pthread_threadid_np(0, &thread_id);

  return (U64)thread_id;
}

link_function void sys_thread_set_name(Str08 name) {
  // pthread_setname_np() on macOS operates on the calling thread.
  char buffer[64];

  Zero_Fill(buffer);

  memory_copy(
    buffer,
    name.txt,
    u64_min(name.len, sizeof(buffer) - 1)
  );

  pthread_setname_np(buffer);
}

// ------------------------------------------------------------
// #-- CPU Binding
//
// macOS does not provide Linux sched_setaffinity-style CPU
// affinity. The Mach affinity-tag API is the closest native
// primitive, but it expresses affinity as a scheduling affinity
// group rather than "run only on CPU N".
//
// We therefore use the CPU id as the affinity tag.
//
// IMPORTANT:
// This is intentionally weaker than Linux CPU affinity.

link_function void sys_thread_bind_to_cpu(SYS_CPU cpu_id) {
  thread_affinity_policy_data_t policy = {
    .affinity_tag = (integer_t)(cpu_id + 1),
  };

  thread_policy_set(
    mach_thread_self(),
    THREAD_AFFINITY_POLICY,
    (thread_policy_t)&policy,
    THREAD_AFFINITY_POLICY_COUNT
  );
}

// ------------------------------------------------------------
// #-- Barriers

function void macos_barrier_init_(MacOS_Barrier *b, U32 count) {
  pthread_mutex_init(&b->mutex, 0);
  pthread_cond_init(&b->cond, 0);
  b->count      = 0;
  b->threshold  = count;
  b->generation = 0;
}

function void macos_barrier_wait_(MacOS_Barrier *b) {
  pthread_mutex_lock(&b->mutex);

  U32 gen = b->generation;
  b->count += 1;

  if (b->count == b->threshold) {
    b->generation += 1;
    b->count = 0;
    pthread_cond_broadcast(&b->cond);
  } else {
    while (gen == b->generation) {
      pthread_cond_wait(&b->cond, &b->mutex);
    }
  }

  pthread_mutex_unlock(&b->mutex);
}

function void macos_barrier_destroy_(MacOS_Barrier *b) {
  pthread_mutex_destroy(&b->mutex);
  pthread_cond_destroy(&b->cond);
}

link_function SYS_Barrier sys_barrier_init(U32 count) {
  MacOS_Handle_Node *macos_handle = macos_handle_allocate();

  macos_handle->value.type = SYS_Handle_Type_Barrier;
  macos_barrier_init_(&macos_handle->value.barrier, count);

  return sys_handle_from_macos_handle(macos_handle);
}

link_function void sys_barrier_wait(SYS_Barrier *barrier) {
  if (barrier->value != SYS_Barrier_None.value) {
    MacOS_Handle_Node *handle =
      macos_handle_from_sys_handle(barrier, SYS_Handle_Type_Barrier);

    macos_barrier_wait_(&handle->value.barrier);
  }
}

link_function void sys_barrier_destroy(SYS_Barrier *barrier) {
  MacOS_Handle_Node *handle =
    macos_handle_from_sys_handle(barrier, SYS_Handle_Type_Barrier);

  macos_barrier_destroy_(&handle->value.barrier);
  macos_handle_free(handle);
}

// ------------------------------------------------------------
// #-- Performance Clock

function void macos_performance_clock_init(void) {
  mach_timebase_info_data_t timebase = {};
  mach_timebase_info(&timebase);

  if (timebase.denom != 0) {
    MacOS_Clock_To_Nanosec =
      (F64)timebase.numer / (F64)timebase.denom;
  } else {
    sys_panic(str08_lit("mach timebase initialization failed"));
  }
}

link_function F64 sys_performance_clock_to_nanoseconds(void) {
  return MacOS_Clock_To_Nanosec;
}

link_function U64 sys_performance_clock_now(void) {
  return (U64)mach_absolute_time();
}

// ------------------------------------------------------------
// #-- CPU / System Information

function void macos_state_init_context(U32 argc, U08 **argv) {
  SYS_Context *context = &MacOS_State.context;

  context->command_line.argc = argc;
  context->command_line.argv = argv;

  // ----------------------------------------------------------
  // CPU name

  char cpu_name[256];
  size_t cpu_name_len = sizeof(cpu_name);

  Zero_Fill(cpu_name);

  if (sysctlbyname(
        "machdep.cpu.brand_string",
        cpu_name,
        &cpu_name_len,
        0,
        0) == 0) {

    context->cpu_name =
      str08((U64)strlen(cpu_name), (U08 *)cpu_name);

  } else {
    // Apple Silicon does not expose the Intel-style
    // machdep.cpu.brand_string.
    //
    // hw.model is still a useful fallback.
    char model[256];
    size_t model_len = sizeof(model);

    Zero_Fill(model);

    if (sysctlbyname(
          "hw.model",
          model,
          &model_len,
          0,
          0) == 0) {

      context->cpu_name =
        str08((U64)strlen(model), (U08 *)model);
    } else {
      context->cpu_name = str08_lit("");
    }
  }

  // ----------------------------------------------------------
  // Logical CPUs

  U32 logical_cores = 0;
  size_t logical_cores_len = sizeof(logical_cores);

  if (sysctlbyname(
        "hw.logicalcpu",
        &logical_cores,
        &logical_cores_len,
        0,
        0) == 0) {

    context->cpu_logical_cores = logical_cores;
  } else {
    context->cpu_logical_cores =
      (U64)sysconf(_SC_NPROCESSORS_ONLN);
  }

  // ----------------------------------------------------------
  // RAM

  U64 ram_capacity = 0;
  size_t ram_capacity_len = sizeof(ram_capacity);

  if (sysctlbyname(
        "hw.memsize",
        &ram_capacity,
        &ram_capacity_len,
        0,
        0) == 0) {

    context->ram_capacity_bytes = ram_capacity;
  }

  // ----------------------------------------------------------
  // VM page size

  context->mmu_page_bytes = (U64)vm_page_size;
}

// ------------------------------------------------------------
// #-- NUMA
//
// macOS does not expose a Linux/libnuma-equivalent NUMA API.
// Keep the abstraction valid by representing the entire machine
// as one logical node.
//
// This is particularly appropriate for the common macOS case,
// where the memory topology is unified.

function void macos_state_init_numa_layout(void) {
  SYS_NUMA_Layout *numa = &MacOS_State.numa_layout;

  numa->nodes_len = 1;
  numa->nodes_dat =
    arena_push_type(&MacOS_State.arena, SYS_NUMA_Node);

  SYS_NUMA_Node *node =
    &numa->nodes_dat[0];

  node->cpus_len =
    MacOS_State.context.cpu_logical_cores;

  node->cpus_dat =
    arena_push_count(
      &MacOS_State.arena,
      SYS_CPU,
      node->cpus_len
    );

  for Iter_Index(it_cpu, node->cpus_len) {
    node->cpus_dat[it_cpu] = it_cpu;
  }
}

// ------------------------------------------------------------
// #-- State Initialization

function void macos_state_init(U32 argc, U08 **argv) {
  Zero_Fill(&MacOS_State);

  // Context must be initialized before arena users.
  macos_state_init_context(argc, argv);

  arena_init(&MacOS_State.arena);

  macos_state_init_numa_layout();

  macos_performance_clock_init();
}

// ------------------------------------------------------------
// #-- macOS Entry Point

int main(int argc, char **argv) {
  macos_state_init(
    (U32)argc,
    (U08 **)argv
  );

  // NOTE(cmat): Call into user code.
  thread_context_init(
    SYS_Barrier_None,
    0,
    str08_lit("Main"),
    0,
    0
  );

  sys_entry_point();

  thread_context_destroy();

  return 0;
}

