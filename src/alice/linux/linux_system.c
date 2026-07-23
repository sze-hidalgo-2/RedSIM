// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <pthread.h>
#include <numa.h>

#include <x86intrin.h>
#include <immintrin.h>

#include <linux/perf_event.h>
#include <linux/io_uring.h>

// ------------------------------------------------------------
// #-- Linux State

typedef struct Linux_Handle {
  SYS_Handle_Type type;
  union {
    I32               file;
    pthread_t         thread;
    pthread_barrier_t barrier;
  };
} Linux_Handle;

typedef struct Linux_Handle_Node {
  struct Linux_Handle_Node  *next;
  struct Linux_Handle_Node  *prev;
  Linux_Handle               value;
} Linux_Handle_Node;

typedef struct Linux_Handle_List {
  Linux_Handle_Node *first;
  Linux_Handle_Node *last;
} Linux_Handle_List;

global struct {
  SYS_Context       context;
  SYS_NUMA_Layout   numa_layout;
  Arena             arena;
  Linux_Handle_List handle_list;
  Linux_Handle_List handle_free_list;
  U08               cpu_id_buffer[48];
} Linux_State = { };

function SYS_Handle sys_handle_from_linux_handle(Linux_Handle_Node *linux_handle) {
  SYS_Handle sys_handle = { .type = linux_handle->value.type, .value = (U64)(void *)linux_handle };
  return sys_handle;
}

function Linux_Handle_Node *linux_handle_from_sys_handle(SYS_Handle *sys_handle, SYS_Handle_Type type_check) {
  Linux_Handle_Node *linux_handle = (Linux_Handle_Node *)(void *)sys_handle->value;
  Assert(linux_handle->value.type == type_check, "invalid handle passed");
  return linux_handle;
}

function Linux_Handle_Node *linux_handle_allocate(void) {
  Linux_Handle_Node *node = 0;
  if (Linux_State.handle_free_list.first) {
    node = Linux_State.handle_free_list.first;
    DLL_Remove_First(Linux_State.handle_free_list.first, Linux_State.handle_free_list.last);
    Zero_Fill(node);
    DLL_Push_Back(Linux_State.handle_list.first, Linux_State.handle_list.last, node);
  } else {
    node = arena_push_type(&Linux_State.arena, Linux_Handle_Node);
    DLL_Push_Back(Linux_State.handle_list.first, Linux_State.handle_list.last, node);
  }

  Zero_Fill(&node->value);
  return node;
}

function void linux_handle_free(Linux_Handle_Node *node) {
  DLL_Remove(Linux_State.handle_list.first, Linux_State.handle_list.last, node);
  Zero_Fill(node);
  DLL_Push_Back(Linux_State.handle_free_list.first,  Linux_State.handle_free_list.last,  node);
}

// ------------------------------------------------------------
// #-- System API Implementation

link_function SYS_Context *sys_context(void) {
  return &Linux_State.context;
}

link_function SYS_NUMA_Layout *sys_numa_layout(void) {
  return &Linux_State.numa_layout;
}

link_function void sys_stream_write(Str08 buffer, SYS_Stream stream) {
  I32 unix_handle = 0;
    
  switch(stream) {
    case SYS_Stream_Standard_Output:  { unix_handle = 1; } break;
    case SYS_Stream_Standard_Error:   { unix_handle = 2; } break;
    Invalid_Default;
  }
    
  write(unix_handle, buffer.txt, buffer.len);
}

link_function void sys_panic(Str08 reason) {
  sys_stream_write(str08_lit("## PANIC -- Aborting Execution ##\n"), SYS_Stream_Standard_Error);
  sys_stream_write(reason, SYS_Stream_Standard_Error);
  sys_stream_write(str08_lit("\n"), SYS_Stream_Standard_Error);
  exit(1);
}

link_function Local_Time sys_local_time_utc(void) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  Local_Time result = local_time_from_unix_time((U64)tv.tv_sec, (U64)tv.tv_usec);
  return result;
}

link_function U08 *sys_memory_reserve(U64 bytes) {
  void *address = mmap(0, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (address == (void*)-1) {
    sys_panic(str08_lit("virtual memory reserve failed"));
  }
    
  return (U08 *)address;
}

link_function void sys_memory_unreserve(void *virtual_base, U64 bytes) {
  if (munmap(virtual_base, bytes)) {
    sys_panic(str08_lit("virtual memory unreserve failed"));
  }
}

link_function void sys_memory_commit(void *virtual_base, U64 bytes, SYS_Commit_Flag mode) {
  unsigned long prot = 0;
  if (mode & SYS_Commit_Flag_Read)       prot |= PROT_READ;
  if (mode & SYS_Commit_Flag_Write)      prot |= PROT_WRITE;
  if (mode & SYS_Commit_Flag_Executable) prot |= PROT_EXEC;
  
  if (mprotect(virtual_base, bytes, prot)) {
    sys_panic(str08_lit("virtual memory commit failed"));
  }
}

link_function void sys_memory_uncommit(void *virtual_base, U64 bytes) {
  if (mprotect(virtual_base, bytes, PROT_NONE)) {
    sys_panic(str08_lit("virtual memory uncommit failed"));
  }
}

link_function B32 sys_directory_create(Str08 folder_path) {
  // TODO(cmat): Handle this better.
  I08 buffer[4096 + 1];
  memory_copy(buffer, folder_path.txt, u64_min(folder_path.len, 4096));
  buffer[u64_min(folder_path.len, 4096)] = 0;
    
  B32 result = mkdir((const char *)buffer, 0755) >= 0;
  return result;
}

link_function B32 sys_directory_delete(Str08 folder_path) {
  // TODO(cmat): Handle this better.
  I08 buffer[4096 + 1];
  memory_copy(buffer, folder_path.txt, u64_min(folder_path.len, 4096));
  buffer[u64_min(folder_path.len, 4096)] = 0;
    
  B32 result = rmdir((const char *)buffer) >= 0;
  return result;
}

link_function SYS_File sys_file_open(Str08 file_path, SYS_File_Access_Flag flags) {
  // TODO(cmat): Handle this better.
  I08 buffer[4096 + 1];
  memory_copy(buffer, file_path.txt, u64_min(file_path.len, 4096));
  buffer[u64_min(file_path.len, 4096)] = 0;
    
  I32 mode = 0;
  if ((flags & SYS_File_Access_Flag_Read) && (flags & SYS_File_Access_Flag_Write)) {
    mode |= O_RDWR;
  } else {
    if (flags & SYS_File_Access_Flag_Read)       mode |= O_RDONLY;
    else if (flags & SYS_File_Access_Flag_Write) mode |= O_WRONLY;
  }
    
  if (flags & SYS_File_Access_Flag_Create)    mode |= O_CREAT;
  if (flags & SYS_File_Access_Flag_Truncate)  mode |= O_TRUNC;
  if (flags & SYS_File_Access_Flag_Append)    mode |= O_APPEND;
    
  I32 fd = open((const char *)buffer, mode, 0644);

  Linux_Handle_Node *linux_handle = linux_handle_allocate();
  linux_handle->value.type        = SYS_Handle_Type_File;
  linux_handle->value.file        = fd;

  SYS_Handle sys_handle = sys_handle_from_linux_handle(linux_handle);
  return sys_handle;
}

link_function U64 sys_file_size(SYS_File *file) {
  U64 result                = 0;
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(file, SYS_Handle_Type_File);
  I32 file_handle           = handle->value.file;
  
  struct stat st;
  if (fstat(file_handle, &st) == 0) {
    result = (U64)st.st_size;
  }
    
  return result;
}

link_function void sys_file_write(SYS_File *file, Range1_U64 range, void *data) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(file, SYS_Handle_Type_File);
  I32 file_handle           = handle->value.file;
  U64 bytes                 = range1_u64_len(range);
  U64 offset                = range.min;
  pwrite(file_handle, data, bytes, offset);
}

link_function void sys_file_read(SYS_File *file, Range1_U64 range, void *data) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(file, SYS_Handle_Type_File);
  I32 file_handle           = handle->value.file;
  U64 bytes                 = range1_u64_len(range);
  U64 offset                = range.min;

  pread(file_handle, data, bytes, offset);
}

link_function void sys_file_close(SYS_File *file) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(file, SYS_Handle_Type_File);
  I32 file_handle           = handle->value.file;
  close(file_handle);
  linux_handle_free(handle);
}

// TODO(cmat): Page alignment fix.
link_function SYS_File_Map sys_file_map(SYS_File *file, Range1_U64 range) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(file, SYS_Handle_Type_File);
  I32 file_handle           = handle->value.file;
  U64 bytes                 = range1_u64_len(range);
  U64 offset                = range.min;
  U64 page_size             = sys_context()->mmu_page_bytes;
  U64 offset_aligned        = (offset / page_size) * page_size;
  U64 offset_remainder      = offset - offset_aligned;
  U64 map_bytes             = bytes + offset_remainder;
  U08 *address              = mmap(0, map_bytes, PROT_READ, MAP_PRIVATE, file_handle, offset_aligned);

  SYS_File_Map map = {
    .map_full   = str08(map_bytes,  address),
    .map_range  = str08(bytes,      address + offset),
  };

  return map;
}

link_function void sys_file_unmap(SYS_File_Map *map) {
  munmap(map->map_full.txt, map->map_full.len);
}

link_function SYS_Thread sys_thread_launch(SYS_Thread_Entry_Point *entry_point, void *user_data) {
  pthread_t thread_handle = 0;
  pthread_create(&thread_handle, 0, entry_point, user_data);

  Linux_Handle_Node *linux_handle = linux_handle_allocate();
  linux_handle->value.type        = SYS_Handle_Type_Thread;
  linux_handle->value.thread      = thread_handle;

  SYS_Handle sys_handle = sys_handle_from_linux_handle(linux_handle);
  return sys_handle;
}

link_function void sys_thread_join(SYS_Thread *thread) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(thread, SYS_Handle_Type_Thread);
  pthread_t thread_handle   = handle->value.thread;

  pthread_join(thread_handle, 0);
  linux_handle_free(handle);
}

link_function U64 sys_thread_id(void) {
  U64 id = (U64)pthread_self();
  return id;
}

link_function void sys_thread_set_name(Str08 name) {
  // NOTE(cmat): From the man pages,
  //  "The thread name [...] is restricted to 16 characters, including the terminating null byte ('\0')."
  char buffer[16];
  Zero_Fill(buffer);
  memory_copy(buffer, name.txt, u64_min(name.len, 15));

  pthread_t thread_handle = pthread_self();
  pthread_setname_np(thread_handle, buffer);
}

link_function void sys_thread_bind_to_cpu(SYS_CPU cpu_id) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET((I32)cpu_id, &cpu_set);

  pthread_t thread_handle = pthread_self();
  pthread_setaffinity_np(thread_handle, sizeof(cpu_set_t), &cpu_set);
}

link_function SYS_Barrier sys_barrier_init(U32 count) {
  Linux_Handle_Node *linux_handle = linux_handle_allocate();
  linux_handle->value.type        = SYS_Handle_Type_Barrier;
  pthread_barrier_init(&linux_handle->value.barrier, 0, count);

  SYS_Handle sys_handle = sys_handle_from_linux_handle(linux_handle);
  return sys_handle;
}

link_function void sys_barrier_wait(SYS_Barrier *barrier) {
  if (barrier->value != SYS_Barrier_None.value) {
    Linux_Handle_Node *handle = linux_handle_from_sys_handle(barrier, SYS_Handle_Type_Barrier);
    pthread_barrier_wait(&handle->value.barrier);
  }
}

link_function void sys_barrier_destroy(SYS_Barrier *barrier) {
  Linux_Handle_Node *handle = linux_handle_from_sys_handle(barrier, SYS_Handle_Type_Barrier);
  pthread_barrier_destroy(&handle->value.barrier);
  linux_handle_free(handle);
}

global F64 Linux_Clock_To_Nanosec = 0;
function long linux_perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

link_function void linux_performance_clock_init(void) {
  if (Linux_Clock_To_Nanosec == 0) {
    struct perf_event_attr perf_event = {
      .type           = PERF_TYPE_HARDWARE,
      .size           = sizeof(struct perf_event_attr),
      .config         = PERF_COUNT_HW_INSTRUCTIONS,
      .disabled       = 1,
      .exclude_kernel = 1,
      .exclude_hv     = 1,
    };

    I32 fd = linux_perf_event_open(&perf_event, 0, -1, -1, 0);

    if (fd != -1) {
      U64 page_size = sys_context()->mmu_page_bytes;
      void *address = mmap(0, page_size, PROT_READ, MAP_SHARED, fd, 0);
      if (address) {
        struct perf_event_mmap_page *page = (struct perf_event_mmap_page *)address;
        if (page->cap_user_time == 1) {
          __uint128_t nano_seconds = 1000000000000000ull;
          nano_seconds            *= page->time_mult;
          nano_seconds           >>= page->time_shift;
          F64 convert_to_seconds   = nano_seconds / 1000000000000000.0;
          Linux_Clock_To_Nanosec   = convert_to_seconds;
        } else {
          sys_panic(str08_lit("perf system doesn't support user time"));
        }
      } else {
        sys_panic(str08_lit("mmap failed for perf_event"));
      }
    } else {
      sys_panic(str08_lit("perf_event_open failed"));
    }
  }
}

link_function F64 sys_performance_clock_to_nanoseconds(void) {
  return Linux_Clock_To_Nanosec;
}

link_function U64 sys_performance_clock_now(void) {
  U64 clock = __rdtsc();
  return clock;
}

// ------------------------------------------------------------
// #-- Linux State Initialization

function void linux_state_init_context(U32 argc, U08 **argv) {
  SYS_Context *context = &Linux_State.context;
  
  context->command_line.argc = argc;
  context->command_line.argv = argv;

#if ARCH_X86  
  // NOTE(cmat): Get CPU name.
  U32 eax, ebx, ecx, edx;
  U08 *cpu_id_at = Linux_State.cpu_id_buffer;
  
  for Iter_Index(it, 3) {
      eax = 0x80000002 + it;
      __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
      
      memory_copy(cpu_id_at + 0,  &eax, 4);
      memory_copy(cpu_id_at + 4,  &ebx, 4);
      memory_copy(cpu_id_at + 8,  &ecx, 4);
      memory_copy(cpu_id_at + 12, &edx, 4);
      cpu_id_at += 16;
  }

  context->cpu_name = str08(Stack_Array_Len(Linux_State.cpu_id_buffer), Linux_State.cpu_id_buffer);
  context->cpu_name = str08_trim(context->cpu_name);

#else
  context->cpu_name = str08_lit("");
  Not_Implemented;
#endif

  context->cpu_logical_cores = (U64)sysconf(_SC_NPROCESSORS_ONLN);
 
  struct sysinfo info = {};
  if (sysinfo(&info) == 0) {
    context->ram_capacity_bytes = (U64)info.totalram * (U64)info.mem_unit;
  }
    
  context->mmu_page_bytes = (U64)sysconf(_SC_PAGESIZE); 
}

function void linux_state_init_numa_layout(void) {
  SYS_NUMA_Layout *numa = &Linux_State.numa_layout;
  if (numa_available() >= 0) {
    numa->nodes_len = numa_max_node() + 1;
    numa->nodes_dat = arena_push_count(&Linux_State.arena, SYS_NUMA_Node, numa->nodes_len);

    struct bitmask *cpu_mask = numa_allocate_cpumask();
    for Iter_Index(it_node, numa->nodes_len) {

      // NOTE(cmat): Get cpu-s corresponding to current node.
      if (numa_node_to_cpus(it_node, cpu_mask) == 0) {
        // NOTE(cmat): Count CPU-s
        U64 cpu_count = 0;
        for Iter_Index(it_cpu, cpu_mask->size) {
          if (numa_bitmask_isbitset(cpu_mask, it_cpu)) {
            cpu_count += 1;
          }
        }

        // NOTE(cmat): Allocate and fill-out.
        numa->nodes_dat[it_node].cpus_len = cpu_count;
        numa->nodes_dat[it_node].cpus_dat = arena_push_count(&Linux_State.arena, SYS_CPU, cpu_count);

        U64 cpu_at = 0;
        for Iter_Index(it_cpu, cpu_mask->size) {
          if (numa_bitmask_isbitset(cpu_mask, it_cpu)) {
            numa->nodes_dat[it_node].cpus_dat[cpu_at++] = it_cpu;
          }
        }
      }
    }

    numa_free_cpumask(cpu_mask);
  }
}

function void linux_state_init(U32 argc, U08 **argv) {
  Zero_Fill(&Linux_State);

  // NOTE(cmat): Necessary to intialize first for arena-use.
  // TODO(cmat): Is there a way to by-pass this? A more elegant solution?
  linux_state_init_context(argc, argv);

  // NOTE(cmat): Now we can safely initialize the arena for the rest of the initializations.
  arena_init(&Linux_State.arena);

  // NOTE(cmat): Get NUMA layout (if available).
  linux_state_init_numa_layout();

  // NOTE(cmat): Setup high-performance clock.
  linux_performance_clock_init();
}

// ------------------------------------------------------------
// #-- Linux Entry Point

int main(int argc, char **argv) {
  linux_state_init((U32)argc, (U08 **)argv);

  // NOTE(cmat): Call into user code.
  // TODO(cmat): This code should not be written for all platforms, should be centralized.
  // TODO(cmat): This is also a mess, profiler_init_for_thread is also called in thread.c, 
  // - should find a way to simplify / make this whole thing more uniform.
  thread_context_init(SYS_Barrier_None, 0, str08_lit("Main"), 0, 0);
  // profiler_startup("spall_trace.spall");
  // profiler_init_for_thread();

  sys_entry_point();

  // profiler_quit_for_thread();
  // profiler_shutdown();
  thread_context_destroy();
}
