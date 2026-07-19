// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- System Data Structures

typedef struct SYS_Context {
  Str08 cpu_name;
  U64   cpu_logical_cores;
  U64   mmu_page_bytes;
  U64   ram_capacity_bytes;
} SYS_Context;

typedef U64 SYS_CPU;

typedef struct SYS_NUMA_Node {
  U64      cpus_len;
  SYS_CPU *cpus_dat;
} SYS_NUMA_Node;

typedef struct SYS_NUMA_Layout {
  U64             nodes_len;
  SYS_NUMA_Node  *nodes_dat;
} SYS_NUMA_Layout;

typedef U32 SYS_Stream;
enum {
  SYS_Stream_Standard_Output,
  SYS_Stream_Standard_Error,
};

typedef U32 SYS_Commit_Flag;
enum {
  SYS_Commit_Flag_Read         = 1 << 0,
  SYS_Commit_Flag_Write        = 1 << 1,
  SYS_Commit_Flag_Executable   = 1 << 2,
};

typedef U32 SYS_File_Access_Flag;
enum {
  SYS_File_Access_Flag_Read        = 1 << 0,
  SYS_File_Access_Flag_Write       = 1 << 1,

  SYS_File_Access_Flag_Create      = 1 << 2,
  SYS_File_Access_Flag_Truncate    = 1 << 3,
  SYS_File_Access_Flag_Append      = 1 << 4,
};

typedef struct SYS_File_Map {
  Str08 map_full;
  Str08 map_range;
} SYS_File_Map;

#define SYS_Thread_Entry_Point_Callback(name_) void *name_(void *user_data)
typedef SYS_Thread_Entry_Point_Callback(SYS_Thread_Entry_Point);

typedef U64 SYS_Handle_Type;
enum {
  SYS_Handle_Type_Invalid = 0,

  SYS_Handle_Type_File,
  SYS_Handle_Type_Thread,
  SYS_Handle_Type_Barrier,
};

typedef struct SYS_Handle {
  SYS_Handle_Type type;
  U64             value;
} SYS_Handle;

typedef SYS_Handle SYS_File;
typedef SYS_Handle SYS_Thread;
typedef SYS_Handle SYS_Barrier;

global SYS_Barrier SYS_Barrier_None = { .type = SYS_Handle_Type_Barrier, .value = 0 };

// ------------------------------------------------------------
// #-- System API

link_function SYS_Context *     sys_context                          (void);
link_function SYS_NUMA_Layout * sys_numa_layout                      (void);
link_function void              sys_stream_write                     (Str08 buffer, SYS_Stream stream);
link_function void              sys_panic                            (Str08 reason);
link_function Local_Time        sys_local_time_utc                   (void);

link_function U08 *             sys_memory_reserve                   (U64 bytes);
link_function void              sys_memory_unreserve                 (void *virtual_base, U64 bytes);
link_function void              sys_memory_commit                    (void *virtual_base, U64 bytes, SYS_Commit_Flag mode);
link_function void              sys_memory_uncommit                  (void *virtual_base, U64 bytes);

link_function B32               sys_directory_create                 (Str08 folder_path);
link_function B32               sys_directory_delete                 (Str08 folder_path);

link_function SYS_File          sys_file_open                        (Str08 file_path, SYS_File_Access_Flag flags);
link_function void              sys_file_close                       (SYS_File *file);
link_function U64               sys_file_size                        (SYS_File *file);
link_function void              sys_file_read                        (SYS_File *file, Range1_U64 range, void *data);
link_function void              sys_file_write                       (SYS_File *file, Range1_U64 range, void *data);
link_function SYS_File_Map      sys_file_map                         (SYS_File *file, Range1_U64 range);
link_function void              sys_file_unmap                       (SYS_File_Map *map);

link_function SYS_Thread        sys_thread_launch                    (SYS_Thread_Entry_Point *entry_point, void *user_data);
link_function void              sys_thread_join                      (SYS_Thread *thread);
link_function U64               sys_thread_id                        (void);
link_function void              sys_thread_set_name                  (Str08 name);

link_function SYS_Barrier       sys_barrier_init                     (U32 count);
link_function void              sys_barrier_wait                     (SYS_Barrier *barrier);
link_function void              sys_barrier_destroy                  (SYS_Barrier *barrier);

link_function F64               sys_performance_clock_to_nanoseconds (void);
link_function U64               sys_performance_clock_now            (void);

link_function void              sys_entry_point                      (void);

#define SYS_File_Scope(file_, str_, mode_) \
  Defer_Scope(*(file_) = sys_file_open(str_, mode_), sys_file_close(file_))

#define SYS_File_Map_Scope(map_, file_, range_) \
  Defer_Scope(*(map_) = sys_file_map(file_, range_), sys_file_unmap(map_))
