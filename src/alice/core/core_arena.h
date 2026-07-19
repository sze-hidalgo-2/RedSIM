// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Address, Pointer Operations
function U64  address_align         (U64 address, U64 align);
function U08 *pointer_align         (void *pointer, U64 align);
function U08 *pointer_offset_bytes  (void *pointer, I64 bytes);

// ------------------------------------------------------------
// #-- Arena

// Arena allocator.
// Arena allocators are implemented using two tricks: MMU, and a doubly linked list
// of reserved chunks. Reserving virtual memory is "free" on most systems
// (you can reserve terabytes of virtual memory without running out),
// we leverage that fact by first reserving large chunks of virtual memory, then comitting as needed.
// If we run out of reserved virtual memory, we allocate a new chunk and add it
// to the linked list.
// -
// Things become tricky when using something like WASM, where there is no
// virtual memory address space (everything is just one, linearly growing
// "physical" address space). The trick in that case is to reserve smaller chunks,
// and treat reserving memory as essentially comitting memory (commit does nothing).
// For now, this is an exception for WASM, and hopefully (someday), the WASM commity
// will wake up from their sleep and actually implement a proper memory model.
// Until then, this is a trick that works.


global const U64 arena_chunk_magic = u64_pack('A','R','E','N','A','C','H','K');

typedef U32 Arena_Push_Flag;
enum {
    Arena_Push_Flag_Zero_Init = 1 << 0
};

typedef struct { 
    U32             align;
    Arena_Push_Flag flags;
} Arena_Push;

enum {
    Arena_Alignment_Default = sizeof(void*),
    Arena_Push_Flags_Default = Arena_Push_Flag_Zero_Init,
};


#if BUILD_DEBUG
typedef struct Arena_Chunk_Header { U64 magic; } Arena_Chunk_Header;
#else
typedef struct Arena_Chunk_Header { } Arena_Chunk_Header;
#endif

typedef struct Arena_Chunk {
    Arena_Chunk_Header  header;
    
    U08                *base_memory;
    U08                *next_page;
    U08                *current;
    U64                 reserved;
    
    struct Arena_Chunk *prev;
    struct Arena_Chunk *next;
} Arena_Chunk;

typedef U32 Arena_Flag;
enum {
    // NOTE(cmat): Allows growth of arena beyond initial reserved block of memory.
    Arena_Flag_Allow_Chaining = 1 << 0,
    
    // NOTE(cmat): Always reserve and commit the whole chunk upon creation.
    Arena_Flag_Commit_Whole_Chunk = 1 << 1,
    
    // NOTE(cmat): If an allocation doesn't fit in the current chunk,
    // before creating a new chunk, backtrack through previous chunks
    // to find one which accepts our allocation.
    // The use-case for this, is for platforms without MMU support (WASM),
    // or where address space is very limited (32bit).
    // Chaining must be enabled for this to have effect!
    Arena_Flag_Backtrack_Before_Chaining = 1 << 2,
    
    // NOTE(cmat): On WASM, we enable backtracking by default
    // since wasting "virtual" memory is wasting phyisical committed memory.
#if OS_WASM
    Arena_Flag_Defaults = Arena_Flag_Allow_Chaining | Arena_Flag_Backtrack_Before_Chaining
#else
    Arena_Flag_Defaults = Arena_Flag_Allow_Chaining
#endif
};

#if OS_WASM
#define arena_default_chunk_bytes u64_kilobytes(64)
#else
#define arena_default_chunk_bytes u64_megabytes(64)
#endif


typedef struct Arena_Init {
    Arena_Flag flags;
    U64        reserve_initial;
} Arena_Init;

typedef struct Arena {
    Arena_Flag      flags; 
    B32             initialized;
    Arena_Chunk    *first_chunk;
    Arena_Chunk    *last_chunk;
} Arena;

function void   arena_init_ext    (Arena *arena, Arena_Init *init);
function void   arena_destroy     (Arena *arena);
function U08   *arena_push_ext    (Arena *arena, U64 bytes, Arena_Push *push);
function Str08  arena_push_str08  (Arena *arena, Str08 str);
function void   arena_clear       (Arena *arena);

#define arena_init(arena, ...) \
  arena_init_ext((arena), &(Arena_Init) { .flags = Arena_Flag_Defaults, .reserve_initial = 0, __VA_ARGS__ })

#define arena_push_size(arena, bytes, ...) \
  arena_push_ext((arena), (bytes), &(Arena_Push) { .align = Arena_Alignment_Default, .flags = Arena_Push_Flags_Default, __VA_ARGS__ })

#define arena_push_type(arena, type, ...) \
  (type *)arena_push_size((arena), sizeof(type),  __VA_ARGS__)

#define arena_push_count(arena, type, count, ...) \
  (type *)arena_push_size((arena), (count) * sizeof(type), __VA_ARGS__)

typedef struct Arena_Temp {
    Arena       *arena;
    Arena_Chunk *rollback_chunk;
    U08         *rollback_current;
    U08         *rollback_page;
} Arena_Temp;

function Arena_Temp arena_temp_start (Arena *arena);
function void       arena_temp_end   (Arena_Temp *temp);

#define Arena_Temp_Scope(arena_, temp_name_) \
for (Arena_Temp temp_name_ = arena_temp_start(arena_); temp_name_.arena; arena_temp_end(&temp_name_))

