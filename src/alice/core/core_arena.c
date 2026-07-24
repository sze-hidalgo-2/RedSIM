// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Address, Pointer Operations

function U64 address_align(U64 address, U64 align) {
  Assert(!(align & (align - 1)), "align is not a power of two");

  U64 remainder = address & (align - 1);
  if (remainder) {
    address += align - remainder;
  }

  return address;
}

function U08 *pointer_align(void *pointer, U64 align) {
  U64 address = (U64)pointer;
  address = address_align(address, align);
  pointer = (U08 *)address;
  return pointer;
}

function U08 *pointer_offset_bytes(void *pointer, I64 bytes) {
  return (((U08 *)pointer) + bytes);
}

// ------------------------------------------------------------
// #-- Arena

function U08 *arena_chunk_allocate(Arena_Chunk *chunk, U64 bytes, Arena_Push *alloc) {
  U08 *alloc_begin  = pointer_align(chunk->current, alloc->align);
  U08 *alloc_end    = alloc_begin + bytes;

  // NOTE(cmat): If we overstep the boundary of the current page,
  // - commit a new page.
  if (alloc_end > chunk->next_page) {

    // NOTE(cmat): If we run out of reserved memory, return zero to signal
    // - that a new chunk should be allocated.
    if ((alloc_end - chunk->base_memory) >= chunk->reserved) {
      return 0;
    } else {
      U64 page_bytes = sys_context()->mmu_page_bytes;
      U64 grow_bytes = address_align((U64)(alloc_end - chunk->next_page), page_bytes);
  
      sys_memory_commit(chunk->next_page, grow_bytes, SYS_Commit_Flag_Read | SYS_Commit_Flag_Write);
      chunk->next_page += grow_bytes;
    }
  }

  chunk->current = alloc_end;

  // NOTE(cmat): Un-poison allocation
  asan_unpoison_region(alloc_begin, bytes);

  // NOTE(cmat): Zero-initialize memory, if requested.
  if (alloc->flags & Arena_Push_Flag_Zero_Init) {
    memory_fill(alloc_begin, 0, bytes);
  }

  return alloc_begin;
}

function void arena_chunk_deallocate(Arena_Chunk *chunk) {
  Arena_Chunk chunk_header = {
    .header = {
#if BUILD_DEBUG
      .magic        = arena_chunk_magic,
#endif
    },

    .base_memory    = chunk->base_memory,
    .next_page      = chunk->base_memory,
    .current        = chunk->base_memory,
    .reserved       = chunk->reserved,
    .prev           = chunk->prev,
    .next           = chunk->next,
  };

  U08  *uncommit_base   = chunk->base_memory;
  U64   uncommit_bytes  = (U64)(chunk_header.next_page - chunk->base_memory);

  asan_poison_region  (uncommit_base, uncommit_bytes);
  sys_memory_uncommit (uncommit_base, uncommit_bytes);

  Arena_Push header_alloc = { .align = Arena_Alignment_Default, .flags = 0 };
  Arena_Chunk *new_header = (Arena_Chunk *)arena_chunk_allocate(&chunk_header, sizeof(Arena_Chunk), &header_alloc);
  memory_copy(new_header, &chunk_header, sizeof(Arena_Chunk)); 
}

function Arena_Chunk *arena_chunk_init(Arena_Chunk *prev, U64 reserve_bytes) {
  U64 page_bytes = sys_context()->mmu_page_bytes;
  
  reserve_bytes = reserve_bytes + sizeof(Arena_Chunk);
  reserve_bytes = address_align(reserve_bytes, page_bytes);

  Arena_Chunk chunk;
  Zero_Fill(&chunk);
  
#if BUILD_DEBUG
  chunk.header.magic = arena_chunk_magic;
#endif

  chunk.base_memory = sys_memory_reserve(reserve_bytes);
  chunk.next_page   = chunk.base_memory;
  chunk.reserved    = reserve_bytes;
  chunk.current     = chunk.base_memory;
  chunk.prev        = prev;
  chunk.next        = 0;

  // NOTE(cmat): Poison reserved memory region with address sanitizer.
  asan_poison_region(chunk.base_memory, reserve_bytes);
  
  // NOTE(cmat): Push chunk header first.
  Arena_Push header_alloc = { .align = Arena_Alignment_Default, .flags = 0 };
  Arena_Chunk *header = (Arena_Chunk *)arena_chunk_allocate(&chunk, sizeof(chunk), &header_alloc);
  memory_copy(header, &chunk, sizeof(Arena_Chunk));

  if (prev) {
    prev->next = header;
  }

  return (Arena_Chunk *)header;
}

function Arena_Chunk *arena_chunk_destroy(Arena_Chunk *chunk) {
  Arena_Chunk *prev = chunk->prev;
  if (prev) {
    chunk->prev->next = 0;
  }
  
  U08 *base_memory    = chunk->base_memory;
  U64 uncommit_bytes  = chunk->next_page - chunk->base_memory;
  U64 unreserve_bytes = chunk->reserved;
  
  asan_poison_region   (base_memory, uncommit_bytes);
  sys_memory_uncommit  (base_memory, uncommit_bytes);
  sys_memory_unreserve (base_memory, unreserve_bytes);

  return prev;
}

function void arena_init_ext(Arena *arena, Arena_Init *config) {
  Assert(!arena->first_chunk && !arena->last_chunk, "reinitializing arena");
  
  Zero_Fill(arena);
  arena->flags        = config->flags;
  arena->initialized  = 1;
 
  if (config->reserve_initial) {
    arena->first_chunk = arena_chunk_init(0, config->reserve_initial);
    arena->last_chunk = arena->first_chunk; 
  }
}

function void arena_destroy(Arena *arena) {
  Arena_Chunk *it = arena->last_chunk;
  while (it) it = arena_chunk_destroy(it);
  Zero_Fill(arena);  
}


// NOTE(cmat): Try pushing an allocation to the current chunk.
// - If the chunk doesn't have enough reserved virtual memory,
// - create a new chunk and chain.
// -
// - This may seem counterintuitive at first; after all, if we're
// - creating a a new chunk, surely we must be leaving gaps in memory.
// - The trick however is, the range in the chunk is only reserved, comitted.
// - This means, in a situation such as:
// -
// - [ ******-------- ] -> [ ************************ ]
// -      OLD CHUNK               NEW CHUNK
// -
// - The region marked with `-` is reserved by the MMU, but not actually comitted
// - to phsyical memory space.
// - Also to note, if the new allocation size to chain is less than the chunk size,
// - the chunk size is allocated.
//
// - [ ******-------- ] -> [ ***********--- ]
// -      OLD CHUNK            NEW CHUNK
// -
// - In cases where virtual memory is limitted or non-existant
// - like WASM, you can use the flag Arena_Flag_Backtrack_Before_Chaining.
// - This iterates through older chunks and try to place our allocations there.
// - Example:
// -
// - [ ******-------- ] -> [ ***********--- ]
// -      OLD CHUNK            NEW CHUNK
// -
// - We have a new allocation, '*****'.
// - We backtrack through our doubly linked list, until we find a chunk
// - that doesn't fail to allocate.
// -
// - [ ******-------- ] -> [ ***********--- ]
// -         ^ insert new alloc here.
// -      OLD CHUNK            NEW CHUNK

function U08 *arena_allocate_within_new_chunk(Arena *arena, U64 bytes, Arena_Push *config) {
    U64 default_chunk_bytes = arena_default_chunk_bytes;
    if (sizeof(Arena_Chunk) < default_chunk_bytes) {
      default_chunk_bytes -= sizeof(Arena_Chunk);
    }

    U64 chunk_reserve = bytes > default_chunk_bytes ? bytes : default_chunk_bytes;

    arena->last_chunk = arena_chunk_init(arena->last_chunk, chunk_reserve);
    Assert(arena->last_chunk, "failed to allocate new chunk");
    
    U08 *user_allocation = arena_chunk_allocate(arena->last_chunk, bytes, config);
    Assert(user_allocation, "failed to allocate memory");

    return user_allocation;
}

function U08 *arena_push_ext(Arena *arena, U64 bytes, Arena_Push *config) {
  U64 default_chunk_bytes = arena_default_chunk_bytes;
  if (sizeof(Arena_Chunk) < default_chunk_bytes) {
    default_chunk_bytes -= sizeof(Arena_Chunk);
  }

  U64 chunk_reserve = bytes > default_chunk_bytes ? bytes : default_chunk_bytes;
  
  // NOTE(cmat): Initialize first chunk if the arena is empty.
  // - This branch is predictable after intialization.
  If_Unlikely(!arena->first_chunk) {
    arena->first_chunk = arena_chunk_init(0, chunk_reserve);
    arena->last_chunk = arena->first_chunk; 
  }
 
  U08 *user_allocation = arena_chunk_allocate(arena->last_chunk, bytes, config);
  if (!user_allocation) {
    if (arena->flags & Arena_Flag_Allow_Chaining) {
      if (arena->flags & Arena_Flag_Backtrack_Before_Chaining) {

        // TODO(cmat): In pathological cases, this seems bad since we're iterating at worst
        // - N-times before finding an open slot.
        // - My gut feeling is we can say the same thing here, as in an open-slot hash-table,
        // - where if you have to iterate through the whole table, your hash is definitely
        // - terrible. If you're iterating this much through your arena chunks, you definitely did
        // - a bad job managing the arena's lifetime. Maybe logging a warning or straight up throwing
        // - an assert at N > 10 could be ok.
        for (Arena_Chunk *it = arena->last_chunk->prev; it != 0; it = it->prev) {
          user_allocation = arena_chunk_allocate(it, bytes, config);
          if (user_allocation) break;
        }

        // NOTE(cmat): If no chunks were open, we chain.
        if (!user_allocation)
          user_allocation = arena_allocate_within_new_chunk(arena, bytes, config);
      } else {
        user_allocation = arena_allocate_within_new_chunk(arena, bytes, config);
      }
    } else {
#if BUILD_DEBUG
      log_fatal("arena error at: %.*s", str_expand(config->meta_caller_info));
#endif
      Assert(0, "arena exceeded reserved limit, and does not have chaining enabled");
    }
  }

  return user_allocation;
}

function Str08 arena_push_str08(Arena *arena, Str08 str) {
  Str08 result = { };
  result.len = str.len;
  result.txt = arena_push_size(arena, str.len);
  memory_copy(result.txt, str.txt, str.len);
    
  return result;
}


function void arena_clear(Arena *arena) {
  // NOTE(cmat): We destroy all chunks, but only deallocate the first one.
  // - If you wish to destroy the first chunk as well, call arena_destroy
  // - (althought that does require calling arena_init again).
  if (arena->last_chunk) {
    Arena_Chunk *it = arena->last_chunk;
    while (it != arena->first_chunk) {
      it = arena_chunk_destroy(it);
    }
    arena_chunk_deallocate(it);
    arena->last_chunk = arena->first_chunk;
  }
}

function Arena_Temp arena_temp_start(Arena *arena) {
  Arena_Temp result;
  if (!arena->last_chunk) {
    result = (Arena_Temp) { .arena = arena };
  } else {
    result = (Arena_Temp) {
        .arena = arena,
        .rollback_chunk   = arena->last_chunk,
        .rollback_current = arena->last_chunk->current,
        .rollback_page    = arena->last_chunk->next_page
    };
  }

  return result;
}

function void arena_temp_end(Arena_Temp *temporary) {
  if (!temporary->rollback_chunk) {
    arena_clear(temporary->arena);
    Zero_Fill(temporary);
  } else {
    Arena_Chunk *it = temporary->arena->last_chunk;
    while (it != temporary->rollback_chunk) it = arena_chunk_destroy(it);

    sys_memory_uncommit(temporary->rollback_page, it->next_page - temporary->rollback_page);
    it->next_page = temporary->rollback_page;
    it->current = temporary->rollback_current;
   
    temporary->arena->last_chunk = temporary->rollback_chunk;
    Zero_Fill(temporary);
  }
}

