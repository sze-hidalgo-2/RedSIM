// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Build Flags

#if !defined(BUILD_ASSERT)
# error "BUILD_ASSERT not defined"
# define BUILD_ASSERT 0
#endif

#if !defined(BUILD_PROFILE)
# error "BUILD_PROFILE not defined"
# define BUILD_PROFILE 0
#endif

// ------------------------------------------------------------
// #-- Codebase Markup

#define function        static
#define link_function   extern

#define global          static
#define link_global     extern

#define local_storage   static
#define thread_storage  _Thread_local

#define Unused_Var(x) ((void)(x))

// ------------------------------------------------------------
// #-- Compiler Identification

#define COMPILER_MSVC   0
#define COMPILER_CLANG  0
#define COMPILER_GCC    0

#if defined(_MSC_VER)
# undef  COMPILER_MSVC
# define COMPILER_MSVC 1
#elif defined(__clang__)
# undef  COMPILER_CLANG
# define COMPILER_CLANG 1
#elif defined(__GNUC__) 
# undef  COMPILER_GCC
# define COMPILER_GCC 1
#else
# error "unsupported compiler"
#endif

// ------------------------------------------------------------
// #-- OS Identification

#define OS_WIN32 0
#define OS_MACOS 0
#define OS_LINUX 0
#define OS_WASM  0

#if defined(__wasm__)
# undef OS_WASM
# define OS_WASM 1
#elif defined(_WIN32) || defined(WIN32)
# undef  OS_WIN32
# define OS_WIN32 1
#elif defined(__APPLE__)
# undef  OS_MACOS
# define OS_MACOS 1
#elif defined(__linux__)
# undef  OS_LINUX
# define OS_LINUX 1
#else
# error "unsupported operating system"
#endif

// ------------------------------------------------------------
// #-- Architecture Identification

#define ARCH_X86  0
#define ARCH_ARM  0
#define ARCH_WASM 0

#if COMPILER_MSVC
# if defined(__x86_64__)
#  undef  ARCH_X86
#   define ARCH_X86 1
# elif defined(_M_ARM)
#   undef  ARCH_ARM
#   define ARCH_ARM 1
# elif defined(__wasm__)
#   undef  ARCH_WASM
#   define ARCH_WASM 1
# else
#   error "unsupported architecture"
# endif
#elif COMPILER_CLANG || COMPILER_GCC
# if defined(__x86_64__)
#   undef  ARCH_X86
#   define ARCH_X86 1
# elif __ARM_ARCH_ISA_A64
#   undef ARCH_ARM
#   define ARCH_ARM 1
# elif defined(__wasm__)
#   undef ARCH_WASM
#   define ARCH_WASM 1
# else
#   error "unsupported architecture"
# endif
#endif

#if ARCH_WASM
# define ARCH_ADDRESS_SIZE 32
#else
# define ARCH_ADDRESS_SIZE 64
#endif

// ------------------------------------------------------------
// #-- Compiler Warnings

#if COMPILER_CLANG
# pragma clang diagnostic ignored "-Winitializer-overrides"
# pragma clang diagnostic ignored "-Wdeprecated-declarations"
# pragma clang diagnostic ignored "-Wformat-invalid-specifier"
# pragma clang diagnostic ignored "-Wmissing-braces"
# pragma clang diagnostic ignored "-Wunused-function"
#endif

// ------------------------------------------------------------
// #-- Macro Utilities

#define Macro_Counter __COUNTER

#define Macro_Stringize_2(x)        #x
#define Macro_Stringize_1(x)        Macro_Stringize_2(x)
#define Macro_Stringize(x)          Macro_Stringize_1(x)

#define Macro_Join_2(x, y)          x ## y
#define Macro_Join_1(x, y)          Macro_Join_2(x, y)
#define Macro_Join(x, y)            Macro_Join_1(x, y)

#define Macro_Max(x_, y_)           ((x) > (y) ? (y) : (x))
#define Macro_Min(x_, y_)           ((x) > (y) ? (y) : (x))

#define Macro_Swap(type_, x_, y_)   do { type_ t = (x_); x_ = y_; y_ = t; } while(0)

// ------------------------------------------------------------
// #-- Compile-Time Assertion

#define Assert_Compiler(condition) _Static_assert(condition, Macro_Stringize(condition))

// ------------------------------------------------------------
// #-- Branch Prediction

#if COMPILER_MSVC
# define If_Likely(x)     if(!!(x))
# define If_Unlikely(x)   if(!!(x))
#elif COMPILER_CLANG || COMPILER_GCC
# define If_Likely(x)     if(__builtin_expect(!!(x), 1))
# define If_Unlikely(x)   if(__builtin_expect(!!(x), 0))
#endif

// ------------------------------------------------------------
// #-- Clang Address Sanitizer

#define ASAN_ENABLED 0
#define asan_poison_region(address, size)   (void)(address), (void)(size)
#define asan_unpoison_region(address, size) (void)(address), (void)(size)

#ifndef __has_feature
# define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
  void __asan_poison_memory_region    (void const volatile *addr, __SIZE_TYPE__ size);
  void __asan_unpoison_memory_region  (void const volatile *addr, __SIZE_TYPE__ size);

# undef ASAN_ENABLED
# undef asan_poison_region
# undef asan_unpoison_region
# define ASAN_ENABLED 1

# define asan_poison_region(address, size)   __asan_poison_memory_region(address, size)
# define asan_unpoison_region(address, size) __asan_unpoison_memory_region(address, size)
#endif

// ------------------------------------------------------------
// #-- Primitive Types

#if COMPILER_GCC || COMPILER_CLANG
  typedef __INT8_TYPE__   I08;
  typedef __INT16_TYPE__  I16;
  typedef __INT32_TYPE__  I32;
  typedef __INT64_TYPE__  I64;
  typedef __UINT8_TYPE__  U08;
  typedef __UINT16_TYPE__ U16;
  typedef __UINT32_TYPE__ U32;
  typedef __UINT64_TYPE__ U64;

#elif COMPILER_MSVC
  typedef __int8   I08;
  typedef __int16  I16;
  typedef __int32  I32;
  typedef __int64  I64;
  typedef __uint8  U08;
  typedef __uint16 U16;
  typedef __uint32 U32;
  typedef __uint64 U64;
#endif

#if ARCH_ADDRESS_SIZE == 32
  typedef U32 UAddr;
  typedef I32 IAddr;
#elif ARCH_ADDRESS_SIZE == 64
  typedef U64 UAddr;
  typedef I64 IAddr;
#endif

typedef float  F32;
typedef double F64;

Assert_Compiler(sizeof(I08) == 1);
Assert_Compiler(sizeof(I16) == 2);
Assert_Compiler(sizeof(I32) == 4);
Assert_Compiler(sizeof(I64) == 8);

Assert_Compiler(sizeof(U08) == 1);
Assert_Compiler(sizeof(U16) == 2);
Assert_Compiler(sizeof(U32) == 4);
Assert_Compiler(sizeof(U64) == 8);

Assert_Compiler(sizeof(F32) == 4);
Assert_Compiler(sizeof(F64) == 8);

typedef I08 B08;
typedef I16 B16;
typedef I32 B32;
typedef I64 B64;

// ------------------------------------------------------------
// #-- Keywords

#if COMPILER_MSVC
# define force_inline inline __forceinline
#elif COMPILER_CLANG || COMPILER_GCC
# define force_inline inline __attribute__((always_inline))
#endif

#define Align_As(x_)              _Alignas(x_)
#define Offset_Of(type_, member_) ((U64)&(((type_ *)0)->member_))

// ------------------------------------------------------------
// #-- Memory Operations

#if COMPILER_CLANG || COMPILER_GCC
  force_inline function void memory_copy    (void *dst, void *src, U64 bytes) { __builtin_memcpy(dst, src, bytes);  }
  force_inline function void memory_fill    (void *dst, U08 fill, U64 bytes)  { __builtin_memset(dst, fill, bytes); }
  force_inline function B32  memory_match   (void *lhs, void *rhs, U64 bytes) { return (__builtin_memcmp(lhs, rhs, bytes) == 0); }
#elif COMPILER_MSVC
# pragma intrinsic(__movsb)
# pragma intrinsic(__stosb)
  force_inline function void memory_copy    (void *dst, void *src, U64 bytes) { __movsb(dst, src, bytes); }
  force_inline function void memory_fill    (void *dst, U08 fill, U64 bytes)  { __stosb(dst, fill, bytes); }

  force_inline function B32 memory_match(void *lhs, void *rhs, U64 bytes) {
    B32  result  = 1;
    U08 *lhs_u08 = (U08 *)lhs;
    U08 *rhs_u08 = (U08 *)rhs;
    for (U64 it = 0; it < bytes; it++) {
      if (lhs_u08[it] != rhs_u08[it]) {
        result = 0;
        break;
      }
    }
    
    return result;
  }
#endif

#define Zero_Fill(type_ptr) memory_fill(type_ptr, 0, sizeof(*(type_ptr)))

// ------------------------------------------------------------
// #-- Stack-Allocated Array Operations

#define Stack_Array_Len(array_)   (sizeof(array_) / sizeof((array_)[0]))
#define Stack_Array_Zero(array_)  memory_fill(array_, 0, sizeof(array_))

// ------------------------------------------------------------
// #-- Atomic Operations

#if COMPILER_GCC || COMPILER_CLANG
  force_inline function U32  atomic_read_u32       (volatile U32 *x)             { return __atomic_load_n(x, __ATOMIC_SEQ_CST);               }
  force_inline function I32  atomic_read_i32       (volatile I32 *x)             { return __atomic_load_n(x, __ATOMIC_SEQ_CST);               }
  force_inline function U32  atomic_write_u32      (volatile U32 *x, U32 value)  { return __atomic_exchange_n(x, value, __ATOMIC_SEQ_CST);    }
  force_inline function I32  atomic_write_i32      (volatile I32 *x, I32 value)  { return __atomic_exchange_n(x, value, __ATOMIC_SEQ_CST);    }
  force_inline function U32  atomic_increment_u32  (volatile U32 *x)             { return __atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST) + 1;     }
  force_inline function I32  atomic_increment_i32  (volatile I32 *x)             { return __atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST) + 1;     }
  force_inline function U32  atomic_decrement_u32  (volatile U32 *x)             { return __atomic_fetch_sub(x, 1, __ATOMIC_SEQ_CST) - 1;     }
  force_inline function I32  atomic_decrement_i32  (volatile I32 *x)             { return __atomic_fetch_sub(x, 1, __ATOMIC_SEQ_CST) - 1;     }

  force_inline function U64  atomic_read_u64       (volatile U64 *x)             { return __atomic_load_n(x, __ATOMIC_SEQ_CST);               }
  force_inline function I64  atomic_read_i64       (volatile I64 *x)             { return __atomic_load_n(x, __ATOMIC_SEQ_CST);               }
  force_inline function U64  atomic_write_u64      (volatile U64 *x, U64 value)  { return __atomic_exchange_n(x, value, __ATOMIC_SEQ_CST);    }
  force_inline function I64  atomic_write_i64      (volatile I64 *x, I64 value)  { return __atomic_exchange_n(x, value, __ATOMIC_SEQ_CST);    }
  force_inline function U64  atomic_increment_u64  (volatile U64 *x)             { return __atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST) + 1;     }
  force_inline function I64  atomic_increment_i64  (volatile I64 *x)             { return __atomic_fetch_add(x, 1, __ATOMIC_SEQ_CST) + 1;     }
  force_inline function U64  atomic_decrement_u64  (volatile U64 *x)             { return __atomic_fetch_sub(x, 1, __ATOMIC_SEQ_CST) - 1;     }
  force_inline function I64  atomic_decrement_i64  (volatile I64 *x)             { return __atomic_fetch_sub(x, 1, __ATOMIC_SEQ_CST) - 1;     }
  force_inline function U64  atomic_add_u64        (volatile U64 *x, U64 add)    { return __atomic_fetch_add(x, add, __ATOMIC_SEQ_CST) + add; }
  force_inline function I64  atomic_add_i64        (volatile I64 *x, I64 add)    { return __atomic_fetch_add(x, add, __ATOMIC_SEQ_CST) + add; }

#elif COMPILER_MSVC
  force_inline function U32  atomic_read_u32       (volatile U32 *x)             { return *x;                                                }
  force_inline function I32  atomic_read_i32       (volatile I32 *x)             { return *x;                                                }
  force_inline function U32  atomic_write_u32      (volatile U32 *x, U32 value)  { InterlockedExchange((I32 *)x, (I32)value);                }
  force_inline function I32  atomic_write_i32      (volatile I32 *x, I32 value)  { InterlockedExchange(x, value);                            }
  force_inline function U32  atomic_increment_u32  (volatile U32 *x)             { return InterlockedIncrement((I32 *)x);                    }
  force_inline function I32  atomic_increment_i32  (volatile I32 *x)             { return InterlockedIncrement(x);                           }
  force_inline function U32  atomic_decrement_u32  (volatile U32 *x)             { return InterlockedDecrement((I32 *)x);                    }
  force_inline function I32  atomic_decrement_i32  (volatile I32 *x)             { return InterlockedDecrement(x);                           }
#endif

// ------------------------------------------------------------
// #-- Spinlock Hint

#if ARCH_X86
# include <immintrin.h>
  force_inline function void spin_lock(void) { _mm_pause();                };
#elif ARCH_ARM
  force_inline function void spin_lock(void) { __asm__ volatile("yield");  };
#elif ARCH_WASM
  force_inline function void spin_lock(void) {                             };
#endif

// ------------------------------------------------------------
// #-- Runtime Assertions

#if BUILD_ASSERT
#define Assert_Header __FILE__ " " Macro_Stringize(__LINE__) ": "
# if COMPILER_GCC || COMPILER_CLANG
#   define Assert(condition, message) do { If_Unlikely (!(condition)) { sys_panic(str08_lit(Assert_Header message "\n")); __builtin_trap(); } } while(0)
# elif COMPILER_MSVC
#   define Assert(condition, message) do { If_Unlikely (!(condition)) { sys_panic(str08_lit(Assert_Header message "\n")); __debugbreak(); } } while(0)
# endif
#else
# define Assert(condition, message) do { } while(0)

#endif

#define Invalid_Default default: { Assert(0, "invalid default in switch"); } break;
#define Not_Implemented Assert(0, "function not implemented");

// ------------------------------------------------------------
// #-- Intrusive Data Structure Operations

#define Stack_Push(first_, node_) Stack_Push_Ext(first_, node_, next)
#define Stack_Push_Ext(first_, node_, next_name_)                                   \
do {                                                                                \
  (node_)->next_name_ = (first_);                                                   \
  (first_) = (node_);                                                               \
} while(0)

#define Stack_Pop(first_) Stack_Pop_Ext(first_, next)
#define Stack_Pop_Ext(first_, next_name_)                                           \
do {                                                                                \
  (first_) = (first_)->next_name_;                                                  \
} while(0)

#define Queue_Push(first_, last_, node_) Queue_Push_Ext(first_, last_, node_, next)
#define Queue_Push_Ext(first_, last_, node_, next_name_)                            \
do {                                                                                \
  if ((first_) == 0) {                                                              \
    (first_) = (node_);                                                             \
    (last_)  = (node_);                                                             \
    (node_)->next_name_ = 0;                                                        \
  } else {                                                                          \
    (last_)->next_name_  = (node_);                                                 \
    (last_)              = (node_);                                                 \
    (node_)->next_name_  = 0;                                                       \
  }                                                                                 \
} while(0)

#define Queue_Pop(first_, last_) Queue_Pop_Ext(first_, last_, next)
#define Queue_Pop_Ext(first_, last_, next_name_)                                    \
do {                                                                                \
  if ((first_) == (last_)) {                                                        \
    (first_) = 0;                                                                   \
    (last_)  = 0;                                                                   \
  } else {                                                                          \
    (first_) = (first_)->next_name_;                                                \
  }                                                                                 \
} while(0)

#define DLL_Push_Back(first_, last_, node_) DLL_Push_Back_Ext(first_, last_, node_, next, prev)
#define DLL_Push_Back_Ext(first_, last_, node_, next_name_, prev_name_)     \
do {                                                                        \
  if ((first_) == 0) {                                                      \
    (first_)            = (node_);                                          \
    (last_)             = (node_);                                          \
    (node_)->next_name_ = 0;                                                \
    (node_)->prev_name_ = 0;                                                \
  } else {                                                                  \
    (last_)->next_name_ = (node_);                                          \
    (node_)->prev_name_ = (last_);                                          \
    (last_)             = (node_);                                          \
    (node_)->next_name_ = 0;                                                \
  }                                                                         \
} while(0)

#define DLL_Push_Front(first_, last_, node_) DLL_Push_Front_Ext(first_, last_, node_, next, prev)
#define DLL_Push_Front_Ext(first_, last_, node_, next_name_, prev_name_) \
  DLL_Push_Back_Ext(last_, first_, node_, prev_name_, next_name_)

#define DLL_Insert_After(first_, last_, insert_at_, node_) DLL_Insert_After_Ext(first_, last_, insert_at_, node_, next, prev)
#define DLL_Insert_After_Ext(first_, last_, insert_at_, node_, next_name_, prev_name_)  \
do {                                                                                    \
  if ((last_) == (insert_at_)) {                                                        \
    DLL_Push_Back_Ext(first_, last_, node_, next_name_, prev_name_);                    \
  } else {                                                                              \
    (node_)->prev_name_                  = (insert_at_);                                \
    (node_)->next_name_                  = (insert_at_)->next_name_;                    \
    (insert_at_)->next_name_->prev_name_ = (node_);                                     \
    (insert_at_)->next_name_             = (node_);                                     \
  }                                                                                     \
} while(0);

#define DLL_Insert_Before(first_, last_, insert_at_, node_) DLL_Insert_Before_Ext(first_, last_, insert_at_, node_, next, prev)
#define DLL_Insert_Before_Ext(first_, last_, insert_at_, node_, next_name_, prev_name_) \
  DLL_Insert_After_Ext(last_, first_, insert_at_, node_, prev_name_, next_name_)

#define DLL_Remove_First(first_, last_) DLL_Remove_First_Ext(first_, last_, next, prev)
#define DLL_Remove_First_Ext(first_, last_, next_name_, prev_name_)                     \
do {                                                                                    \
  if ((first_) == (last_)) {                                                            \
    (first_)  = 0;                                                                      \
    (last_)   = 0;                                                                      \
  } else {                                                                              \
    (first_)              = (first_)->next_name_;                                       \
    (first_)->prev_name_  = 0;                                                          \
  }                                                                                     \
} while(0)

#define DLL_Remove_Last(first_, last_) DLL_Remove_Last_Ext(first_, last_, next, prev)
#define DLL_Remove_Last_Ext(first_, last_, next_name_, prev_name_)  \
  DLL_Remove_First_Ext(last_, first_, prev_name_, next_name_)

#define DLL_Remove(first_, last_, node_) DLL_Remove_Ext(first_, last_, node_, next, prev)
#define DLL_Remove_Ext(first_, last_, node_, next_name_, prev_name_)                    \
do {                                                                                    \
  if ((first_) == (node_)) {                                                            \
    DLL_Remove_First_Ext(first_, last_, next_name_, prev_name_);                        \
  } else if ((last_) == (node_)) {                                                      \
    DLL_Remove_Last_Ext(first_, last_, next_name_, prev_name_);                         \
  } else {                                                                              \
    (node_)->next_name_->prev_name_ = (node_)->prev_name_;                              \
    (node_)->prev_name_->next_name_ = (node_)->next_name_;                              \
  }                                                                                     \
} while(0)

// ------------------------------------------------------------
// #-- Loop Constructs
// NOTE(cmat): Based on Ryan Fleury's work / blog.

// NOTE(cmat): Defer_Scope. Works similar to D's scope(exit); the end statement is executed
// at the end of the scope. #define Defer_Scope_1(defer_id, start, end) for (B32 defer_id = ((start), 0); defer_id == 0; defer_id = ((end), 1))
#define Defer_Scope_1(defer_id, start, end) for (B32 defer_id = ((start), 0); defer_id == 0; defer_id = ((end), 1))
#define Defer_Scope(start, end) Defer_Scope_1(Macro_Join(_defer_id_, __COUNTER__), start, end)

#define Iter_Index(it_, count_) (U64 it_ = 0;             it_ < (count_);     it_++)
#define Iter_Range(it_, range_) (U64 it_ = (range_).min;  it_ < (range_).max; it_++)

// ------------------------------------------------------------
// #-- Local Time

typedef struct {
  U32 year;
  U08 month;
  U08 day;
  U08 hours;
  U08 minutes;
  U08 seconds;
  U32 microseconds;
} Local_Time;

function Local_Time local_time_from_unix_time(U64 unix_seconds, U64 unix_microseconds);

