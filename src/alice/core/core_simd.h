// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- SIMD

#if ARCH_X86
#include <immintrin.h>

// NOTE(cmat): Basic 4x wide types
typedef union {
  __m128  simd;
  F32     data[4];
} F32_X04;

// NOTE(cmat): Basic 4x wide ops
force_inline function F32_X04  f32_x04_load                        (F32 *ptr)                                { return (F32_X04)  { .simd = _mm_loadu_ps(ptr) }; }
force_inline function F32_X04  f32_x04_load_aligned                (F32 *ptr)                                { return (F32_X04)  { .simd = _mm_load_ps(ptr) }; }
force_inline function F32_X04  f32_x04_load_f32                    (F32 x)                                   { return (F32_X04)  { .simd = _mm_set1_ps(x) }; }

force_inline function void      f32_x04_store                      (F32 *ptr, F32_X04 value)                 { _mm_storeu_ps(ptr, value.simd); }
force_inline function void      f32_x04_store_aligned              (F32 *ptr, F32_X04 value)                 { _mm_store_ps(ptr, value.simd); }

force_inline function F32_X04  f32_x04_add                         (F32_X04 lhs, F32_X04 rhs)                { return (F32_X04)  { .simd = _mm_add_ps(lhs.simd, rhs.simd) }; }
force_inline function F32_X04  f32_x04_sub                         (F32_X04 lhs, F32_X04 rhs)                { return (F32_X04)  { .simd = _mm_sub_ps(lhs.simd, rhs.simd) }; }
force_inline function F32_X04  f32_x04_mul                         (F32_X04 lhs, F32_X04 rhs)                { return (F32_X04)  { .simd = _mm_mul_ps(lhs.simd, rhs.simd) }; }
force_inline function F32_X04  f32_x04_div                         (F32_X04 lhs, F32_X04 rhs)                { return (F32_X04)  { .simd = _mm_div_ps(lhs.simd, rhs.simd) }; }
force_inline function F32_X04  f32_x04_square_root                 (F32_X04 x)                               { return (F32_X04)  { .simd = _mm_sqrt_ps(x.simd) }; }
force_inline function F32_X04  f32_x04_fused_mul_add               (F32_X04 a, F32_X04 b, F32_X04 c)         { return (F32_X04)  { .simd = _mm_fmadd_ps(a.simd, b.simd, c.simd) }; }
force_inline function F32_X04  f32_x04_fused_mul_sub               (F32_X04 a, F32_X04 b, F32_X04 c)         { return (F32_X04)  { .simd = _mm_fmsub_ps(a.simd, b.simd, c.simd) }; }

#else
#error "Unsupported Architecture for SIMD"

#endif
