// ------------------------------------------------------------
// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
//  Scalar Operations

#if ARCH_X86
# include <emmintrin.h>
# include <xmmintrin.h>

  function F32 f32_sqrt(F32 x)  {
    __m128 v    = _mm_set_ss(x);
    v           = _mm_sqrt_ss(v);
    F32 result  = _mm_cvtss_f32(v);
    return result;
  }

  function F32 f32_floor(F32 x) {
    __m128 v = _mm_set_ss(x);
    I32 i    = (I32)_mm_cvttss_si32(v);
    F32 t    = (F32)i;
    t        = (t > x) ? (t - 1.f) : t;
    return t;
  }

  function F32 f32_ceil(F32 x) {
    __m128 v = _mm_set_ss(x);
    I32 i    = (I32)_mm_cvttss_si32(v);
    F32 t    = (F32)i;
    t        = (t < x) ? (t + 1.f) : t;
    return t;
  }

  function F64 f64_sqrt(F64 x)  {
    __m128d v   = _mm_set_sd(x);
    v           = _mm_sqrt_sd(v, v);
    F64 result  = _mm_cvtsd_f64(v);
    return result;
  }
 
  function F64 f64_floor(F64 x) {
    __m128d v = _mm_set_sd(x);
    I64 i     = (I64)_mm_cvttsd_si64(v);
    F64 t     = (F64)i;
    t         = (t > x) ? (t - 1.f) : t;
    return t;
  }

  function F64 f64_ceil(F64 x) {
    __m128d v = _mm_set_sd(x);
    I64 i     = (I64)_mm_cvttsd_si64(v);
    F64 t     = (F64)i;
    t         = (t < x) ? (t + 1.f) : t;
    return t;
  }

#else
  // TODO(cmat): ARM implementation.
# error "unsupported architecture"

#endif

// ------------------------------------------------------------
// #-- Triggonometry

// NOTE(cmat): Based on minimax approximation, degree 7.
// Coefficients: https://gist.github.com/publik-void/067f7f2fef32dbe5c27d6e215f824c91#sin-rel-error-minimized-degree-7
function F32 f32_sin(F32 x) {
  F32 s  = 0;
  F32 x2 = 0;

  x        -= f32_pi_2 * f32_floor((x + f32_pi) / f32_pi_2);  // NOTE(cmat): wrap between [-pi, pi)
  s         = f32_sign(x);                                    // NOTE(cmat): Store sign for end result.
  x         = f32_abs(x);                                     // NOTE(cmat): wrap between [0, pi)
  x         = (x > f32_pi_h) ? (f32_pi - x) : x;              // NOTE(cmat): wrap between [0, pi/2)
  x2        = x * x;
  x         = x   * (+0.999996615908002773079325846913220383    +
              x2  * (-0.16664828381895056829366054140948866     +
              x2  * (+0.00830632522715989396465411782615901079  -
                     +0.00018363653976946785297280224158683484  *  x2)));

  return s * x;
}

function F32 f32_cos(F32 x) {
  F32 result = f32_sin(x + f32_pi_h);
  return result;
}

function F32 f32_tan(F32 x) {
  F32 result = f32_div_safe(f32_sin(x), f32_cos(x));
  return result;
}

