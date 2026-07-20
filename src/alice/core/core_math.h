// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Convenience Macros
#define Power_2(pot_) (1 << (U64)(pot_))

// ------------------------------------------------------------
// #-- Constants

global const F32 f32_pi          = 3.14159265358979323846f;
global const F32 f32_pi_2        = 6.28318530717958647693f;
global const F32 f32_pi_h        = 1.57079632679489661923f;
global const F32 f32_e           = 2.71828182845904523536f;
global const F32 f32_root_2      = 1.41421356237309504880f;
global const F32 f32_ln_2        = 0.693147180559945309417f;

global const F64 f64_pi          = 3.14159265358979323846;
global const F64 f64_pi_2        = 6.28318530717958647693;
global const F64 f64_pi_h        = 1.57079632679489661923;
global const F64 f64_e           = 2.71828182845904523536;
global const F64 f64_root_2      = 1.41421356237309504880;
global const F64 f64_ln_2        = 0.693147180559945309417;

global const F32 f32_noz_epsilon = 1.0e-6f;

// ------------------------------------------------------------
// #-- Integer Packing

#define u32_pack(a, b, c, d)              ((U32)(d) << 24 | (U32)(c) << 16 | (U32)(b) << 8 | (U32)(a))
#define u64_pack(a, b, c, d, e, f, g, h)  ((U64)(h) << 56 | (U64)(g) << 48 | (U64)(f) << 40 | (U64)(e) << 32 | (U64)(d) << 24 | (U64)(c) << 16 | (U64)(b) << 8  | (U64)(a))

// ------------------------------------------------------------
// #-- Large Values

#define u64_kilobytes(x)  ((U64)(x)          * 1024LL)
#define u64_megabytes(x)  (u64_kilobytes(x)  * 1024LL)
#define u64_gigabytes(x)  (u64_megabytes(x)  * 1024LL)
#define u64_terabytes(x)  (u64_gigabytes(x)  * 1024LL)

#define u64_thousands(x)  ((U64)(x)         * 1000LL)
#define u64_millions(x)   (u64_thousands(x) * 1000LL)
#define u64_billions(x)   (u64_millions(x)  * 1000LL)
#define u64_trillions(x)  (u64_billions(x)  * 1000LL)

// ------------------------------------------------------------
//  #-- Limits

global const I08 i08_limit_min = -127 - 1;
global const I08 i08_limit_max =  127;
global const I16 i16_limit_min = -32767 - 1;
global const I16 i16_limit_max =  32767;
global const I32 i32_limit_min = -2147483647 - 1;
global const I32 i32_limit_max =  2147483647;
global const I64 i64_limit_min = -9223372036854775807LL - 1LL;
global const I64 i64_limit_max =  9223372036854775807LL;

global const U08 u08_limit_min = 0;
global const U08 u08_limit_max = 255U;
global const U16 u16_limit_min = 0;
global const U16 u16_limit_max = 65535U;
global const U32 u32_limit_min = 0;
global const U32 u32_limit_max = 4294967295U;
global const U64 u64_limit_min = 0;
global const U64 u64_limit_max = 18446744073709551615ULL;

global const F32 f32_limit_min = -3.40282347e+38F;
global const F32 f32_limit_max = +3.40282347e+38F;
global const F64 f64_limit_min = -1.7976931348623157e+308;
global const F64 f64_limit_max = +1.7976931348623157e+308;

// ------------------------------------------------------------
// #-- Scalar Operations

function F32 f32_sqrt   (F32 x);
function F32 f32_ceil   (F32 x);
function F32 f32_floor  (F32 x);

function F64 f64_sqrt   (F64 x);
function F64 f64_ceil   (F64 x);
function F64 f64_floor  (F64 x);

force_inline function U08 u08_min      (U08 lhs, U08 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function U08 u08_max      (U08 lhs, U08 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function U08 u08_clamp    (U08 x, U08 a, U08 b)   { return u08_min(u08_max(x, a), b);     }

force_inline function U16 u16_min      (U16 lhs, U16 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function U16 u16_max      (U16 lhs, U16 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function U16 u16_clamp    (U16 x, U16 a, U16 b)   { return u16_min(u16_max(x, a), b);     }

force_inline function U32 u32_min      (U32 lhs, U32 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function U32 u32_max      (U32 lhs, U32 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function U32 u32_clamp    (U32 x, U32 a, U32 b)   { return u32_min(u32_max(x, a), b);     }

force_inline function U64 u64_min      (U64 lhs, U64 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function U64 u64_max      (U64 lhs, U64 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function U64 u64_clamp    (U64 x, U64 a, U64 b)   { return u64_min(u64_max(x, a), b);     }

force_inline function I08 i08_min      (I08 lhs, I08 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function I08 i08_max      (I08 lhs, I08 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function I08 i08_clamp    (I08 x, I08 a, I08 b)   { return i08_min(i08_max(x, a), b);     }
force_inline function I08 i08_sign     (I08 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function I08 i08_abs      (I08 x)                 { return x < 0 ? (I08)-x : x;           }

force_inline function I16 i16_min      (I16 lhs, I16 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function I16 i16_max      (I16 lhs, I16 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function I16 i16_clamp    (I16 x, I16 a, I16 b)   { return i16_min(i16_max(x, a), b);     }
force_inline function I16 i16_sign     (I16 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function I16 i16_abs      (I16 x)                 { return x < 0 ? (I16)-x : x;           }

force_inline function I32 i32_min      (I32 lhs, I32 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function I32 i32_max      (I32 lhs, I32 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function I32 i32_clamp    (I32 x, I32 a, I32 b)   { return i32_min(i32_max(x, a), b);     }
force_inline function I32 i32_sign     (I32 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function I32 i32_abs      (I32 x)                 { return x < 0 ? -x : x;                }

force_inline function I64 i64_min      (I64 lhs, I64 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function I64 i64_max      (I64 lhs, I64 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function I64 i64_clamp    (I64 x, I64 a, I64 b)   { return i64_min(i64_max(x, a), b);     }
force_inline function I64 i64_sign     (I64 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function I64 i64_abs      (I64 x)                 { return x < 0 ? -x : x;                }

force_inline function F32 f32_min      (F32 lhs, F32 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function F32 f32_max      (F32 lhs, F32 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function F32 f32_clamp    (F32 x, F32 a, F32 b)   { return f32_min(f32_max(x, a), b);     }
force_inline function F32 f32_sign     (F32 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function F32 f32_abs      (F32 x)                 { return x < 0 ? -x : x;                }
force_inline function F32 f32_div_safe (F32 a, F32 b)          { return b == 0 ? 0 : a / b;            }

force_inline function F64 f64_min      (F64 lhs, F64 rhs)      { return lhs < rhs ? lhs : rhs;         }
force_inline function F64 f64_max      (F64 lhs, F64 rhs)      { return lhs > rhs ? lhs : rhs;         }
force_inline function F64 f64_clamp    (F64 x, F64 a, F64 b)   { return f64_min(f64_max(x, a), b);     }
force_inline function F64 f64_sign     (F64 x)                 { return x == 0 ? 0 : (x < 0 ? -1 : 1); }
force_inline function F64 f64_abs      (F64 x)                 { return x < 0 ? -x : x;                }
force_inline function F64 f64_div_safe (F64 a, F64 b)          { return b == 0 ? 0 : a / b;            }

force_inline function F32 f32_fract    (F32 x)                 { return x - f32_floor(x); }
force_inline function F64 f64_fract    (F64 x)                 { return x - f64_floor(x); }

// ------------------------------------------------------------
// #-- Triggonometry

function F32 f32_sin(F32 x);
function F32 f32_cos(F32 x);
function F32 f32_tan(F32 x);

// ------------------------------------------------------------
// #-- Exponentials, Logarithms

function F32 f32_log(F32 x);

// ------------------------------------------------------------
// #-- Vectors

// TODO(cmat): This is definitely something we want to generate with a meta-program.

typedef union {
    struct { I32 x, y; };
    struct { I32 r, g; };
    struct { I32 u, v; };
    struct { I32 width, height; };  
    I32 dat[2];
} V2_I32;

typedef union {
    struct { U16 x, y; };
    struct { U16 r, g; };
    struct { U16 u, v; };
    struct { U16 width, height; };  
    U16 dat[2];
} V2_U16;

typedef union {
    struct { U32 x, y; };
    struct { U32 r, g; };
    struct { U32 u, v; };
    struct { U32 width, height; };  
    U32 dat[2];
} V2_U32;

typedef union {
    struct { U64 x, y; };
    struct { U64 r, g; };
    struct { U64 u, v; };
    struct { U64 width, height; };  
    U64 dat[2];
} V2_U64;

typedef union {
    struct { F32 x, y; };
    struct { F32 r, g; };
    struct { F32 u, v; };
    struct { F32 width, height; };  
    F32 dat[2];
} V2_F32;

typedef union {
    struct { F64 x, y; };
    struct { F64 r, g; };
    struct { F64 u, v; };
    struct { F64 width, height; };
    F64 dat[2];
} V2_F64;

typedef union {
    struct { I32 x, y, z; };
    struct { I32 r, g, b; };
    struct { I32 h, s, v; };
    
    struct { V2_I32 xy; I32 _pad_0; };
    struct { I32 _pad_1; V2_I32 yz; };
    
    struct { V2_I32 rg; I32 _pad_2; };
    struct { I32 _pad_3; V2_I32 gb; };
    
    struct { V2_I32 hs; I32 _pad_4; };
    struct { I32 _pad_5; V2_I32 sv; };
    
    struct { I32 width, height, depth; };
    I32 dat[3];
} V3_I32;

typedef union {
    struct { U16 x, y, z; };
    struct { U16 r, g, b; };
    struct { U16 h, s, v; };
    
    struct { V2_U16 xy; U16 _pad_0; };
    struct { U16 _pad_1; V2_U16 yz; };
    
    struct { V2_U16 rg; U16 _pad_2; };
    struct { U16 _pad_3; V2_U16 gb; };
    
    struct { V2_U16 hs; U16 _pad_4; };
    struct { U16 _pad_5; V2_U16 sv; };
    
    struct { U16 width, height, depth; };
    U16 dat[3];
} V3_U16;

typedef union {
    struct { U32 x, y, z; };
    struct { U32 r, g, b; };
    struct { U32 h, s, v; };
    
    struct { V2_U32 xy; U32 _pad_0; };
    struct { U32 _pad_1; V2_U32 yz; };
    
    struct { V2_U32 rg; U32 _pad_2; };
    struct { U32 _pad_3; V2_U32 gb; };
    
    struct { V2_U32 hs; U32 _pad_4; };
    struct { U32 _pad_5; V2_U32 sv; };
    
    struct { U32 width, height, depth; };
    U32 dat[3];
} V3_U32;

typedef union {
    struct { U64 x, y, z; };
    struct { U64 r, g, b; };
    struct { U64 h, s, v; };
    
    struct { V2_U64 xy; U64 _pad_0; };
    struct { U64 _pad_1; V2_U64 yz; };
    
    struct { V2_U64 rg; U64 _pad_2; };
    struct { U64 _pad_3; V2_U64 gb; };
    
    struct { V2_U64 hs; U64 _pad_4; };
    struct { U64 _pad_5; V2_U64 sv; };
    
    struct { U64 width, height, depth; };
    U64 dat[3];
} V3_U64;

typedef union {
    struct { F32 x, y, z; };
    struct { F32 r, g, b; };
    struct { F32 h, s, v; };
    
    struct { V2_F32 xy; F32 _pad_0; };
    struct { F32 _pad_1; V2_F32 yz; };
    
    struct { V2_F32 rg; F32 _pad_2; };
    struct { F32 _pad_3; V2_F32 gb; };
    
    struct { V2_F32 hs; F32 _pad_4; };
    struct { F32 _pad_5; V2_F32 sv; };
    
    struct { F32 width, height, depth; };
    F32 dat[3];
} V3_F32;

typedef union {
    struct { F64 x, y, z; };
    struct { F64 r, g, b; };
    struct { F64 h, s, v; };
    
    struct { V2_F64 xy; };
    struct { F64 _pad_1; V2_F64 yz; };
    
    struct { V2_F64 rg; F64 _pad_2; };
    struct { F64 _pad_3; V2_F64 gb; };
    
    struct { V2_F64 hs; F64 _pad_4; };
    struct { F64 _pad_5; V2_F64 sv; };
    
    struct { F64 width, height, depth; };
    F64 dat[3];
} V3_F64;

typedef union {
    struct { I32 x, y, z, w; };
    struct { I32 r, g, b, a; };
    
    struct { V2_I32 xy; V2_I32 _pad_0; };
    struct { V2_I32 _pad_1; V2_I32 zw; };
    
    struct { V2_I32 rg; V2_I32 _pad_2; };
    struct { V2_I32 _pad_3; V2_I32 ba; };
    
    struct { V2_I32 hs; V2_I32 _pad_4; };
    struct { V2_I32 _pad_5; V2_I32 va; };
    
    struct { I32 _pad_6;  V2_I32 yz; I32 _pad_7; };
    struct { I32 _pad_8;  V2_I32 gb; I32 _pad_9; };
    struct { I32 _pad_10; V2_I32 sv; I32 _pad_11; };
    
    struct { V3_I32 xyz; I32 _pad_12; };
    struct { V3_I32 rgb; I32 _pad_13; };
    struct { V3_I32 hsv; I32 _pad_14; };
    
    I32 dat[4];
} V4_I32;

typedef union {
    struct { U16 x, y, z, w; };
    struct { U16 r, g, b, a; };
    
    struct { V2_U16 xy; V2_U16 _pad_0; };
    struct { V2_U16 _pad_1; V2_U16 zw; };
    
    struct { V2_U16 rg; V2_U16 _pad_2; };
    struct { V2_U16 _pad_3; V2_U16 ba; };
    
    struct { V2_U16 hs; V2_U16 _pad_4; };
    struct { V2_U16 _pad_5; V2_U16 va; };
    
    struct { U16 _pad_6;  V2_U16 yz; U16 _pad_7; };
    struct { U16 _pad_8;  V2_U16 gb; U16 _pad_9; };
    struct { U16 _pad_10; V2_U16 sv; U16 _pad_11; };
    
    struct { V3_U16 xyz; U16 _pad_12; };
    struct { V3_U16 rgb; U16 _pad_13; };
    struct { V3_U16 hsv; U16 _pad_14; };
    
    U16 dat[4];
} V4_U16;

typedef union {
    struct { U32 x, y, z, w; };
    struct { U32 r, g, b, a; };
    
    struct { V2_U32 xy; V2_U32 _pad_0; };
    struct { V2_U32 _pad_1; V2_U32 zw; };
    
    struct { V2_U32 rg; V2_U32 _pad_2; };
    struct { V2_U32 _pad_3; V2_U32 ba; };
    
    struct { V2_U32 hs; V2_U32 _pad_4; };
    struct { V2_U32 _pad_5; V2_U32 va; };
    
    struct { U32 _pad_6;  V2_U32 yz; U32 _pad_7; };
    struct { U32 _pad_8;  V2_U32 gb; U32 _pad_9; };
    struct { U32 _pad_10; V2_U32 sv; U32 _pad_11; };
    
    struct { V3_U32 xyz; U32 _pad_12; };
    struct { V3_U32 rgb; U32 _pad_13; };
    struct { V3_U32 hsv; U32 _pad_14; };
    
    U32 dat[4];
} V4_U32;

typedef union {
    struct { U64 x, y, z, w; };
    struct { U64 r, g, b, a; };
    
    struct { V2_U64 xy; V2_U64 _pad_0; };
    struct { V2_U64 _pad_1; V2_U64 zw; };
    
    struct { V2_U64 rg; V2_U64 _pad_2; };
    struct { V2_U64 _pad_3; V2_U64 ba; };
    
    struct { V2_U64 hs; V2_U64 _pad_4; };
    struct { V2_U64 _pad_5; V2_U64 va; };
    
    struct { U64 _pad_6;  V2_U64 yz; U64 _pad_7; };
    struct { U64 _pad_8;  V2_U64 gb; U64 _pad_9; };
    struct { U64 _pad_10; V2_U64 sv; U64 _pad_11; };
    
    struct { V3_U64 xyz; U64 _pad_12; };
    struct { V3_U64 rgb; U64 _pad_13; };
    struct { V3_U64 hsv; U64 _pad_14; };
    
    U64 dat[4];
} V4_U64;

typedef union {
    struct { F32 x, y, z, w; };
    struct { F32 r, g, b, a; };
    
    struct { V2_F32 xy; V2_F32 _pad_0; };
    struct { V2_F32 _pad_1; V2_F32 zw; };
    
    struct { V2_F32 rg; V2_F32 _pad_2; };
    struct { V2_F32 _pad_3; V2_F32 ba; };
    
    struct { V2_F32 hs; V2_F32 _pad_4; };
    struct { V2_F32 _pad_5; V2_F32 va; };
    
    struct { F32 _pad_6;  V2_F32 yz; F32 _pad_7; };
    struct { F32 _pad_8;  V2_F32 gb; F32 _pad_9; };
    struct { F32 _pad_10; V2_F32 sv; F32 _pad_11; };
    
    struct { V3_F32 xyz; F32 _pad_12; };
    struct { V3_F32 rgb; F32 _pad_13; };
    struct { V3_F32 hsv; F32 _pad_14; };
    
    F32 dat[4];
} V4_F32;

typedef union {
    struct { F64 x, y, z, w; };
    struct { F64 r, g, b, a; };
    
    struct { V2_F64 xy; V2_F64 _pad_0; };
    struct { V2_F64 _pad_1; V2_F64 zw; };
    
    struct { V2_F64 rg; V2_F64 _pad_2; };
    struct { V2_F64 _pad_3; V2_F64 ba; };
    
    struct { V2_F64 hs; V2_F64 _pad_4; };
    struct { V2_F64 _pad_5; V2_F64 va; };
    
    struct { F64 _pad_6;  V2_F64 yz; F64 _pad_7; };
    struct { F64 _pad_8;  V2_F64 gb; F64 _pad_9; };
    struct { F64 _pad_10; V2_F64 sv; F64 _pad_11; };
    
    struct { V3_F64 xyz; F64 _pad_12; };
    struct { V3_F64 rgb; F64 _pad_13; };
    struct { V3_F64 hsv; F64 _pad_14; };
    
    F64 dat[4];
  } V4_F64;

typedef union V5_F32 {
  struct { F32 x1, x2, x3, x4, x5;                      };
  struct { V3_F32 x123; V2_F32 x45;                     };
  struct { F32 _padding_1; V3_F32 x234; F32 _padding_2; };
  struct { V2_F32 x12;  V3_F32 x345;                    };
} V5_F32;

Assert_Compiler(sizeof(V5_F32) == 5 * sizeof(F32));

typedef V2_F32 V2F;
typedef V3_F32 V3F;
typedef V4_F32 V4F;
typedef V5_F32 V5F;

typedef V2_I32 V2I;
typedef V3_I32 V3I;
typedef V4_I32 V4I;

typedef V2_U32 V2U;
typedef V3_U32 V3U;
typedef V4_U32 V4U;

Assert_Compiler(sizeof(V2_I32) == 2 * sizeof(I32));
Assert_Compiler(sizeof(V2_U32) == 2 * sizeof(U32));
Assert_Compiler(sizeof(V2_F32) == 2 * sizeof(F32));
Assert_Compiler(sizeof(V2_F64) == 2 * sizeof(F64));

Assert_Compiler(sizeof(V3_I32) == 3 * sizeof(I32));
Assert_Compiler(sizeof(V3_U32) == 3 * sizeof(U32));
Assert_Compiler(sizeof(V3_F32) == 3 * sizeof(F32));
Assert_Compiler(sizeof(V3_F64) == 3 * sizeof(F64));

Assert_Compiler(sizeof(V4_I32) == 4 * sizeof(I32));
Assert_Compiler(sizeof(V4_U32) == 4 * sizeof(U32));
Assert_Compiler(sizeof(V4_F32) == 4 * sizeof(F32));
Assert_Compiler(sizeof(V4_F64) == 4 * sizeof(F64));

// NOTE(cmat): Cast Macros
#define V2_Expand(v) ((v).x), ((v).y)
#define V3_Expand(v) ((v).x), ((v).y), ((v).z)
#define V4_Expand(v) ((v).x), ((v).y), ((v).z), ((v).w)
#define V5_Expand(v) ((v).x1), ((v).x2), ((v).x3), ((v).x4), ((v).x5)

// NOTE(cmat): Cast Macros.
#define V2_Cast(v, type_) (V2##_##type_) { (type_)((v).x), (type_)((v).y) }
#define V3_Cast(v, type_) (V3##_##type_) { (type_)((v).x), (type_)((v).y), (type_)((v).z) }
#define V4_Cast(v, type_) (V3##_##type_) { (type_)((v).x), (type_)((v).y), (type_)((v).z), (type_)((v).w) }

force_inline function V2F     v2f       (F32 x, F32 y)                              { return (V2F) { x, y };       }
force_inline function V3F     v3f       (F32 x, F32 y, F32 z)                       { return (V3F) { x, y, z };    }
force_inline function V4F     v4f       (F32 x, F32 y, F32 z, F32 w)                { return (V4F) { x, y, z, w }; }
force_inline function V5F     v5f       (F32 x1, F32 x2, F32 x3, F32 x4, F32 x5)    { return (V5F) { x1, x2, x3, x4, x5 }; }

force_inline function V2I     v2i       (I32 x, I32 y)                  { return (V2I) { x, y };       }
force_inline function V3I     v3i       (I32 x, I32 y, I32 z)           { return (V3I) { x, y, z };    }
force_inline function V4I     v4i       (I32 x, I32 y, I32 z, I32 w)    { return (V4I) { x, y, z, w }; }

force_inline function V2U     v2u       (U32 x, U32 y)                  { return (V2U) { x, y };       }
force_inline function V3U     v3u       (U32 x, U32 y, U32 z)           { return (V3U) { x, y, z };    }
force_inline function V4U     v4u       (U32 x, U32 y, U32 z, U32 w)    { return (V4U) { x, y, z, w }; }

force_inline function V2_U16  v2_u16    (U16 x, U16 y)                  { return (V2_U16) { x, y };       }
force_inline function V3_U16  v3_u16    (U16 x, U16 y, U16 z)           { return (V3_U16) { x, y, z };    }
force_inline function V4_U16  v4_u16    (U16 x, U16 y, U16 z, U16 w)    { return (V4_U16) { x, y, z, w }; }

force_inline function V2_U64  v2_u64    (U64 x, U64 y)                  { return (V2_U64) { x, y };       }
force_inline function V3_U64  v3_u64    (U64 x, U64 y, U64 z)           { return (V3_U64) { x, y, z };    }
force_inline function V4_U64  v4_u64    (U64 x, U64 y, U64 z, U64 w)    { return (V4_U64) { x, y, z, w }; }

force_inline function V2F     v2f_f32   (F32 x) { return (V2F) { x, x };       }
force_inline function V3F     v3f_f32   (F32 x) { return (V3F) { x, x, x };    }
force_inline function V4F     v4f_f32   (F32 x) { return (V4F) { x, x, x, x }; }
force_inline function V2I     v2i_s     (I32 x) { return (V2I) { x, x };       }
force_inline function V3I     v3i_s     (I32 x) { return (V3I) { x, x, x };    }
force_inline function V4I     v4i_s     (I32 x) { return (V4I) { x, x, x, x }; }

// NOTE(cmat): 2D
inline function V2F v2f_add           (V2F lhs, V2F rhs)         { return (V2F) { lhs.x + rhs.x, lhs.y + rhs.y }; }
inline function V2F v2f_sub           (V2F lhs, V2F rhs)         { return (V2F) { lhs.x - rhs.x, lhs.y - rhs.y }; }
inline function V2F v2f_had           (V2F lhs, V2F rhs)         { return (V2F) { lhs.x * rhs.x, lhs.y * rhs.y }; }
inline function V2F v2f_mul           (F32 lhs, V2F rhs)         { return (V2F) { lhs * rhs.x, lhs * rhs.y };     }
inline function V2F v2f_div           (V2F lhs, F32 rhs)         { return (V2F) { lhs.x / rhs, lhs.y / rhs };     }
inline function V2F v2f_add_f32       (V2F lhs, F32 rhs)         { return (V2F) { lhs.x + rhs, lhs.y + rhs };     }
inline function V2F v2f_sub_f32       (V2F lhs, F32 rhs)         { return (V2F) { lhs.x - rhs, lhs.y - rhs };     }
inline function V2F v2f_mul_f32       (V2F lhs, F32 rhs)         { return (V2F) { lhs.x * rhs, lhs.y * rhs };     }

inline function V2I v2i_add            (V2I lhs, V2I rhs)         { return (V2I) { lhs.x + rhs.x, lhs.y + rhs.y }; }
inline function V2I v2i_sub            (V2I lhs, V2I rhs)         { return (V2I) { lhs.x - rhs.x, lhs.y - rhs.y }; }
inline function V2I v2i_had            (V2I lhs, V2I rhs)         { return (V2I) { lhs.x * rhs.x, lhs.y * rhs.y }; }
inline function V2I v2i_mul            (I32 lhs, V2I rhs)         { return (V2I) { lhs * rhs.x, lhs * rhs.y };     }
inline function V2I v2i_div            (V2I lhs, I32 rhs)         { return (V2I) { lhs.x / rhs, lhs.y / rhs };     }
inline function V2I v2i_add_i32        (V2I lhs, I32 rhs)         { return (V2I) { lhs.x + rhs, lhs.y + rhs };     }
inline function V2I v2i_sub_i32        (V2I lhs, I32 rhs)         { return (V2I) { lhs.x - rhs, lhs.y - rhs };     }
inline function V2I v2i_mul_i32        (V2I lhs, I32 rhs)         { return (V2I) { lhs.x * rhs, lhs.y * rhs };     }

inline function V2U v2u_add            (V2U lhs, V2U rhs)         { return (V2U) { lhs.x + rhs.x, lhs.y + rhs.y }; }
inline function V2U v2u_sub            (V2U lhs, V2U rhs)         { return (V2U) { lhs.x - rhs.x, lhs.y - rhs.y }; }
inline function V2U v2u_had            (V2U lhs, V2U rhs)         { return (V2U) { lhs.x * rhs.x, lhs.y * rhs.y }; }
inline function V2U v2u_mul            (U32 lhs, V2U rhs)         { return (V2U) { lhs * rhs.x, lhs * rhs.y };     }
inline function V2U v2u_div            (V2U lhs, U32 rhs)         { return (V2U) { lhs.x / rhs, lhs.y / rhs };     }
inline function V2U v2u_add_u32        (V2U lhs, U32 rhs)         { return (V2U) { lhs.x + rhs, lhs.y + rhs };     }
inline function V2U v2u_sub_u32        (V2U lhs, U32 rhs)         { return (V2U) { lhs.x - rhs, lhs.y - rhs };     }
inline function V2U v2u_mul_u32        (V2U lhs, U32 rhs)         { return (V2U) { lhs.x * rhs, lhs.y * rhs };     }

inline function V2_U16 v2_u16_add      (V2_U16 lhs, V2_U16 rhs)   { return (V2_U16) { lhs.x + rhs.x, lhs.y + rhs.y }; }
inline function V2_U16 v2_u16_sub      (V2_U16 lhs, V2_U16 rhs)   { return (V2_U16) { lhs.x - rhs.x, lhs.y - rhs.y }; }
inline function V2_U16 v2_u16_had      (V2_U16 lhs, V2_U16 rhs)   { return (V2_U16) { lhs.x * rhs.x, lhs.y * rhs.y }; }
inline function V2_U16 v2_u16_mul      (U16    lhs, V2_U16 rhs)   { return (V2_U16) { lhs * rhs.x, lhs * rhs.y };     }
inline function V2_U16 v2_u16_div      (V2_U16 lhs, U16    rhs)   { return (V2_U16) { lhs.x / rhs, lhs.y / rhs };     }
inline function V2_U16 v2_u16_add_u16  (V2_U16 lhs, U16    rhs)   { return (V2_U16) { lhs.x + rhs, lhs.y + rhs };     }
inline function V2_U16 v2_u16_sub_u16  (V2_U16 lhs, U16    rhs)   { return (V2_U16) { lhs.x - rhs, lhs.y - rhs };     }
inline function V2_U16 v2_u16_mul_u16  (V2_U16 lhs, U16    rhs)   { return (V2_U16) { lhs.x * rhs, lhs.y * rhs };     }

// NOTE(cmat): 3D
inline function V3F v3f_add         (V3F lhs, V3F rhs)      { return (V3F) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };  }
inline function V3F v3f_sub         (V3F lhs, V3F rhs)      { return (V3F) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };  }
inline function V3F v3f_mul         (F32 lhs, V3F rhs)      { return (V3F) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };        }
inline function V3F v3f_had         (V3F lhs, V3F rhs)      { return (V3F) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };  }
inline function V3F v3f_div         (V3F lhs, F32 rhs)      { return (V3F) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };        }
inline function V3F v3f_add_f32     (V3F lhs, F32 rhs)      { return (V3F) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs };        }
inline function V3F v3f_sub_f32     (V3F lhs, F32 rhs)      { return (V3F) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs };        }
inline function V3F v3f_mul_f32     (V3F lhs, F32 rhs)      { return (V3F) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };        }

inline function V3I v3i_add         (V3I lhs, V3I rhs)      { return (V3I) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};   }
inline function V3I v3i_sub         (V3I lhs, V3I rhs)      { return (V3I) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};   }
inline function V3I v3i_had         (V3I lhs, V3I rhs)      { return (V3I) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };  }
inline function V3I v3i_mul         (I32 lhs, V3I rhs)      { return (V3I) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };        }
inline function V3I v3i_div         (V3I lhs, I32 rhs)      { return (V3I) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };        }
inline function V3I v3i_add_i32     (V3I lhs, I32 rhs)      { return (V3I) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs };        }
inline function V3I v3i_sub_i32     (V3I lhs, I32 rhs)      { return (V3I) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs };        }
inline function V3I v3i_mul_i32     (V3I lhs, I32 rhs)      { return (V3I) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };        }

inline function V3U v3u_add         (V3U lhs, V3U rhs)      { return (V3U) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};   }
inline function V3U v3u_sub         (V3U lhs, V3U rhs)      { return (V3U) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};   }
inline function V3U v3u_had         (V3U lhs, V3U rhs)      { return (V3U) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };  }
inline function V3U v3u_mul         (U32 lhs, V3U rhs)      { return (V3U) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };        }
inline function V3U v3u_div         (V3U lhs, U32 rhs)      { return (V3U) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };        }
inline function V3U v3u_add_u32     (V3U lhs, U32 rhs)      { return (V3U) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs };        }
inline function V3U v3u_sub_u32     (V3U lhs, U32 rhs)      { return (V3U) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs };        }
inline function V3U v3u_mul_u32     (V3U lhs, U32 rhs)      { return (V3U) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };        }

inline function V3_U16 v3_u16_add         (V3_U16 lhs, V3_U16 rhs)      { return (V3_U16) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};   }
inline function V3_U16 v3_u16_sub         (V3_U16 lhs, V3_U16 rhs)      { return (V3_U16) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};   }
inline function V3_U16 v3_u16_had         (V3_U16 lhs, V3_U16 rhs)      { return (V3_U16) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };  }
inline function V3_U16 v3_u16_mul         (U16    lhs, V3_U16 rhs)      { return (V3_U16) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };        }
inline function V3_U16 v3_u16_div         (V3_U16 lhs, U16 rhs)         { return (V3_U16) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };        }
inline function V3_U16 v3_u16_add_u16     (V3_U16 lhs, U16 rhs)         { return (V3_U16) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs };        }
inline function V3_U16 v3_u16_sub_u16     (V3_U16 lhs, U16 rhs)         { return (V3_U16) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs };        }
inline function V3_U16 v3_u16_mul_u16     (V3_U16 lhs, U16 rhs)         { return (V3_U16) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };        }

inline function V3_F64 v3_f64_add         (V3_F64 lhs, V3_F64 rhs)      { return (V3_F64) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};   }
inline function V3_F64 v3_f64_sub         (V3_F64 lhs, V3_F64 rhs)      { return (V3_F64) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};   }
inline function V3_F64 v3_f64_had         (V3_F64 lhs, V3_F64 rhs)      { return (V3_F64) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };  }
inline function V3_F64 v3_f64_mul         (F64    lhs, V3_F64 rhs)      { return (V3_F64) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };        }
inline function V3_F64 v3_f64_div         (V3_F64 lhs, F64 rhs)         { return (V3_F64) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };        }
inline function V3_F64 v3_f64_add_f64     (V3_F64 lhs, F64 rhs)         { return (V3_F64) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs };        }
inline function V3_F64 v3_f64_sub_f64     (V3_F64 lhs, F64 rhs)         { return (V3_F64) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs };        }
inline function V3_F64 v3_f64_mul_f64     (V3_F64 lhs, F64 rhs)         { return (V3_F64) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };        }


// NOTE(cmat): 4D
inline function V4F v4f_add         (V4F lhs, V4F rhs)      { return (V4F) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline function V4F v4f_sub         (V4F lhs, V4F rhs)      { return (V4F) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};  }
inline function V4F v4f_had         (V4F lhs, V4F rhs)      { return (V4F) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};  }
inline function V4F v4f_mul         (F32 lhs, V4F rhs)      { return (V4F) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w };         }
inline function V4F v4f_div         (V4F lhs, F32 rhs)      { return (V4F) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs };         }
inline function V4F v4f_add_f32     (V4F lhs, F32 rhs)      { return (V4F) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs };         }
inline function V4F v4f_sub_f32     (V4F lhs, F32 rhs)      { return (V4F) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs };         }
inline function V4F v4f_mul_f32     (V4F lhs, F32 rhs)      { return (V4F) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs };         }

inline function V4I v4i_add         (V4I lhs, V4I rhs)      { return (V4I) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline function V4I v4i_sub         (V4I lhs, V4I rhs)      { return (V4I) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};  }
inline function V4I v4i_had         (V4I lhs, V4I rhs)      { return (V4I) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};  }
inline function V4I v4i_mul         (I32 lhs, V4I rhs)      { return (V4I) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w };         }
inline function V4I v4i_div         (V4I lhs, I32 rhs)      { return (V4I) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs };         }
inline function V4I v4i_add_i32     (V4I lhs, I32 rhs)      { return (V4I) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs };         }
inline function V4I v4i_sub_i32     (V4I lhs, I32 rhs)      { return (V4I) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs };         }
inline function V4I v4i_mul_i32     (V4I lhs, I32 rhs)      { return (V4I) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs };         }

inline function V4U v4u_add         (V4U lhs, V4U rhs)      { return (V4U) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline function V4U v4u_sub         (V4U lhs, V4U rhs)      { return (V4U) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};  }
inline function V4U v4u_had         (V4U lhs, V4U rhs)      { return (V4U) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};  }
inline function V4U v4u_mul         (U32 lhs, V4U rhs)      { return (V4U) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w };         }
inline function V4U v4u_div         (V4U lhs, U32 rhs)      { return (V4U) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs };         }
inline function V4U v4u_add_u32     (V4U lhs, U32 rhs)      { return (V4U) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs };         }
inline function V4U v4u_sub_u32     (V4U lhs, U32 rhs)      { return (V4U) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs };         }
inline function V4U v4u_mul_u32     (V4U lhs, U32 rhs)      { return (V4U) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs };         }

inline function V4_U16 v4_u16_add         (V4_U16 lhs, V4_U16 rhs)   { return (V4_U16) { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
inline function V4_U16 v4_u16_sub         (V4_U16 lhs, V4_U16 rhs)   { return (V4_U16) { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};  }
inline function V4_U16 v4_u16_had         (V4_U16 lhs, V4_U16 rhs)   { return (V4_U16) { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};  }
inline function V4_U16 v4_u16_mul         (U16 lhs, V4_U16 rhs)      { return (V4_U16) { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w };         }
inline function V4_U16 v4_u16_div         (V4_U16 lhs, U16 rhs)      { return (V4_U16) { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs };         }
inline function V4_U16 v4_u16_add_u32     (V4_U16 lhs, U16 rhs)      { return (V4_U16) { lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs };         }
inline function V4_U16 v4_u16_sub_u32     (V4_U16 lhs, U16 rhs)      { return (V4_U16) { lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs };         }
inline function V4_U16 v4_u16_mul_u32     (V4_U16 lhs, U16 rhs)      { return (V4_U16) { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs };         }

// NOTE(cmat): 5D
inline function V5F v5f_add         (V5F lhs, V5F rhs)      { return (V5F) { lhs.x1 + rhs.x1, lhs.x2 + rhs.x2, lhs.x3 + rhs.x3, lhs.x4 + rhs.x4, lhs.x5 + rhs.x5 };   }
inline function V5F v5f_sub         (V5F lhs, V5F rhs)      { return (V5F) { lhs.x1 - rhs.x1, lhs.x2 - rhs.x2, lhs.x3 - rhs.x3, lhs.x4 - rhs.x4, lhs.x5 - rhs.x5 };   }
inline function V5F v5f_had         (V5F lhs, V5F rhs)      { return (V5F) { lhs.x1 * rhs.x1, lhs.x2 * rhs.x2, lhs.x3 * rhs.x3, lhs.x4 * rhs.x4, lhs.x5 * rhs.x5 };   }
inline function V5F v5f_mul         (F32 lhs, V5F rhs)      { return (V5F) { lhs * rhs.x1, lhs * rhs.x2, lhs * rhs.x3, lhs * rhs.x4, lhs * rhs.x5 };                  }
inline function V5F v5f_div         (V5F lhs, F32 rhs)      { return (V5F) { lhs.x1 / rhs, lhs.x2 / rhs, lhs.x3 / rhs, lhs.x4 / rhs, lhs.x5 / rhs };                  }
inline function V5F v5f_add_f32     (V5F lhs, F32 rhs)      { return (V5F) { lhs.x1 + rhs, lhs.x2 + rhs, lhs.x3 + rhs, lhs.x4 + rhs, lhs.x5 + rhs };                  }
inline function V5F v5f_sub_f32     (V5F lhs, F32 rhs)      { return (V5F) { lhs.x1 - rhs, lhs.x2 - rhs, lhs.x3 - rhs, lhs.x4 - rhs, lhs.x5 - rhs };                  }
inline function V5F v5f_mul_f32     (V5F lhs, F32 rhs)      { return (V5F) { lhs.x1 * rhs, lhs.x2 * rhs, lhs.x3 * rhs, lhs.x4 * rhs, lhs.x5 * rhs};                   }


inline function F32 v2f_dot          (V2F lhs, V2F rhs)     { return lhs.x * rhs.x + lhs.y * rhs.y;                                 }
inline function F32 v3f_dot          (V3F lhs, V3F rhs)     { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;                 }
inline function F32 v4f_dot          (V4F lhs, V4F rhs)     { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }

inline function F32 v2f_len2         (V2F x)                { return v2f_dot(x, x); }
inline function F32 v3f_len2         (V3F x)                { return v3f_dot(x, x); }
inline function F32 v4f_len2         (V4F x)                { return v4f_dot(x, x); }

inline function F32 v2f_len          (V2F x)                { return f32_sqrt(v2f_len2(x)); }
inline function F32 v3f_len          (V3F x)                { return f32_sqrt(v3f_len2(x)); }
inline function F32 v4f_len          (V4F x)                { return f32_sqrt(v4f_len2(x)); }

inline function F32 v2f_dist         (V2F lhs, V2F rhs)     { return v2f_len(v2f_sub(lhs, rhs)); }
inline function F32 v3f_dist         (V3F lhs, V3F rhs)     { return v3f_len(v3f_sub(lhs, rhs)); }
inline function F32 v4f_dist         (V4F lhs, V4F rhs)     { return v4f_len(v4f_sub(lhs, rhs)); }

inline function V2F v2f_nor          (V2F x)                { return v2f_div(x, v2f_len(x)); }
inline function V3F v3f_nor          (V3F x)                { return v3f_div(x, v3f_len(x)); }
inline function V4F v4f_nor          (V4F x)                { return v4f_div(x, v4f_len(x)); }

inline function V2F v2f_noz          (V2F x)                { F32 len = v2f_len(x); return len <= f32_noz_epsilon ? v2f_f32(0.f) : v2f_div(x, len); }
inline function V3F v3f_noz          (V3F x)                { F32 len = v3f_len(x); return len <= f32_noz_epsilon ? v3f_f32(0.f) : v3f_div(x, len); }
inline function V4F v4f_noz          (V4F x)                { F32 len = v4f_len(x); return len <= f32_noz_epsilon ? v4f_f32(0.f) : v4f_div(x, len); }

inline function V2F v2f_abs          (V2F x)                { return v2f(f32_abs(x.x), f32_abs(x.y));                             }
inline function V3F v3f_abs          (V3F x)                { return v3f(f32_abs(x.x), f32_abs(x.y), f32_abs(x.z));               }
inline function V4F v4f_abs          (V4F x)                { return v4f(f32_abs(x.x), f32_abs(x.y), f32_abs(x.z), f32_abs(x.w)); }

inline function V2F v2f_saturate     (V2F x)                { return v2f(f32_clamp(x.x, 0.f, 1.f), f32_clamp(x.y, 0.f, 1.f));                                                     }
inline function V3F v3f_saturate     (V3F x)                { return v3f(f32_clamp(x.x, 0.f, 1.f), f32_clamp(x.y, 0.f, 1.f), f32_clamp(x.z, 0.f, 1.f));                           }
inline function V4F v4f_saturate     (V4F x)                { return v4f(f32_clamp(x.x, 0.f, 1.f), f32_clamp(x.y, 0.f, 1.f), f32_clamp(x.z, 0.f, 1.f), f32_clamp(x.w, 0.f, 1.f)); }

inline function F32 v2f_largest      (V2F x)                { return f32_max(x.x, x.y);                              }
inline function F32 v3f_largest      (V3F x)                { return f32_max(x.x, f32_max(x.y, x.z));                }
inline function F32 v4f_largest      (V4F x)                { return f32_max(x.x, f32_max(x.y, f32_max(x.z, x.w)));  }

inline function V2F v2f_frac         (V2F x)                { return v2f(f32_fract(x.x), f32_fract(x.y));                                 }
inline function V3F v3f_frac         (V3F x)                { return v3f(f32_fract(x.x), f32_fract(x.y), f32_fract(x.z));                 }
inline function V4F v4f_frac         (V4F x)                { return v4f(f32_fract(x.x), f32_fract(x.y), f32_fract(x.z), f32_fract(x.w)); }

inline function B32 v2f_all          (V2F x)                { return x.x != 0.f && x.y != 0.f;                             }
inline function B32 v3f_all          (V3F x)                { return x.x != 0.f && x.y != 0.f && x.z != 0.f;               }
inline function B32 v4f_all          (V4F x)                { return x.x != 0.f && x.y != 0.f && x.z != 0.f && x.w != 0.f; }

inline function B32 v2f_any          (V2F x)                { return x.x != 0.f || x.y != 0.f;                             }
inline function B32 v3f_any          (V3F x)                { return x.x != 0.f || x.y != 0.f || x.z != 0.f;               }
inline function B32 v4f_any          (V4F x)                { return x.x != 0.f || x.y != 0.f || x.z != 0.f || x.w != 0.f; }

inline function V2F v2f_reflect      (V2F in, V2F normal)   { return v2f_sub(in, v2f_mul( v2f_dot(in, normal), v2f_mul(2.f, normal))); }
inline function V3F v3f_reflect      (V3F in, V3F normal)   { return v3f_sub(in, v3f_mul( v3f_dot(in, normal), v3f_mul(2.f, normal))); }
inline function V4F v4f_reflect      (V4F in, V4F normal)   { return v4f_sub(in, v4f_mul( v4f_dot(in, normal), v4f_mul(2.f, normal))); }

inline function V2F v2f_rcp          (V2F x)                { return v2f(1.f / x.x, 1.f / x.y);                       }
inline function V3F v3f_rcp          (V3F x)                { return v3f(1.f / x.x, 1.f / x.y, 1.f / x.z);            }
inline function V4F v4f_rcp          (V4F x)                { return v4f(1.f / x.x, 1.f / x.y, 1.f / x.z, 1.f / x.w); }

// NOTE(cmat): Cross only works in 3 and 5 dimensions.
inline function V3F v3f_cross(V3F lhs, V3F rhs) {
    return (V3F) {
        .x = lhs.y * rhs.z - lhs.z * rhs.y,
        .y = lhs.z * rhs.x - lhs.x * rhs.z,
        .z = lhs.x * rhs.y - lhs.y * rhs.x
    };
}

// NOTE(cmat): Sorting
inline function void v3_u32_sort(V3_U32 *v) {
  if (v->x > v->y) { Macro_Swap(U32, v->x, v->y); }
  if (v->y > v->z) { Macro_Swap(U32, v->y, v->z); }
  if (v->x > v->y) { Macro_Swap(U32, v->x, v->y); }
}

inline function void v4_u32_sort(V4_U32 *v) {
  if (v->x > v->y) { Macro_Swap(U32, v->x, v->y); }
  if (v->y > v->z) { Macro_Swap(U32, v->y, v->z); }
  if (v->z > v->w) { Macro_Swap(U32, v->z, v->w); }

  if (v->x > v->y) { Macro_Swap(U32, v->x, v->y); }
  if (v->y > v->z) { Macro_Swap(U32, v->y, v->z); }

  if (v->x > v->y) { Macro_Swap(U32, v->x, v->y); }
}

inline function void v4_u64_sort(V4_U64 *v) {
  if (v->x > v->y) { Macro_Swap(U64, v->x, v->y); }
  if (v->y > v->z) { Macro_Swap(U64, v->y, v->z); }
  if (v->z > v->w) { Macro_Swap(U64, v->z, v->w); }

  if (v->x > v->y) { Macro_Swap(U64, v->x, v->y); }
  if (v->y > v->z) { Macro_Swap(U64, v->y, v->z); }

  if (v->x > v->y) { Macro_Swap(U64, v->x, v->y); }
}

// ------------------------------------------------------------
// #-- Matrices

// TODO(cmat): Change multiplication order.
// TODO(cmat): Automate this with code-generation as well.

typedef union {
    struct { V2_I32 row_1, row_2; };
    struct { I32  e11, e12,
        e21, e22; };
    I32 ele[2][2];
    I32 dat[2 * 2];
} M2_I32;

typedef union {
    struct { V2_F32 row_1, row_2; };
    struct { F32  e11, e12,
        e21, e22; };
    F32 ele[2][2];
    F32 dat[2 * 2];
} M2_F32;

typedef union {
    struct { V2_F64 row_1, row_2; };
    struct { F64  e11, e12,
        e21, e22; };
    F64 ele[2][2];
    F64 dat[2 * 2];
} M2_F64;

typedef union {
    struct { V3_I32 row_1, row_2, row_3; };
    struct { I32  e11, e12, e13,
        e21, e22, e23,
        e31, e32, e33; };
    I32 ele[3][3];
    U32 dat[3 * 3];
} M3_I32;

typedef union {
    struct { V3_F32 row_1, row_2, row_3; };
    struct { F32  e11, e12, e13,
        e21, e22, e23,
        e31, e32, e33; };
    F32 ele[3][3];
    F32 dat[3 * 3];
} M3_F32;

typedef union {
    struct { V3_F64 row_1, row_2, row_3; };
    struct { F64  e11, e12, e13,
        e21, e22, e23,
        e31, e32, e33; };
    F64 ele[3][3];
    F64 dat[3 * 3];
} M3_F64;

typedef union {
    struct { V4_I32 row_1, row_2, row_3, row_4; };
    struct { I32  e11, e12, e13, e14,
        e21, e22, e23, e24,
        e31, e32, e33, e34,
        e41, e42, e43, e44; };
    I32 ele[4][4];
    I32 dat[4 * 4];
} M4_I32;

typedef union {
    struct { V4_F32 row_1, row_2, row_3, row_4; };
    
    struct { F32  e11, e12, e13, e14,
        e21, e22, e23, e24,
        e31, e32, e33, e34,
        e41, e42, e43, e44; };
    F32 ele[4][4];
    F32 dat[4 * 4];
} M4_F32;

typedef union {
    struct { V4_F64 row_1, row_2, row_3, row_4; };
    
    struct { F64  e11, e12, e13, e14,
        e21, e22, e23, e24,
        e31, e32, e33, e34,
        e41, e42, e43, e44; };
    F64 ele[4][4];
    F64 dat[4 * 4];
} M4_F64;

typedef M2_I32 M2I;
typedef M3_I32 M3I;
typedef M4_I32 M4I;

typedef M2_F32 M2F;
typedef M3_F32 M3F;
typedef M4_F32 M4F;

Assert_Compiler(sizeof(M2_I32) == 2 * 2 * sizeof(I32));
Assert_Compiler(sizeof(M2_F32) == 2 * 2 * sizeof(F32));
Assert_Compiler(sizeof(M2_F64) == 2 * 2 * sizeof(F64));

Assert_Compiler(sizeof(M3_I32) == 3 * 3 * sizeof(I32));
Assert_Compiler(sizeof(M3_F32) == 3 * 3 * sizeof(F32));
Assert_Compiler(sizeof(M3_F64) == 3 * 3 * sizeof(F64));

Assert_Compiler(sizeof(M4_I32) == 4 * 4 * sizeof(I32));
Assert_Compiler(sizeof(M4_F32) == 4 * 4 * sizeof(F32));
Assert_Compiler(sizeof(M4_F64) == 4 * 4 * sizeof(F64));

inline function M2F m2f_id(void) {
    M2F result;
    Zero_Fill(&result);
    result.e11 = 1;
    result.e22 = 1;
    return result;
}

inline function M3F m3f_id(void) {
    M3F result;
    Zero_Fill(&result);
    result.e11 = 1;
    result.e22 = 1;
    result.e33 = 1;
    return result;
}

inline function M4F m4f_id(void) {
    M4F result;
    Zero_Fill(&result);
    result.e11 = 1;
    result.e22 = 1;
    result.e33 = 1;
    result.e44 = 1;
    return result;
}

inline function M4F m4f_diag(V4F diag) {
    M4F result;
    Zero_Fill(&result);
    result.e11 = diag.dat[0];
    result.e22 = diag.dat[1];
    result.e33 = diag.dat[2];
    result.e44 = diag.dat[3];
    return result;
}


M2F m2f_mul(M2F lhs, M2F rhs) {
    M2F result;
    Zero_Fill(&result);
    
    for Iter_Index(row, 2) {
        for Iter_Index(col, 2) {
            result.ele[row][col] = rhs.ele[row][0] * lhs.ele[0][col] +
                rhs.ele[row][1] * lhs.ele[1][col];
        }
    }
    
    return result;
}

M3F m3f_mul(M3F lhs, M3F rhs) {
    M3F result;
    Zero_Fill(&result);
    
    for Iter_Index(row, 3) {
        for Iter_Index(col, 3) {
            result.ele[row][col] = rhs.ele[row][0] * lhs.ele[0][col] +
                rhs.ele[row][1] * lhs.ele[1][col] +
                rhs.ele[row][2] * lhs.ele[2][col];
        }
    }
    
    return result;
}

M4F m4f_mul(M4F lhs, M4F rhs) {
    M4F result;
    Zero_Fill(&result);
    
    for Iter_Index(row, 4) {
        for Iter_Index(col, 4) {
            result.ele[row][col] = rhs.ele[row][0] * lhs.ele[0][col] +
                rhs.ele[row][1] * lhs.ele[1][col] +
                rhs.ele[row][2] * lhs.ele[2][col] +
                rhs.ele[row][3] * lhs.ele[3][col];
        }
    }
    
    return result;
}

V4F m4f_mul_v4f(V4F lhs, M4F rhs) {
    V4F result;
    Zero_Fill(&result);
    
    for Iter_Index(row, 4) {
        result.dat[row] = rhs.ele[row][0] * lhs.dat[0] +
            rhs.ele[row][1] * lhs.dat[1] +
            rhs.ele[row][2] * lhs.dat[2] +
            rhs.ele[row][3] * lhs.dat[3];
    }
    
    return result;
}

inline function M2F m2f_trans(M2F x) {
    M2F result = {};
    Zero_Fill(&result);
    
    for Iter_Index(col, 2) {
        for Iter_Index(row, 2) {
            result.ele[col][row] = x.ele[row][col];
        }
    }
    
    return result;
}

inline function M3F m3f_trans(M3F x) {
    M3F result = {};
    Zero_Fill(&result);
    
    for Iter_Index(col, 3) {
        for Iter_Index(row, 3) {
            result.ele[col][row] = x.ele[row][col];
        }
    }
    
    return result;
}

inline function M4F m4f_trans(M4F x) {
    M4F result = {};
    Zero_Fill(&result);
    
    for Iter_Index(col, 4) {
        for Iter_Index(row, 4) {
            result.ele[col][row] = x.ele[row][col];
        }
    }
    
    return result;
}

inline function M2F m2f_add(M2F lhs, M2F rhs) {
    M2F result = { };
    for Iter_Index(it, 2*2) {
        result.dat[it] = lhs.dat[it] + rhs.dat[it];
    }
    
    return result;
}

inline function M3F m3f_add(M3F lhs, M3F rhs) {
    M3F result = { };
    for Iter_Index(it, 3*3) {
        result.dat[it] = lhs.dat[it] + rhs.dat[it];
    }
    
    return result;
}

inline function M4F m4f_add(M4F lhs, M4F rhs) {
    M4F result = { };
    for Iter_Index(it, 4*4) {
        result.dat[it] = lhs.dat[it] + rhs.dat[it];
    }
    
    return result;
}

inline function M2F m2f_sub(M2F lhs, M2F rhs) {
    M2F result = { };
    for Iter_Index(it, 2*2) {
        result.dat[it] = lhs.dat[it] - rhs.dat[it];
    }
    
    return result;
}

inline function M3F m3f_sub(M3F lhs, M3F rhs) {
    M3F result = { };
    for Iter_Index(it, 3*3) {
        result.dat[it] = lhs.dat[it] - rhs.dat[it];
    }
    
    return result;
}

inline function M4F m4f_sub(M4F lhs, M4F rhs) {
    M4F result = { };
    for Iter_Index(it, 4*4) {
        result.dat[it] = lhs.dat[it] - rhs.dat[it];
    }
    
    return result;
}

inline function M2F m2f_had(M2F lhs, M2F rhs) {
    M2F result = { };
    for Iter_Index(it, 2*2) {
        result.dat[it] = lhs.dat[it] * rhs.dat[it];
    }
    
    return result;
}

inline function M3F m3f_had(M3F lhs, M3F rhs) {
    M3F result = { };
    for Iter_Index(it, 3*3) {
        result.dat[it] = lhs.dat[it] * rhs.dat[it];
    }
    
    return result;
}

inline function M4F m4f_had(M4F lhs, M4F rhs) {
    M4F result = { };
    for Iter_Index(it, 4*4) {
        result.dat[it] = lhs.dat[it] * rhs.dat[it];
    }
    
    return result;
}

inline function M2F m2f_mul_f32(F32 lhs, M2F rhs) {
    M2F result = { };
    for Iter_Index(it, 2*2) {
        result.dat[it] = lhs * rhs.dat[it];
    }
    
    return result;
}

inline function M3F m3f_mul_f32(F32 lhs, M3F rhs) {
    M3F result = { };
    for Iter_Index(it, 3*3) {
        result.dat[it] = lhs * rhs.dat[it];
    }
    
    return result;
}

inline function M4F m4f_mul_f32(F32 lhs, M4F rhs) {
    M4F result = { };
    for Iter_Index(it, 4*4) {
        result.dat[it] = lhs * rhs.dat[it];
    }
    
    return result;
}

inline function M2F m2f_div_f32(M2F lhs, F32 rhs) {
    M2F result = { };
    for Iter_Index(it, 2*2) {
        result.dat[it] = lhs.dat[it] / rhs;
    }
    
    return result;
}

inline function M3F m3f_div_f32(M3F lhs, F32 rhs) {
    M3F result = { };
    for Iter_Index(it, 3*3) {
        result.dat[it] = lhs.dat[it] / rhs;
    }
    
    return result;
}

inline function M4F m4f_div_f32(M4F lhs, F32 rhs) {
    M4F result = { };
    for Iter_Index(it, 4*4) {
        result.dat[it] = lhs.dat[it] / rhs;
    }
    
    return result;
}

inline function F32 m2f_trace(M2F x) { return x.e11 + x.e22; }
inline function F32 m3f_trace(M3F x) { return x.e11 + x.e22 + x.e33; }
inline function F32 m4f_trace(M4F x) { return x.e11 + x.e22 + x.e33 + x.e44; }

function F32 m2f_det(M2F x);
function F32 m3f_det(M3F x);
function F32 m4f_det(M4F x);
function B32 m4f_inv(M4F x, M4F *solved);

// ------------------------------------------------------------
// #-- Homogeneous matrix ops

function M4F m4f_hom_scale(V3F scale) {
    M4F result = {
        .e11 = scale.x,
        .e22 = scale.y,
        .e33 = scale.z,
        .e44 = 1,
    };
    
    return result;
}

function M4F m4f_hom_translate(V3F translate) {
    M4F result = {
        .e11 = 1,
        .e22 = 1,
        .e33 = 1,
        .e44 = 1,
        
        .e14 = translate.x,
        .e24 = translate.y,
        .e34 = translate.z,
    };
    
    return result;
}

function M4F m4f_hom_rotate_x(F32 angle) {
    F32 c = f32_cos(angle);
    F32 s = f32_sin(angle);

    M4F result = {
        .e11 = 1,

        .e22 =  c,
        .e23 = -s,

        .e32 =  s,
        .e33 =  c,

        .e44 = 1,
    };

    return result;
}

function M4F m4f_hom_rotate_y(F32 angle) {
    F32 c = f32_cos(angle);
    F32 s = f32_sin(angle);

    M4F result = {
        .e11 =  c,
        .e13 =  s,

        .e22 =  1,

        .e31 = -s,
        .e33 =  c,

        .e44 = 1,
    };

    return result;
}

function M4F m4f_hom_rotate_z(F32 angle) {
    F32 c = f32_cos(angle);
    F32 s = f32_sin(angle);

    M4F result = {
        .e11 =  c,
        .e12 = -s,

        .e21 =  s,
        .e22 =  c,

        .e33 =  1,
        .e44 =  1,
    };

    return result;
}

function M4F m4f_hom_rotate_xyz(V3F angles) {
    M4F rx = m4f_hom_rotate_x(angles.x);
    M4F ry = m4f_hom_rotate_y(angles.y);
    M4F rz = m4f_hom_rotate_z(angles.z);

    return m4f_mul(rx, m4f_mul(ry, rz));
}

function M4F m4f_hom_look_at(V3F up, V3F eye, V3F look_at) {
    V3F z_axis = v3f_noz(v3f_sub(eye, look_at));
    V3F x_axis = v3f_noz(v3f_cross(up, z_axis));
    V3F y_axis = v3f_noz(v3f_cross(z_axis, x_axis));
    
    M4F result = {
        .e11 = x_axis.x, .e12 = x_axis.y, .e13 = x_axis.z, .e14 = -v3f_dot(x_axis, eye),
        .e21 = y_axis.x, .e22 = y_axis.y, .e23 = y_axis.z, .e24 = -v3f_dot(y_axis, eye),
        .e31 = z_axis.x, .e32 = z_axis.y, .e33 = z_axis.z, .e34 = -v3f_dot(z_axis, eye),
        .e41 = 0,        .e42 = 0,        .e43 = 0,        .e44 = 1,
    };
    
    return result;
}

function M4F m4f_hom_perspective(F32 aspect_ratio, F32 field_of_view, F32 near_plane, F32 far_plane) {
    F32 tan_fov_by_2 = f32_tan(.5f * field_of_view);
    F32 plane_dist   = far_plane - near_plane;
    
    M4F result = {
        .e11 = f32_div_safe(1.f, aspect_ratio * tan_fov_by_2),
        .e22 = f32_div_safe(1.f, tan_fov_by_2),
        .e33 = -f32_div_safe(far_plane + near_plane, plane_dist),
        .e34 = -f32_div_safe(2.f * far_plane * near_plane, plane_dist),
        .e43 = -1,
    };
    
    return result;
}

function M4F m4f_hom_orthographic(V2F bottom_left, V2F top_right, F32 near_plane, F32 far_plane) {  
    M4F result = {
        .e11 = f32_div_safe(2.f, top_right.x - bottom_left.x),
        .e22 = f32_div_safe(2.f, top_right.y - bottom_left.y),
        .e33 = f32_div_safe(-2.f, far_plane - near_plane),
        .e44 = 1.f,
        
        .e41 = -f32_div_safe(top_right.x + bottom_left.x, top_right.x - bottom_left.x),
        .e42 = -f32_div_safe(top_right.y + bottom_left.y, top_right.y - bottom_left.y),
        .e43 = -f32_div_safe(far_plane + near_plane,      far_plane - near_plane),
    };
    
    return result;
}

// ------------------------------------------------------------
// #-- Ranges

typedef struct Range1_U64 { U64 min; U64 max; } Range1_U64;
typedef struct Range3_F32 { V3F min; V3F max; } Range3_F32;

inline function Range1_U64 range1_u64(U64 min, U64 max) {
  return (Range1_U64) { .min = min, .max = max };
}

inline function U64 range1_u64_len(Range1_U64 r) {
  return r.max - r.min;
}

inline function Range3_F32 range3_f32(V3F min, V3F max) {
  return (Range3_F32) { .min = min, .max = max }; 
}

inline function V3F range3_f32_len(Range3_F32 r) {
  return v3f_sub(r.max, r.min);
}

inline function F32 range3_f32_largest_axis(Range3_F32 r, U32 *axis) {
  V3F d         = v3f_sub(r.max, r.min);
  U32 max_index = 0;
  F32 max_value = d.x;

  if (d.y > max_value) {
    max_index = 1;
    max_value = d.y;
  }
  
  if (d.z > max_value) {
    max_index = 2;
    max_value = d.z;
  }

  if (axis) {
    *axis = max_index;
  }

  return max_value;
}
