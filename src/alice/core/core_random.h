// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

typedef U64 Random_Seed;

function U64 random_next (Random_Seed *seed);

// NOTE(cmat): Random values in the range: [min, max].
inline function I64 i64_random            (Random_Seed *seed, I64 min, I64 max) { return min + (I64)(random_next(seed) % (U64)(max - min + 1)); }
inline function U64 u64_random            (Random_Seed *seed, U64 min, U64 max) { return min + (U64)(random_next(seed) % (U64)(max - min + 1)); }

// NOTE(cmat): Random unilateral values (between [0, 1]).
inline function F32 f32_random_unilateral (Random_Seed *seed) { return (F32)random_next(seed) / (F32)u64_limit_max;                                                                               }
inline function F64 f64_random_unilateral (Random_Seed *seed) { return (F64)random_next(seed) / (F64)u64_limit_max;                                                                               }
inline function V2F v2f_random_unilateral (Random_Seed *seed) { return v2f(f32_random_unilateral(seed), f32_random_unilateral(seed));                                                             }
inline function V3F v3f_random_unilateral (Random_Seed *seed) { return v3f(f32_random_unilateral(seed), f32_random_unilateral(seed),  f32_random_unilateral(seed));                               }
inline function V4F v4f_random_unilateral (Random_Seed *seed) { return v4f(f32_random_unilateral(seed), f32_random_unilateral(seed),  f32_random_unilateral(seed), f32_random_unilateral(seed));  }

// NOTE(cmat): Random bilateral values (between [-1, 1]).
inline function F32 f32_random_bilateral  (Random_Seed *seed) { return 2.f * f32_random_unilateral(seed) - 1.f;                                                                                   }
inline function F64 f64_random_bilateral  (Random_Seed *seed) { return 2.  * f64_random_unilateral(seed) - 1.;                                                                                    }
inline function V2F v2f_random_bilateral  (Random_Seed *seed) { return v2f(f32_random_bilateral (seed), f32_random_bilateral (seed));                                                             }
inline function V3F v3f_random_bilateral  (Random_Seed *seed) { return v3f(f32_random_bilateral (seed), f32_random_bilateral (seed),  f32_random_bilateral (seed));                               }
inline function V4F v4f_random_bilateral  (Random_Seed *seed) { return v4f(f32_random_bilateral (seed), f32_random_bilateral (seed),  f32_random_bilateral (seed),  f32_random_bilateral(seed));  }

