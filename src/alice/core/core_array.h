// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Reordering

// TODO(cmat): Make this API more user friendly.
function void array_reorder(U64 array_len, U64 type_size, U08 *array_dat, U64 key_size, U64 *key_dat);

// ------------------------------------------------------------
// #-- Radix Sort

// TODO(cmat): Change API to take sizes and offsets instead of multiples of U32's / U64's.
function void array_sort_radix_u32(U64 array_len, U64 array_stride, U64 array_offset, U32 *array_dat);
function void array_sort_radix_u64(U64 array_len, U64 array_stride, U64 array_offset, U64 *array_dat);

// ------------------------------------------------------------
// #-- Space Filling Curves

typedef U64 Morton64;
function Morton64 morton64_encode_v3f(V3F position_unit);

