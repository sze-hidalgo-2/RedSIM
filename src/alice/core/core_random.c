// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// NOTE(cmat): XORSHIFT implementation.
U64 random_next (Random_Seed *seed) {
    U64 x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    
    *seed = x;
    return x;
}

