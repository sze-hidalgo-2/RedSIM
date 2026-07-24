# RedSIM: A 3D Finite-Volume HPC Solver CFD Solver

RedSIM is a Finite-Volume HPC Solver, for 3D tetrahedral meshes, meant
for HPC-scale simulations. It leverages MPI + pthreads, with manual NUMA-node managment,
thus significantly minimizing the number of MPI ranks and communication needed between
them.

The code was developed by Matyas Constans, the mathematical models and algorithms developed by Zoltan Horvath.

It was built on top of the [ALICE Engine](https://github.com/matt-const/alice) for efficient
memory managment with arenas, threading (within and outside of NUMA nodes), SIMD, and arena allocations.

It currently support Linux.

## Build Instructions

### Linux
Requires any MPI compiler to be installed to be installed.
Build with ``./build.sh [build-flags]``

### Optional Build Flags
`release`: Fully optimized release build (full optimizations).
`asan`: Enable address sanitization
`no_debug`: Disable debug symbol generation
`no_profile`: No profiling output
`fast_math`: Enable fast-math optimization. May break accuracy and stability.

