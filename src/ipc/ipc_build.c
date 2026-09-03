#if BUILD_IPC_MPI
# undef function
# include <mpi.h>
# define function static
#endif

#include "ipc_base.c"
