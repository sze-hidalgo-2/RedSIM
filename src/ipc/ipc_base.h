// NOTE(cmat): Initialization, Shutdown.
function void ipc_init        (void);
function void ipc_shutdown    (void);

// NOTE(cmat): Communication primitives.
function U32  ipc_rank_index    (void);
function U32  ipc_rank_count    (void);
function void ipc_rank_barrier  (void);

// NOTE(cmat): Logging utilities.
function void log_ipc_context   (void);
