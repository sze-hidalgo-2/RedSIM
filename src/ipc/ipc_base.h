// NOTE(cmat): Types.
typedef struct IPC_Handle_Node {
  struct IPC_Handle_Node *next;
} IPC_Handle_Node;

typedef struct IPC_Handle_List {
  U64              count;
  IPC_Handle_Node *first;
  IPC_Handle_Node *last;
} IPC_Handle_List;

typedef IPC_Handle_Node IPC_Sync_Node;
typedef IPC_Handle_List IPC_Sync_List;

// NOTE(cmat): Initialization, Shutdown.
function void ipc_init        (void);
function void ipc_shutdown    (void);

// NOTE(cmat): Communication primitives.
function U32      ipc_rank_index              (void);
function U32      ipc_rank_count              (void);
function void     ipc_rank_barrier            (void);
function void     ipc_rank_sync_list_init     (IPC_Sync_List *sync_list);
function void     ipc_rank_sync_list_consume  (IPC_Sync_List *sync_list);
function void     ipc_rank_send               (IPC_Sync_List *sync_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag);
function void     ipc_rank_receive            (IPC_Sync_List *sync_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag);

#define IPC_Sync_Scope(sync_list_) \
  Defer_Scope(ipc_rank_sync_list_init(sync_list_), ipc_rank_sync_list_consume(sync_list_))

// NOTE(cmat): Logging utilities.
function void log_ipc_context   (void);
