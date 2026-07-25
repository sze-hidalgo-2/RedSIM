// NOTE(cmat): Types.
typedef struct IPC_Handle_Node {
  struct IPC_Handle_Node *next;
} IPC_Handle_Node;

typedef struct IPC_Handle_List {
  U64              count;
  IPC_Handle_Node *first;
  IPC_Handle_Node *last;
} IPC_Handle_List;

typedef IPC_Handle_Node IPC_Request_Node;
typedef IPC_Handle_List IPC_Request_List;

// NOTE(cmat): Initialization, Shutdown.
function void ipc_init        (void);
function void ipc_shutdown    (void);

// NOTE(cmat): Global Rank info.
function U32      ipc_rank_index                (void);
function U32      ipc_rank_count                (void);

// NOTE(cmat): Node-local and node-global  Rank info. (NUMA index, NUMA count).
function U32      ipc_rank_local_node_index     (void);
function U32      ipc_rank_local_node_count     (void);

function U32      ipc_rank_global_node_count    (void);

// NOTE(cmat): Communication primitives.
function void     ipc_rank_barrier              (void);
function void     ipc_rank_request_list_init    (IPC_Request_List *request_list);
function void     ipc_rank_request_list_destroy (IPC_Request_List *request_list);
function void     ipc_rank_request_list_start   (IPC_Request_List *request_list);
function void     ipc_rank_request_list_wait    (IPC_Request_List *request_list);
function void     ipc_rank_record_send          (IPC_Request_List *request_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag);
function void     ipc_rank_record_receive       (IPC_Request_List *request_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag);
function F64      ipc_rank_minimum              (F64 value);

#define IPC_Request_Scope(request_list_) \
  Defer_Scope(ipc_rank_request_list_start(request_list_), ipc_rank_request_list_wait(request_list_))

// NOTE(cmat): Logging utilities.
function void log_ipc_context(void);
