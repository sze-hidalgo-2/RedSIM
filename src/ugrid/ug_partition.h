typedef struct UG_Partition_Block {
  U64   cells_len;
  U32  *cells_dat;
} UG_Partition_Block;

typedef struct UG_Partition {
  U32                 blocks_len;
  UG_Partition_Block *blocks_dat;
  U32                *cells_block_index;
  U32                *cells_local_index;

  Range3_F32         *bounds_dat;
} UG_Partition;

// NOTE(cmat): Custom implementations.
function void ug_partition_rcb        (UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);

// NOTE(cmat): Zoltan-based implementations.
function void ug_partition_zoltan_rcb   (UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);
function void ug_partition_zoltan_graph (UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);
