typedef struct UG_Partition_Block {
  U64   cells_len;
  U32  *cells_dat;
} UG_Partition_Block;

typedef struct UG_Partition {
  U32                 blocks_len;
  UG_Partition_Block *blocks_dat;
  U32                *cells_block_index;
} UG_Partition;

function void ug_partition_rcb(UG_Partition *partition, Arena *arena, UG_Mesh *mesh, U32 partition_count);
