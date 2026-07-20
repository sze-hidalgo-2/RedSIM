typedef struct UG_Partition_Bucket {
  U64  cells_len;
  U32 *cells_dat;
} UG_Partition_Bucket;

typedef struct UG_Partition {
  U32                   partition_count;
  UG_Partition_Bucket  *buckets_dat;
} UG_Partition;

function void ug_partition_rcb(UG_Partition *partition, Arena *arena, UG_Grid *grid, U32 partition_count);
