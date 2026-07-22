// NOTE(cmat): We are storing face information twice, duplicated.
// - This is done on purpose, since it allows us to write SIMD code
// - processing each faces at once, and also gives better cache locality,
// - since we can just fetch the data for the 4 faces all at once,
// - instead of chasing pointers for each face.
//
// NOTE(cmat): For adjacent cell indices, we have the following layout:
// [ inner cells | halo cells | ghost cells ]
// - inner cells in [0, cells.len)
// - halo  cells in [cells.len, cells.len + halos.len)
// - ghost cells in [cells.len + halos.len, cells.len + halos.len + ghosts.len)
//
typedef struct UG_Cell_Faces {
  U32 adjacent  [4]; // NOTE(cmat): Adjacent cell.
  F32 area      [4]; // NOTE(cmat): Face area.
  F32 normal_x  [4]; // NOTE(cmat): Outward normal vector.
  F32 normal_y  [4];
  F32 normal_z  [4];
} UG_Cell_Faces;

typedef struct UG_Cells {
  U64            len;
  V3F           *center;
  F32           *volume;
  UG_Cell_Faces *faces;
} UG_Cells;

typedef struct UG_Halos {
  U64         len;
  U64         block_len;
  Range1_U64 *block_range;
  U32        *cell_send;
} UG_Halos;

typedef struct UG_Ghosts {
  U64  len;
  U32 *parent_cell;
  U08 *parent_face;
  U32 *marker_index;
} UG_Ghosts;

typedef struct UG_Mesh {
  Range3_F32  bounds;
  UG_Grid    *grid;
  UG_Cells    cells;
  UG_Halos    halos;
  UG_Ghosts   ghosts;
} UG_Mesh;

typedef struct UG_Mesh_Array {
  U32      len;
  UG_Mesh *dat;
} UG_Mesh_Array;

struct UG_Partition;

function void ug_mesh_init_from_grid        (UG_Mesh *mesh, UG_Grid *grid, Arena *arena);
function void ug_mesh_array_from_partition  (UG_Mesh_Array *mesh_array, UG_Mesh *global_mesh, struct UG_Partition *partition, Arena *arena);
// function void ug_mesh_optimize_reorder    (UG_Mesh *mesh);

