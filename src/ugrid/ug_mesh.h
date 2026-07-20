// NOTE(cmat): We are storing face information twice, duplicated.
// - This is done on purpose, since it allows us to write SIMD code
// - processing each faces at once, and also gives better cache locality,
// - since we can just fetch the data for the 4 faces all at once,
// - instead of chasing pointers for each face.

typedef struct UG_Cell_Faces {
  U32 adjacent  [4]; // NOTE(cmat): Adjacent cell.
  F32 area      [4]; // NOTE(cmat): Face area.
  F32 normal_x  [4]; // NOTE(cmat): Outward normal vector.
  F32 normal_y  [4];
  F32 normal_z  [4];
} UG_Cell_Faces;

typedef struct UG_Cell_Faces_Verts {
  V3U verts[4];
} UG_Cell_Faces_Verts;

typedef struct UG_Cells {
  U64                  len;
  // V4U                 *verts;
  V3F                 *center;
  F32                 *volume;
  UG_Cell_Faces       *faces;
  UG_Cell_Faces_Verts *faces_verts;
} UG_Cells;

typedef struct UG_Ghosts {
  U64  len;
  U32 *parent_cell;
  U08 *parent_face;
  U32 *marker_index;
} UG_Ghosts;

typedef struct UG_Mesh {
  Arena       arena;
  U64         dimension;
  Range3_F32  bounds;

  UG_Grid     grid;


  UG_Cells    cells;
  UG_Ghosts   ghosts;
} UG_Mesh;

function void ug_mesh_init                (UG_Mesh *mesh);
function void ug_mesh_compute_cells_faces (UG_Mesh *mesh);
function void ug_mesh_assign_ghosts       (UG_Mesh *mesh);
function void ug_mesh_compute_cells       (UG_Mesh *mesh);
function void ug_mesh_reorder_cells       (UG_Mesh *mesh);

