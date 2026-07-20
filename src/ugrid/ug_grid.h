global const U32 UG_Cell_Invalid = u32_limit_max;
global const U08 UG_Face_Invalid = u08_limit_max;

typedef struct UG_Grid_Marker_Elems {
  U64  len;
  V3U *verts;
} UG_Grid_Marker_Elems;

typedef struct UG_Grid_Markers {
  U64                    len;
  Str08                 *tags;
  UG_Grid_Marker_Elems  *elems;
} UG_Grid_Markers;

typedef struct UG_Grid_Verts {
  U64  len;
  F32 *x;
  F32 *y;
  F32 *z;
} UG_Grid_Verts;

typedef struct UG_Grid_Elems {
  U64  len;
  V4U *verts;
} UG_Grid_Elems;

typedef struct UG_Grid {
  Arena             arena;
  UG_Grid_Verts     verts;
  UG_Grid_Elems     elems;
  UG_Grid_Markers   markers;
} UG_Grid;

function void ug_grid_init(UG_Grid *grid);
