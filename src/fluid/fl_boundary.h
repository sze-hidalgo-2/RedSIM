typedef U32 FL_Boundary_Type;
enum {
  FL_Boundary_Type_Slip,
  FL_Boundary_Type_Farfield,
};

typedef struct FL_Boundary_Farfield {
  F32 density;
  V3F velocity;
  F32 pressure;
} FL_Boundary_Farfield;

typedef struct FL_Boundary {
  FL_Boundary_Type      type;
  FL_Boundary_Farfield  farfield;
} FL_Boundary;

typedef struct FL_Boundary_Map {
  U64               map_len;
  FL_Boundary      *map_dat;
} FL_Boundary_Map;

function void         fl_boundary_map_init      (FL_Boundary_Map *boundary, Arena *arena, U64 len);
function FL_Boundary *fl_boundary_map_by_index  (FL_Boundary_Map *boundary, U64 index);
