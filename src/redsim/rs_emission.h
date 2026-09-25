// NOTE(cmat): Loader for line-source emission CSVs of the form:
//   x0,y0,x1,y1,value_1,value_2
// - Each row becomes one line segment (x0,y0)->(x1,y1) tagged with value_1.
// - value_2 is present in the file but unused here.

typedef struct CSV_Emission_Line {
  V3F a;        // NOTE(cmat): Segment start, world space (z = emission_z).
  V3F b;        // NOTE(cmat): Segment end,   world space (z = emission_z).
  F32 value_1;  // NOTE(cmat): Scalar source strength for this line (e.g. kg/s), to be
                // - distributed across the cells the line passes through.
} CSV_Emission_Line;

typedef struct CSV_Emission_Line_Array {
  U64                 len;
  CSV_Emission_Line  *dat;
} CSV_Emission_Line_Array;

// NOTE(cmat): Reads `file_path`, skips the header row, and parses every remaining row.
// - `emission_z` is the world-space Z to place every 2D (x,y) point at (traffic
// - emissions are usually specified as ground-plane lines, released a little above
// - the road surface).
// - Loads on lane 0 only and broadcasts the result to every lane, same pattern as
// - ugf_grid_init_from_su2 - safe to call unconditionally from every lane.
function CSV_Emission_Line_Array csv_emission_lines_load(Arena *arena, Str08 file_path, F32 emission_z);
