// NOTE(cmat): Loader for the traffic-emission NOx time-of-day/day-of-week ratio CSV:
//   Year;Month;Day;Day of the week;Hour;Period factor;Anual_factor
// - Expected to be one row per hour, in chronological order, with no gaps, and row 0
// - lining up with the simulation's start date/hour - so a later simulated time can be
// - found by row index (elapsed hours) instead of a date search.

typedef struct RS_NOX_Ratio_Row {
  U32   year;
  U32   month;
  U32   day;
  U32   hour;
  Str08 weekday;       // NOTE(cmat): "Day of the week" column, kept for logging/sanity checks only.
  F64   period_factor; // NOTE(cmat): "Period factor" column -> feeds `day_weight`.
  F64   annual_factor; // NOTE(cmat): "Anual_factor" column, kept for reference / future use.
} RS_NOX_Ratio_Row;

typedef struct RS_NOX_Ratio_Table {
  U64               len;
  RS_NOX_Ratio_Row *dat;

  // NOTE(cmat): Calendar date/hour of table row 0 (i.e. the simulation's start time).
  U32 start_year;
  U32 start_month;
  U32 start_day;
  U32 start_hour;
} RS_NOX_Ratio_Table;

// NOTE(cmat): Reads `file_path`, skips the header row, and parses every remaining row.
// - Loads on lane 0 only and broadcasts the result to every lane, same pattern as
// - csv_emission_lines_load - safe to call unconditionally from every lane.
function RS_NOX_Ratio_Table rs_nox_ratio_table_load(Arena *arena, Str08 file_path);

// NOTE(cmat): Returns the "Period factor" (day_weight) for `elapsed_seconds` of simulated
// - time after the table's start date/hour (row 0). Floors to the containing hour, and
// - clamps to the first/last row if `elapsed_seconds` falls outside the table's range.
function F64 rs_nox_ratio_table_day_weight(RS_NOX_Ratio_Table *table, F64 elapsed_seconds);
