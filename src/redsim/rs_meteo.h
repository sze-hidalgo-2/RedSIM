// NOTE(cmat): Loaders for the AEMET-style daily/hourly meteorological station CSVs used to
// drive the flow's boundary conditions:
//   - 3129_GLOBAL_RADIATION-2014.csv : INDICATIVO;<year>;MES;DIA;RGLO01..RGLO24 (W/m^2, hour-averaged)
//                                      (the CSV's year column header is Spanish "ANO", Latin-1 encoded)
//   - 3195_TEMPERATURE_2014.csv      : STATION-CODE;YEAR;MONTH;DAY;T00..T23   (tenths of degC, instantaneous)
//   - 3195_WIND-2014.csv             : STATION-CODE;YEAR;MONTH;DAY;DIR_00;speed_00;...;DIR_23;speed_23
//                                      (DIR: tens of deg from N, direction wind comes FROM; speed: tenths of m/s)
// - All three cover the full year 2014 (365 rows, one per calendar day, Jan 1 -> Dec 31, no gaps),
//   even though only a slice of it (the simulated window) is ever looked up - so lookups are by
//   calendar date/hour (day-of-year -> row index), not by row order matching the sim directly
//   (unlike rs_nox_ratios.c, whose CSV was trimmed to exactly the sim window).
// - All values are converted to physical SI-ish units (W/m^2, degC, deg, m/s) at parse time;
//   Kelvin/radian conversions for the solver's boundary structs happen at the call site in
//   rs_entry.c, since those units are specific to FL_Boundary_Atmospheric/FL_Boundary_Radiation_Wall.
// - Time is UTC throughout (per the station data descriptions).

typedef struct RS_Meteo_Radiation_Row {
  U32 year, month, day;
  F64 global_radiation_w_m2[24]; // index h (0-23): average GHI over UTC hour [h, h+1) -> RGLO(h+1).
} RS_Meteo_Radiation_Row;

typedef struct RS_Meteo_Radiation_Table {
  U64                     len;
  RS_Meteo_Radiation_Row *dat;
} RS_Meteo_Radiation_Table;

typedef struct RS_Meteo_Temperature_Row {
  U32 year, month, day;
  F64 temperature_c[24]; // index h (0-23): instantaneous reading AT UTC hh:00 -> T(h).
} RS_Meteo_Temperature_Row;

typedef struct RS_Meteo_Temperature_Table {
  U64                       len;
  RS_Meteo_Temperature_Row *dat;
} RS_Meteo_Temperature_Table;

typedef struct RS_Meteo_Wind_Row {
  U32 year, month, day;
  F64 dir_from_deg[24]; // index h (0-23): meteorological bearing (deg, clockwise from N) wind comes FROM, AT hh:00 UTC.
  F64 speed_m_s[24];    // index h (0-23): wind speed (m/s) AT hh:00 UTC.
} RS_Meteo_Wind_Row;

typedef struct RS_Meteo_Wind_Table {
  U64                len;
  RS_Meteo_Wind_Row *dat;
} RS_Meteo_Wind_Table;

// NOTE(cmat): Same lane-0-loads-then-broadcasts pattern as csv_emission_lines_load /
// - rs_nox_ratio_table_load - safe to call unconditionally from every lane.
function RS_Meteo_Radiation_Table   rs_meteo_radiation_table_load   (Arena *arena, Str08 file_path);
function RS_Meteo_Temperature_Table rs_meteo_temperature_table_load (Arena *arena, Str08 file_path);
function RS_Meteo_Wind_Table        rs_meteo_wind_table_load        (Arena *arena, Str08 file_path);

// NOTE(cmat): Look up by calendar date + integer UTC hour-of-day (0-23). Clamps to the
// - nearest valid day/hour if given a date outside the table's range (shouldn't happen for
// - the 2014-11-06..2014-11-30 sim window, since the tables cover the full year).
function F64  rs_meteo_radiation_lookup    (RS_Meteo_Radiation_Table   *table, U32 year, U32 month, U32 day, U32 hour);
function F64  rs_meteo_temperature_lookup  (RS_Meteo_Temperature_Table *table, U32 year, U32 month, U32 day, U32 hour);
function void rs_meteo_wind_lookup         (RS_Meteo_Wind_Table        *table, U32 year, U32 month, U32 day, U32 hour, F64 *out_dir_from_deg, F64 *out_speed_m_s);

// NOTE(cmat): Converts a meteorological wind bearing (deg, clockwise from North, direction
// - the wind comes FROM) into the standard math angle (radians, counter-clockwise from +X)
// - of the vector the wind blows TOWARD - i.e. directly usable as FL_Boundary_Atmospheric's
// - `wind_angle`, given fl_boundary_atmosphere_velocity computes
// - (cos(wind_angle), sin(wind_angle)) as (Easting, Northing) - matching the UTM domain
// - convention used elsewhere in rs_entry.c (e.g. the emission-line offset v3f(441918, 4474610, 0)).
function F64 rs_meteo_wind_angle_math_rad(F64 dir_from_deg);

// ------------------------------------------------------------
// #-- Calendar helpers (Gregorian, proleptic - fine for any year we'll ever simulate)

function B32 rs_meteo_is_leap_year         (U32 year);
function U32 rs_meteo_days_in_year         (U32 year);
function U32 rs_meteo_day_of_year          (U32 year, U32 month, U32 day); // 1-based (Jan 1 -> 1).
function void rs_meteo_date_from_day_of_year(U32 year, U32 day_of_year, U32 *out_month, U32 *out_day);

// ------------------------------------------------------------
// #-- Simulated calendar time + solar position

typedef struct RS_Sim_Time {
  U32 year, month, day;
  U32 hour;      // integer UTC hour-of-day [0,23] (floor of hour_frac) - for hourly table lookups.
  F64 hour_frac; // fractional UTC hour-of-day [0,24) - for continuous calculations (solar position).
} RS_Sim_Time;

// NOTE(cmat): Converts (simulation start date/hour) + (elapsed simulated seconds) into a
// - calendar date/time. Handles day (and, in principle, year) rollover, even though the
// - current 2014-11-06..2014-11-30 window never crosses a year boundary.
function RS_Sim_Time rs_sim_time_from_elapsed(U32 start_year, U32 start_month, U32 start_day, U32 start_hour, F64 elapsed_seconds);

typedef struct RS_Solar_Position {
  F64 declination_rad;
  F64 hour_angle_rad;
  F64 equation_of_time_min;
  F64 cos_zenith; // NOTE(cmat): NOT clamped - can be <= 0 when the sun is below the horizon; clamp at the call site before dividing by it.
} RS_Solar_Position;

// NOTE(cmat): Spencer (1971) Fourier-series solar declination + equation-of-time, combined
// - with the standard hour-angle formula, to get cos(solar zenith angle) for a given
// - lat/lon (deg) and UTC date/time - accurate to a few tenths of a degree / minutes,
// - which is plenty for a boundary condition (no measured sun-angle data exists to compare
// - against, so this is the only option, as opposed to full NOAA-precision ephemeris code).
function RS_Solar_Position rs_solar_position(F64 latitude_deg, F64 longitude_deg, RS_Sim_Time time);
