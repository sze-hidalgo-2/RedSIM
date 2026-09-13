typedef U32 FL_Boundary_Type;
enum {
  FL_Boundary_Type_Slip,
  FL_Boundary_Type_No_Slip,
  FL_Boundary_Type_Farfield,
  FL_Boundary_Type_Atmospheric,
  FL_Boundary_Type_Radiation_Wall,
};

typedef struct FL_Boundary_Farfield {
  F32 density;
  V3F velocity;
  F32 pressure;
} FL_Boundary_Farfield;

typedef struct FL_Boundary_Atmospheric {
  F32 temperature_ground;
  F32 pressure_ground;
  F32 gravity;
  F32 lapse_rate;
  F32 wind_angle;
  F32 wind_d;
  F32 wind_z0;
  F32 wind_z_ref;
  F32 wind_u_ref;

  F32 wind_z_cap;


  F32 wind_u_tau;    // added earlier
  F32 mo_length;     // added earlier
  F32 temp_star;     // NEW: temperature scale T*, precomputed per timestep
  F32 thermal_z0;    // NEW: thermal roughness length z_0h (typically ~0.1 * wind_z0)

  F32 latitude_rad;
  F32 longitude_rad;
  F32 day_of_year;
  F32 hour_of_day_utc;   // updated each boundary refresh
} FL_Boundary_Atmospheric;

typedef struct FL_Boundary_Radiation_Wall {
  F32 solar_irradiance;
  F32 gamma_coeff;
  F32 albedo;
  F32 sky_view_factor;
  F32 diffuse_fraction;
  F32 cos_zenith;
  F32 thermal_conductivity;
  F32 temperature_min;
  F32 temperature_max;

  V2F domain_center;
  V2F domain_radius;
} FL_Boundary_Radiation_Wall;

typedef struct FL_Boundary {
  FL_Boundary_Type            type;
  FL_Boundary_Farfield        farfield;
  FL_Boundary_Atmospheric     atmospheric;
  FL_Boundary_Radiation_Wall  radiation_wall;
} FL_Boundary;

typedef struct FL_Boundary_Map {
  U64               map_len;
  FL_Boundary      *map_dat;
} FL_Boundary_Map;

function void         fl_boundary_map_init      (FL_Boundary_Map *boundary, Arena *arena, U64 len);
function FL_Boundary *fl_boundary_map_by_index  (FL_Boundary_Map *boundary, U64 index);


force_inline function F32 fl_boundary_atmosphere_psi_m(F32 zeta);
force_inline function F32 fl_boundary_atmosphere_psi_h(F32 zeta);
force_inline function F32 fl_boundary_atmosphere_temperature_kelvin(F32 z, FL_Boundary_Atmospheric *atm);
force_inline function F32 fl_boundary_atmosphere_pressure(F32 z, FL_Boundary_Atmospheric *atm, F32 R);
force_inline function F32 fl_boundary_atmosphere_density(F32 z, FL_Boundary_Atmospheric *atm, F32 R);
force_inline function V3F fl_boundary_atmosphere_velocity(F32 z, FL_Boundary_Atmospheric *atm);

// NEW: per-timestep calibration from measured met data (host-side, not per-cell).
function void fl_boundary_atmosphere_calibrate(FL_Boundary_Atmospheric *atm, F32 U_meas, F32 z_meas, F32 T_a, F32 GHI, F32 rho_air, F32 cp_air);
