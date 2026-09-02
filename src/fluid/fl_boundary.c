#include <math.h>

// ------------------------------------------------------------
// #-- Atmosphere Farfield Modelling

force_inline function F32 fl_boundary_atmosphere_temperature_kelvin(F32 z, FL_Boundary_Atmospheric *atm) {
  F32 result = atm->temperature_ground - atm->lapse_rate * z;
  return result;
}

force_inline function F32 fl_boundary_atmosphere_pressure(F32 z, FL_Boundary_Atmospheric *atm, F32 R) {
  F32 result = atm->pressure_ground * powf(1 - (atm->lapse_rate * z) / atm->temperature_ground, atm->gravity / (R * atm->lapse_rate));
  return result;
}

force_inline function F32 fl_boundary_atmosphere_density(F32 z, FL_Boundary_Atmospheric *atm, F32 R) {
  F32 result = fl_boundary_atmosphere_pressure(z, atm, R) / (R * fl_boundary_atmosphere_temperature_kelvin(z, atm));
  return result;
}

force_inline function V3F fl_boundary_atmosphere_velocity(F32 z, FL_Boundary_Atmospheric *atm) {
  V3F result          = { };
  F32 wind_magnitude  = f32_log(f32_max(z - atm->wind_d, atm->wind_z0) / atm->wind_z0);
  wind_magnitude      = f32_div_safe(wind_magnitude, f32_log((atm->wind_z_ref - atm->wind_d) / atm->wind_z0));
  wind_magnitude      = atm->wind_u_ref * wind_magnitude;

  result.x            = wind_magnitude * f32_cos(atm->wind_angle);
  result.y            = wind_magnitude * f32_sin(atm->wind_angle);
  result.z            = 0;

  return result;
}

force_inline function F32 fl_boundary_radiation_heat_flux(FL_Boundary_Radiation_Wall *rad) {
  F32 I_diff = rad->diffuse_fraction * rad->solar_irradiance;
  F32 I_dir  = (rad->solar_irradiance - I_diff * rad->sky_view_factor) / rad->cos_zenith;
  F32 q_bc   = rad->gamma_coeff * (1.f - rad->albedo) * (I_dir * rad->cos_zenith + I_diff * rad->sky_view_factor);
  return q_bc;
}

// NOTE(cmat): Mesh-independent radiative-convective equilibrium wall temperature.
// - Solves q_solar = h_conv * (T_wall - T_air) + sigma * eps * (T_wall^4 - T_air^4)
//   for T_wall, using a local linearization of the radiative loss term around T_air
//   so we get a closed-form solution instead of an iterative one. This replaces the
//   old dn-based conduction estimate, which depended on local mesh cell size near
//   the wall (dn = distance from cell center to ghost center) and therefore gave
//   different wall temperatures purely from mesh resolution changes, not physics.
// - h_conv uses a simple bulk forced-convection estimate; C_H (~0.002-0.005) is a
//   typical near-surface heat-transfer coefficient for atmospheric boundary layers.
// - Reuses rad->gamma_coeff as effective longwave emissivity (0-1), since it's
//   already a dimensionless absorption/efficiency-style coefficient on the struct.
force_inline function F32 fl_boundary_radiation_wall_equilibrium_temperature(
    FL_Boundary_Radiation_Wall *rad, F32 T_air, F32 rho_air, F32 wind_speed, FL_Material *mat) {
  F32 q_solar = fl_boundary_radiation_heat_flux(rad); // W/m^2

  // NOTE(cmat): mat->cp is the solver's internal *non-dimensional* specific heat
  // (1/(gamma-1)) used in normalized flux/EOS math — NOT a physical J/(kg*K)
  // value. For this dimensional heat-balance calc we need the real cp, computed
  // from gas_constant_R (which is always physical/unscaled, unlike gas_constant).
  F32 cp_dim = mat->gas_constant_R * mat->gamma / (mat->gamma - 1.f); // J/(kg*K)

  F32 C_H     = 0.003f;
  F32 h_conv  = rho_air * cp_dim * C_H * f32_max(wind_speed, 0.5f); // W/(m^2*K)

  F32 stefan_boltzmann = 5.670374e-8f;
  F32 emissivity       = rad->gamma_coeff;
  F32 h_rad = 4.f * stefan_boltzmann * emissivity * T_air * T_air * T_air;

  F32 T_wall = T_air + q_solar / (h_conv + h_rad);
  return T_wall;
}

// ------------------------------------------------------------
// #-- Boundary Condition Handling

function void fl_boundary_map_init(FL_Boundary_Map *boundary, Arena *arena, U64 len) {
  Zero_Fill(boundary);

  boundary->map_len = len;

  if (lane_index() == 0) {
    boundary->map_dat = arena_push_count(arena, FL_Boundary, len);
  }
  
  lane_broadcast_ptr(&boundary->map_dat, 0);

  // NOTE(cmat): All boundaries are initialized to slip by default.
  for Iter_Range(it, lane_range(len)) {
    boundary->map_dat[it] = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
  }

  lane_barrier();
}

function FL_Boundary *fl_boundary_map_by_index(FL_Boundary_Map *boundary, U64 index) {
  FL_Boundary *result = 0;
  if (index < boundary->map_len) {
    result = &boundary->map_dat[index];
  }

  return result;
}

force_inline function V5F fl_boundary_map_ghost(FL_Boundary_Map *bmap, U32 marker_index, V5F inner, V3F inner_center, V3F ghost_center, V3F normal, FL_Scale *scale, FL_Material *mat, V3F gravity) {
  V5F result  = v5f(0, 0, 0, 0, 0);
  F32 rho     = inner.x1;
  F32 rho_v1  = inner.x2;
  F32 rho_v2  = inner.x3;
  F32 rho_v3  = inner.x4;
  F32 energy  = inner.x5; 

  FL_Boundary *boundary = &bmap->map_dat[marker_index];

  // NOTE(cmat): Compute boundary pressure, density velocity,
  // - for different farfield types.
  F32 boundary_pressure = 0;
  F32 boundary_density  = 0;
  V3F boundary_velocity = { };

  switch (boundary->type) {
    case FL_Boundary_Type_Farfield: {
      boundary_pressure = boundary->farfield.pressure;
      boundary_density  = boundary->farfield.density;
      boundary_velocity = boundary->farfield.velocity;
    } break;

    case FL_Boundary_Type_Atmospheric: {
      F32 z = (ghost_center.z * scale->length) + scale->offset.z;
      boundary_pressure = fl_boundary_atmosphere_pressure (z, &boundary->atmospheric, mat->gas_constant_R);
      boundary_density  = fl_boundary_atmosphere_density  (z, &boundary->atmospheric, mat->gas_constant_R);
      boundary_velocity = fl_boundary_atmosphere_velocity (z, &boundary->atmospheric);

      boundary_pressure = fl_scale_normalize_pressure (scale, boundary_pressure);
      boundary_density  = fl_scale_normalize_density  (scale, boundary_density);
      boundary_velocity = fl_scale_normalize_velocity (scale, boundary_velocity);
    } break;
  }

  switch (boundary->type) {
    case FL_Boundary_Type_Slip: {
      V3F velocity        = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      V3F ghost_velocity  = v3f_reflect(velocity, normal);
      F32 inner_pressure  = (mat->gamma - 1.f) * (energy - 0.5f * rho * v3f_len2(velocity));
      F32 ghost_pressure  = inner_pressure + rho * v3f_dot(gravity, v3f_sub(ghost_center, inner_center));

      result.x1 = rho;
      result.x2 = rho * ghost_velocity.x;
      result.x3 = rho * ghost_velocity.y;
      result.x4 = rho * ghost_velocity.z;
      result.x5 = fl_state_energy_from_pressure(mat, result.x1, result.x234, ghost_pressure);
    } break;

    case FL_Boundary_Type_No_Slip: {
      V3F velocity       = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      V3F ghost_velocity = v3f_mul(-1.f, velocity);
      F32 inner_pressure = (mat->gamma - 1.f) * (energy - 0.5f * rho * v3f_len2(velocity));
      F32 ghost_pressure = inner_pressure + rho * v3f_dot(gravity, v3f_sub(ghost_center, inner_center));

      result.x1 = rho;
      result.x2 = rho * ghost_velocity.x;
      result.x3 = rho * ghost_velocity.y;
      result.x4 = rho * ghost_velocity.z;
      result.x5 = fl_state_energy_from_pressure(mat, result.x1, result.x234, ghost_pressure);
    } break;

    case FL_Boundary_Type_Radiation_Wall: {
      V3F velocity       = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      F32 v2_inner       = v3f_len2(velocity);
      F32 P_inner_nd     = (mat->gamma - 1.f) * (energy - 0.5f * rho * v2_inner);
      F32 P_inner        = fl_scale_denormalize_pressure(scale, P_inner_nd);
      F32 rho_inner_dim  = fl_scale_denormalize_density(scale, rho);
      F32 T_inner        = P_inner / (rho_inner_dim * mat->gas_constant_R);

      F32 wind_speed_dim = fl_scale_denormalize_velocity(scale, f32_sqrt(v2_inner));
      F32 T_ghost        = fl_boundary_radiation_wall_equilibrium_temperature(&boundary->radiation_wall, T_inner, rho_inner_dim, wind_speed_dim, mat);
      T_ghost            = f32_max(boundary->radiation_wall.temperature_min, f32_min(boundary->radiation_wall.temperature_max, T_ghost));

      V3F delta_pos      = v3f_sub(ghost_center, inner_center);
      V3F gravity_dim    = v3f_mul((scale->sound_speed * scale->sound_speed) / scale->length, gravity);
      F32 P_ghost        = P_inner + rho_inner_dim * v3f_dot(gravity_dim, delta_pos) * scale->length;
      F32 rho_ghost_dim  = P_ghost / (mat->gas_constant_R * T_ghost);
      F32 rho_ghost      = fl_scale_normalize_density  (scale, rho_ghost_dim);
      F32 P_ghost_nd     = fl_scale_normalize_pressure (scale, P_ghost);
      V3F ghost_velocity = v3f_mul(-1.f, velocity);

      result.x1          = rho_ghost;
      result.x2          = rho_ghost * ghost_velocity.x;
      result.x3          = rho_ghost * ghost_velocity.y;
      result.x4          = rho_ghost * ghost_velocity.z;
      result.x5          = fl_state_energy_from_pressure(mat, result.x1, result.x234, P_ghost_nd);
    } break;

    // NOTE(cmat): Riemann-invariant farfield.
    // - Notably, this handles cases where the velocity is parallel to farfield boundary faces.
    case FL_Boundary_Type_Atmospheric:
    case FL_Boundary_Type_Farfield: {
      V3F inner_velocity    = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      F32 kinetic           = 0.5f * v3f_len2(inner_velocity);
      F32 inner_pressure    = (mat->gamma - 1.f) * (energy - rho * kinetic);
      F32 vn_i              = v3f_dot(inner_velocity, normal);
      F32 vn_inf            = v3f_dot(boundary_velocity, normal);
      F32 a_i               = f32_sqrt(mat->gamma * inner_pressure   / rho);
      F32 a_inf             = f32_sqrt(mat->gamma * boundary_pressure / boundary_density);

      // NOTE(cmat): Riemann invariants along outgoing/incoming characteristics.
      F32 R_plus            = vn_i   + 2.f * a_i   / (mat->gamma - 1.f);
      F32 R_minus           = vn_inf - 2.f * a_inf / (mat->gamma - 1.f);

      F32 vn_b              = 0.5f * (R_plus + R_minus);
      F32 a_b               = 0.25f * (mat->gamma - 1.f) * (R_plus - R_minus);

      // NOTE(cmat): Now we compute the upwind-selection; and tangential velocity.
      F32 rho_b, p_b    = 0.f;
      V3F tangential_b  = v3f(0.f, 0.f, 0.f);

      // NOTE(cmat): Outflow case.
      if (vn_b >= 0.f) {
        F32 s        = inner_pressure / powf(rho, mat->gamma);
        rho_b        = powf(a_b * a_b / (mat->gamma * s), 1.f / (mat->gamma - 1.f));
        tangential_b = v3f_sub(inner_velocity, v3f_mul(vn_i, normal));

      // NOTE(cmat): Inflow case.
      } else {
        F32 s        = boundary_pressure / powf(boundary_density, mat->gamma);
        rho_b        = powf(a_b * a_b / (mat->gamma * s), 1.f / (mat->gamma - 1.f));
        tangential_b = v3f_sub(boundary_velocity, v3f_mul(vn_inf, normal));
      }

      p_b       = rho_b * a_b * a_b / mat->gamma;
      V3F vel_b = v3f_add(tangential_b, v3f_mul(vn_b, normal));

      result.x1 = rho_b;
      result.x2 = rho_b * vel_b.x;
      result.x3 = rho_b * vel_b.y;
      result.x4 = rho_b * vel_b.z;
      result.x5 = p_b / (mat->gamma - 1.f) + 0.5f * rho_b * v3f_len2(vel_b);
    } break;
  }

  return result;
}

