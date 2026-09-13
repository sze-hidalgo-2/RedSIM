#include <math.h>

#define FL_VON_KARMAN 0.41f

// ------------------------------------------------------------
// #-- Atmosphere Farfield Modelling

force_inline function F32 fl_solar_cos_zenith(F32 latitude_rad, F32 day_of_year, F32 hour_utc, F32 longitude_rad) {
  F32 declination = 0.4093f * sinf(2.f * f32_pi * (284.f + day_of_year) / 365.f);
  F32 solar_time  = hour_utc + longitude_rad * (180.f / f32_pi) / 15.f;
  F32 hour_angle  = (solar_time - 12.f) * (f32_pi / 12.f);
  F32 cos_zenith  = sinf(latitude_rad) * sinf(declination)
                   + cosf(latitude_rad) * cosf(declination) * cosf(hour_angle);
  return f32_max(cos_zenith, 0.f);
}

force_inline function F32 fl_solar_clear_sky_GHI(F32 cos_zenith, F32 solar_constant, F32 transmittance) {
  F32 result = 0.f;
  if (cos_zenith > 0.f) {
    result = solar_constant * cos_zenith * transmittance;
  }
  return result;
}

force_inline function F32 fl_boundary_atmosphere_psi_m(F32 zeta) {
    F32 result = 0;
    if (zeta < 0.f) {
        F32 x = powf(1.f - 16.f * zeta, 0.25f);
        result = 2.f * logf((1.f + x) * 0.5f)
               + logf((1.f + x * x) * 0.5f)
               - 2.f * atanf(x) + (f32_pi * 0.5f);
    } else {
        F32 zeta_limited = f32_min(zeta, 1.f);
        result = -5.f * zeta_limited;
    }
    return result;
}


force_inline function F32 fl_boundary_atmosphere_psi_h(F32 zeta) {
    F32 result = 0;
    if (zeta < 0.f) {
        // Unstable: same x as psi_m, different combination
        F32 x = powf(1.f - 16.f * zeta, 0.25f);
        result = 2.f * logf((1.f + x * x) * 0.5f);
    } else {
        // Stable: Pr_t ~ 1 assumption, same form as psi_m
        F32 zeta_limited = f32_min(zeta, 1.f);
        result = -5.f * zeta_limited;
    }

    return result;
}

#if 0
force_inline function F32 fl_boundary_atmosphere_temperature_kelvin(F32 z, FL_Boundary_Atmospheric *atm) {
  F32 result = atm->temperature_ground - atm->lapse_rate * z;
  return result;
}
#else

force_inline function F32 fl_boundary_atmosphere_temperature_surface_layer(F32 z, FL_Boundary_Atmospheric *atm) {
    F32 z_capped = f32_max(z, atm->wind_d + atm->thermal_z0);
    F32 zeta     = f32_div_safe(z_capped - atm->wind_d, atm->mo_length);
    F32 psi_h    = fl_boundary_atmosphere_psi_h(zeta);
    F32 log_term = logf((z_capped - atm->wind_d) / atm->thermal_z0);
    F32 result   = atm->temperature_ground + (atm->temp_star / FL_VON_KARMAN) * (log_term - psi_h);
    return result;
}

force_inline function F32 fl_boundary_atmosphere_temperature_kelvin(F32 z, FL_Boundary_Atmospheric *atm) {
    F32 result;
    if (z <= atm->wind_z_cap) {
        result = fl_boundary_atmosphere_temperature_surface_layer(z, atm);
    } else {
        // Above the surface layer: revert to a standard lapse rate,
        // anchored for continuity at the cap height.
        F32 T_cap = fl_boundary_atmosphere_temperature_surface_layer(atm->wind_z_cap, atm);
        result = T_cap - atm->lapse_rate * (z - atm->wind_z_cap);
    }
    return result;
}

#endif

force_inline function F32 fl_boundary_atmosphere_pressure(F32 z, FL_Boundary_Atmospheric *atm, F32 R) {
  F32 result = atm->pressure_ground * powf(1 - (atm->lapse_rate * z) / atm->temperature_ground, atm->gravity / (R * atm->lapse_rate));
  return result;
}

force_inline function F32 fl_boundary_atmosphere_density(F32 z, FL_Boundary_Atmospheric *atm, F32 R) {
  F32 result = fl_boundary_atmosphere_pressure(z, atm, R) / (R * fl_boundary_atmosphere_temperature_kelvin(z, atm));
  return result;
}

#if 0
force_inline function V3F fl_boundary_atmosphere_velocity(F32 z, FL_Boundary_Atmospheric *atm) {
  V3F result = { };
  F32 z_capped = f32_min(z, atm->wind_z_cap);   // new field, e.g. top of surface/boundary layer (~200-300m)
  F32 wind_magnitude = logf(f32_max(z_capped - atm->wind_d, atm->wind_z0) / atm->wind_z0);
  wind_magnitude = f32_div_safe(wind_magnitude, logf((atm->wind_z_ref - atm->wind_d) / atm->wind_z0));
  wind_magnitude = atm->wind_u_ref * wind_magnitude;

  result.x            = wind_magnitude * f32_cos(atm->wind_angle);
  result.y            = wind_magnitude * f32_sin(atm->wind_angle);
  result.z            = 0;

  return result;
}
#else
force_inline function V3F fl_boundary_atmosphere_velocity(F32 z, FL_Boundary_Atmospheric *atm) {
  V3F result = { };

  F32 z_capped = f32_min(z, atm->wind_z_cap);
  F32 height   = f32_max(z_capped - atm->wind_d, atm->wind_z0);
  F32 zeta     = f32_div_safe(height, atm->mo_length);
  F32 psi_m    = fl_boundary_atmosphere_psi_m(zeta);

  F32 wind_magnitude = (atm->wind_u_tau / FL_VON_KARMAN) * (logf(height / atm->wind_z0) - psi_m);
  wind_magnitude = f32_max(wind_magnitude, 0.f);

  result.x = wind_magnitude * f32_cos(atm->wind_angle);
  result.y = wind_magnitude * f32_sin(atm->wind_angle);
  result.z = 0;

  return result;
}
function void fl_boundary_atmosphere_calibrate(FL_Boundary_Atmospheric *atm, F32 U_meas, F32 z_meas, F32 T_a, F32 GHI, F32 rho_air, F32 cp_air) {
  F32 alpha = 0.18f;
  F32 Bo    = 1.5f;

  F32 H_0 = 0;
  if (GHI > 10.f) {
    F32 R_n = (1.f - alpha) * GHI - 100.f;
    H_0 = ((1.f - 0.15f) * R_n) / (1.f + (1.f / Bo));
  } else {
    H_0 = -0.1f * f32_abs(GHI - 100.f);
  }
  // NOTE(cmat): preserve sign when clamping away from zero, unlike a naive
  // abs-clamp — losing the sign here would force every near-neutral case
  // into the unstable branch regardless of actual day/night conditions.
  if (f32_abs(H_0) < 0.001f) {
    H_0 = (H_0 >= 0.f) ? 0.001f : -0.001f;
  }

  F32 height = f32_max(z_meas - atm->wind_d, atm->wind_z0);
  F32 psi_m  = 0.f;
  F32 u_tau  = (U_meas * FL_VON_KARMAN) / logf(height / atm->wind_z0);
  F32 L      = 1e10f;

  for (int iter = 0; iter < 5; iter += 1) {
    L = -((u_tau * u_tau * u_tau) * rho_air * cp_air * T_a) / (FL_VON_KARMAN * atm->gravity * H_0);

    F32 zeta = height / L;
    psi_m = fl_boundary_atmosphere_psi_m(zeta);

    u_tau = (U_meas * FL_VON_KARMAN) / (logf(height / atm->wind_z0) - psi_m);
  }

  atm->wind_u_tau = u_tau;
  atm->mo_length  = L;
  atm->temp_star  = -H_0 / (rho_air * cp_air * u_tau);

  if (atm->thermal_z0 <= 0.f) {
    atm->thermal_z0 = 0.1f * atm->wind_z0;
  }
}

function void fl_boundary_conditions_compute_at_time(
    FL_Boundary_Atmospheric *atm, FL_Boundary_Radiation_Wall *wall,
    F32 hour_of_day_utc, F32 *out_GHI_measured, F32 *out_rho_air, FL_Material *material) {

  atm->hour_of_day_utc = hour_of_day_utc;

  F32 cos_zenith = fl_solar_cos_zenith(atm->latitude_rad, atm->day_of_year, hour_of_day_utc, atm->longitude_rad);
  F32 GHI        = fl_solar_clear_sky_GHI(cos_zenith, 1361.f, 0.75f);

  wall->cos_zenith       = cos_zenith;
  wall->solar_irradiance = GHI;

  *out_GHI_measured = GHI;
  *out_rho_air       = atm->pressure_ground / (material->gas_constant_R * atm->temperature_ground);

  fl_boundary_atmosphere_calibrate(atm, atm->wind_u_ref, atm->wind_z_ref, atm->temperature_ground,
                                    GHI, *out_rho_air, material->gas_constant_R * material->gamma / (material->gamma - 1.f));
}


#endif

force_inline function F32 fl_boundary_radiation_heat_flux(FL_Boundary_Radiation_Wall *rad) {
  // NOTE(cmat): solar_irradiance is treated as GHI (horizontal), so we split it
  // directly into horizontal direct + diffuse components — no cos_zenith division
  // needed (that was only correct if solar_irradiance were DNI).
  F32 I_diff_h = rad->diffuse_fraction * rad->solar_irradiance;
  F32 I_dir_h  = rad->solar_irradiance - I_diff_h;
  F32 q_bc     = rad->gamma_coeff * (1.f - rad->albedo) * (I_dir_h + I_diff_h * rad->sky_view_factor);
  return q_bc;
}


function F32 sdf_rectangle(V2F p, V2F center, V2F half_size) {
    V2F q = v2f_sub(v2f_abs(v2f_sub(p, center)), half_size);

    V2F outside = v2f(f32_max(q.x, 0.f), f32_max(q.y, 0.f));

    F32 outside_distance = v2f_len(outside);
    F32 inside_distance  = f32_min(f32_max(q.x, q.y), 0.f);

    return outside_distance + inside_distance;
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
force_inline function F32 fl_boundary_radiation_wall_equilibrium_temperature(FL_Boundary_Radiation_Wall *rad, V3F inner_center, F32 T_air, F32 rho_air, F32 wind_speed, FL_Material *mat) {
  F32 q_solar = fl_boundary_radiation_heat_flux(rad); // W/m^2

  F32 border_distance   = sdf_rectangle(inner_center.xy, rad->domain_center, rad->domain_radius);
  F32 dist_to_edge      = -border_distance;              // positive when inside the domain
  F32 edge_buffer       = 0.25f * v2f_largest(rad->domain_radius);
  if (dist_to_edge < edge_buffer) {
    q_solar = 0;
  }

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
      F32 T_ghost        = fl_boundary_radiation_wall_equilibrium_temperature(&boundary->radiation_wall, inner_center, T_inner, rho_inner_dim, wind_speed_dim, mat);
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

