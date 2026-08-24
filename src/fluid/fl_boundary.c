#include <math.h>

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

force_inline function V5F fl_boundary_map_ghost(FL_Boundary_Map *bmap, U32 marker_index, V5F inner, V3F normal, F32 gamma) {
  V5F result  = v5f(0, 0, 0, 0, 0);
  F32 rho     = inner.x1;
  F32 rho_v1  = inner.x2;
  F32 rho_v2  = inner.x3;
  F32 rho_v3  = inner.x4;
  F32 energy  = inner.x5; 

  FL_Boundary *boundary = &bmap->map_dat[marker_index];
  switch (boundary->type) {
    case FL_Boundary_Type_Slip: {
      V3F velocity        = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      V3F ghost_velocity  = v3f_reflect(velocity, normal);

      result.x1 = rho;
      result.x2 = rho * ghost_velocity.x;
      result.x3 = rho * ghost_velocity.y;
      result.x4 = rho * ghost_velocity.z;
      result.x5 = energy;
    } break;

    case FL_Boundary_Type_No_Slip: {
      V3F velocity = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      V3F ghost_velocity = v3f_mul(-1.f, velocity);

      result.x1 = rho;
      result.x2 = rho * ghost_velocity.x;
      result.x3 = rho * ghost_velocity.y;
      result.x4 = rho * ghost_velocity.z;
      result.x5 = energy;
    } break;

    // NOTE(cmat): Riemann-invariant farfield.
    // - Notably, this handles cases where the velocity is parallel to farfield boundary faces.
    case FL_Boundary_Type_Farfield: {
      F32 boundary_pressure = boundary->farfield.pressure;
      F32 boundary_density  = boundary->farfield.density;
      V3F boundary_velocity = boundary->farfield.velocity;
      V3F inner_velocity    = v3f_div(v3f(rho_v1, rho_v2, rho_v3), rho);
      F32 kinetic           = 0.5f * v3f_len2(inner_velocity);
      F32 inner_pressure    = (gamma - 1.f) * (energy - rho * kinetic);
      F32 vn_i              = v3f_dot(inner_velocity, normal);
      F32 vn_inf            = v3f_dot(boundary_velocity, normal);
      F32 a_i               = f32_sqrt(gamma * inner_pressure   / rho);
      F32 a_inf             = f32_sqrt(gamma * boundary_pressure / boundary_density);

      // NOTE(cmat): Riemann invariants along outgoing/incoming characteristics.
      F32 R_plus            = vn_i   + 2.f * a_i   / (gamma - 1.f);
      F32 R_minus           = vn_inf - 2.f * a_inf / (gamma - 1.f);

      F32 vn_b              = 0.5f * (R_plus + R_minus);
      F32 a_b               = 0.25f * (gamma - 1.f) * (R_plus - R_minus);

      // NOTE(cmat): Now we compute the upwind-selection; and tangential velocity.
      F32 rho_b, p_b    = 0.f;
      V3F tangential_b  = v3f(0.f, 0.f, 0.f);

      // NOTE(cmat): Outflow case.
      if (vn_b >= 0.f) {
        F32 s        = inner_pressure / powf(rho, gamma);
        rho_b        = powf(a_b * a_b / (gamma * s), 1.f / (gamma - 1.f));
        tangential_b = v3f_sub(inner_velocity, v3f_mul(vn_i, normal));

      // NOTE(cmat): Inflow case.
      } else {
        F32 s        = boundary_pressure / powf(boundary_density, gamma);
        rho_b        = powf(a_b * a_b / (gamma * s), 1.f / (gamma - 1.f));
        tangential_b = v3f_sub(boundary_velocity, v3f_mul(vn_inf, normal));
      }

      p_b       = rho_b * a_b * a_b / gamma;
      V3F vel_b = v3f_add(tangential_b, v3f_mul(vn_b, normal));

      result.x1 = rho_b;
      result.x2 = rho_b * vel_b.x;
      result.x3 = rho_b * vel_b.y;
      result.x4 = rho_b * vel_b.z;
      result.x5 = p_b / (gamma - 1.f) + 0.5f * rho_b * v3f_len2(vel_b);
    } break;

#if 0
    // NOTE(cmat): Far-field.
    case FL_Boundary_Type_Farfield: {
      F32 boundary_pressure = boundary->farfield.pressure;
      F32 boundary_density  = boundary->farfield.density;
      V3F boundary_velocity = boundary->farfield.velocity;
      
      F32 vn = v3f_dot(boundary_velocity, normal);                        // NOTE(cmat): Projected velocity.
      F32 a  = f32_sqrt(gamma * boundary_pressure / boundary_density);    // NOTE(cmat): Speed of sound.
      F32 M  = vn / a;                                                    // NOTE(cmat): Mach number.

      // NOTE(cmat): According to the Mach number, choose proper boundary configuration.

      if (M <= -1.f) { 
        // NOTE(cmat): I. Supersonic Inflow. All characteristics from outside.
        result.x1 = boundary_density;
        result.x2 = boundary_density * boundary_velocity.x;
        result.x3 = boundary_density * boundary_velocity.y;
        result.x4 = boundary_density * boundary_velocity.z;
        result.x5 = boundary_pressure / (gamma - 1.f) + .5f * boundary_density * v3f_len2(boundary_velocity);
      
      } else if (M > -1.f && M < .0f) {
        // NOTE(cmat): II. Subsonic Inflow. 1 outgoing characteristics, 4 incoming.
        F32 kinetic         = 0.5f * v3f_len2(v3f(rho_v1,rho_v2,rho_v3)) / rho;
        F32 inner_pressure  = (gamma - 1.f) * (energy - kinetic);

        result.x1 = boundary_density;
        result.x2 = boundary_density * boundary_velocity.x;
        result.x3 = boundary_density * boundary_velocity.y;
        result.x4 = boundary_density * boundary_velocity.z;
        result.x5 = (inner_pressure / (gamma - 1.f)) + .5f * boundary_density * v3f_len2(boundary_velocity);

      } else if (M >= 0.f && M < 1.f) {
        // NOTE(cmat): III. Subsonic Outflow. 4 outgoing characteristics, 1 incoming.
        result.x1 = rho;
        result.x2 = rho_v1;
        result.x3 = rho_v2;
        result.x4 = rho_v3;
        result.x5 = boundary_pressure / (gamma - 1.f) + .5f * v3f_len2(v3f(rho_v1, rho_v2, rho_v3)) / rho;
      } else {
        // NOTE(cmat): IV. Supersonic Outflow. All characteristics from inside.
        result.x1 = rho;
        result.x2 = rho_v1;
        result.x3 = rho_v2;
        result.x4 = rho_v3;
        result.x5 = energy;
      }
    } break;
#endif
  }

  return result;
}

