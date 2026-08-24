function FL_Flux fl_flux_viscous(V5F left_primitive, V3F left_grad[5], V5F right_primitive, V3F right_grad[5], V3F normal, F32 area, F32 volume, F32 mu, F32 k_thermal, F32 gas_constant, F32 gamma, F32 prandtl) {
  // NOTE(cmat): grad[0] = d(rho)/dx,y,z
  // NOTE(cmat): grad[1] = d(u)/dx,y,z
  // NOTE(cmat): grad[2] = d(v)/dx,y,z
  // NOTE(cmat): grad[3] = d(w)/dx,y,z
  // NOTE(cmat): grad[4] = d(p)/dx,y,z

  // NOTE(cmat): Face-averaged velocity (for the viscous work term tau_ij * u_i).
  V3F left_vel   = v3f(left_primitive.x2,  left_primitive.x3,  left_primitive.x4);
  V3F right_vel  = v3f(right_primitive.x2, right_primitive.x3, right_primitive.x4);
  V3F face_vel   = v3f_mul(.5f, v3f_add(left_vel, right_vel));

  // NOTE(cmat): Face-averaged velocity gradient tensor.
  F32 du_dx = .5f * (left_grad[1].x + right_grad[1].x);
  F32 du_dy = .5f * (left_grad[1].y + right_grad[1].y);
  F32 du_dz = .5f * (left_grad[1].z + right_grad[1].z);
  F32 dv_dx = .5f * (left_grad[2].x + right_grad[2].x);
  F32 dv_dy = .5f * (left_grad[2].y + right_grad[2].y);
  F32 dv_dz = .5f * (left_grad[2].z + right_grad[2].z);
  F32 dw_dx = .5f * (left_grad[3].x + right_grad[3].x);
  F32 dw_dy = .5f * (left_grad[3].y + right_grad[3].y);
  F32 dw_dz = .5f * (left_grad[3].z + right_grad[3].z);

  F32 div_u = du_dx + dv_dy + dw_dz;

  // NOTE(cmat): Newtonian stress tensor (Stokes' hypothesis, lambda = -2/3 mu).
  F32 tau_xx = 2.f * mu * du_dx - (2.f / 3.f) * mu * div_u;
  F32 tau_yy = 2.f * mu * dv_dy - (2.f / 3.f) * mu * div_u;
  F32 tau_zz = 2.f * mu * dw_dz - (2.f / 3.f) * mu * div_u;
  F32 tau_xy = mu * (du_dy + dv_dx);
  F32 tau_xz = mu * (du_dz + dw_dx);
  F32 tau_yz = mu * (dv_dz + dw_dy);

  V3F tau_n = v3f(
    tau_xx * normal.x + tau_xy * normal.y + tau_xz * normal.z,
    tau_xy * normal.x + tau_yy * normal.y + tau_yz * normal.z,
    tau_xz * normal.x + tau_yz * normal.y + tau_zz * normal.z
  );

  // NOTE(cmat): Temperature gradient via ideal gas law T = p / (rho * R),
  // - computed per-side (nonlinear) then averaged, since grad(T) itself
  // - isn't stored directly.
  F32 rho_l = left_primitive.x1;
  F32 rho_r = right_primitive.x1;
  F32 p_l   = left_primitive.x5;
  F32 p_r   = right_primitive.x5;

  V3F grad_T_l = v3f_mul(1.f / (rho_l * rho_l * gas_constant), v3f_sub(v3f_mul(rho_l, left_grad[4]),  v3f_mul(p_l, left_grad[0])));
  V3F grad_T_r = v3f_mul(1.f / (rho_r * rho_r * gas_constant), v3f_sub(v3f_mul(rho_r, right_grad[4]), v3f_mul(p_r, right_grad[0])));
  V3F grad_T_face = v3f_mul(.5f, v3f_add(grad_T_l, grad_T_r));

  // NOTE(cmat): Fourier heat flux q = -k * grad(T). Energy viscous flux
  // - across the face is (tau . V) . n - q . n = (tau . V) . n + k * (grad(T) . n).
  F32 heat_term = k_thermal * v3f_dot(grad_T_face, normal);
  F32 work_term = v3f_dot(tau_n, face_vel);

  V5F viscous_state = v5f(0.f, tau_n.x, tau_n.y, tau_n.z, work_term + heat_term);

  // NOTE(cmat): Diffusive (viscous) stability limit. Unlike the convective
  // - spectral radius, this scales as ~1/dx^2 rather than ~1/dx, so it must
  // - be accumulated separately and NOT folded into flux.lambda_max.
  // - Standard form (Blazek): lambda_visc = (mu/rho) * max(4/3, gamma/Pr) * area^2 / volume.
  F32 rho_face          = .5f * (rho_l + rho_r);
  F32 visc_coeff        = f32_max(4.f / 3.f, gamma / prandtl);
  F32 lambda_visc_face  = (mu / rho_face) * visc_coeff * (area * area) / volume;

  FL_Flux flux      = { };
  flux.state        = viscous_state;
  flux.lambda_max   = 0.f;              // NOTE(cmat): no convective wave speed here.
  flux.lambda_visc  = lambda_visc_face; // NOTE(cmat): diffusive spectral radius contribution for this face.
  return flux;
}

function void fl_solver_compute_residual_range(FL_Solver_Euler *euler, FL_State *state, FL_State *residual, FL_Gradient_State *grad, Range1_U64 range, B32 compute_time_step, F64 *cell_time_step) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell        = range.min + it_range;
    UG_Cell_Faces *faces          = &mesh->cells.faces[it_cell];
    V5F            cell_residual  = v5f(0, 0, 0, 0, 0);
    V5F left_primitive = {
      .x1 = state->rho[it_cell],
      .x2 = state->rho_v1[it_cell] / state->rho[it_cell],
      .x3 = state->rho_v2[it_cell] / state->rho[it_cell],
      .x4 = state->rho_v3[it_cell] / state->rho[it_cell],
      .x5 = fl_state_get_pressure(state, it_cell),
    };
    // NOTE(cmat): Primitive gradients. We're already storing these as primitive values.
    V3F left_grad[5] = { };
    for Iter_Index(it_state, 5) {
      left_grad[it_state] = v3f(grad->states[it_state].grad_x[it_cell], grad->states[it_state].grad_y[it_cell], grad->states[it_state].grad_z[it_cell]);
    }
    F64 spectral_sum      = 0.f; // NOTE(cmat): convective (inviscid) spectral radius accumulator.
    F64 spectral_sum_visc = 0.f; // NOTE(cmat): diffusive (viscous) spectral radius accumulator.
    for Iter_Index(it_face, 4) {
      U32 adjacent     = faces->adjacent[it_face];
      V3F normal       = v3f(faces->normal_x[it_face], faces->normal_y[it_face], faces->normal_z[it_face]);
      F32 area         = faces->area[it_face];
      V3F face_center  = v3f(faces->center_x[it_face], faces->center_y[it_face], faces->center_z[it_face]);
      // NOTE(cmat): Reconstruct left primitive state.
      F32 left_face_primitive[5] = { };
      V3F center_delta = v3f_sub(face_center, mesh->cells.center[it_cell]);
      for Iter_Index(it_state, 5) {
        F32 limiter                   = euler->limiter.states[it_state][it_cell];
        left_face_primitive[it_state] = left_primitive.dat[it_state] + limiter * v3f_dot(left_grad[it_state], center_delta);
      }
      // NOTE(cmat): Convert left primitive state to conservative state.
      V3F left_momentum = v3f_mul(left_face_primitive[0], v3f(left_face_primitive[1], left_face_primitive[2], left_face_primitive[3]));
      V5F left_state    = v5f(left_face_primitive[0], left_momentum.x, left_momentum.y, left_momentum.z, fl_state_energy_from_pressure(state, left_face_primitive[0], left_momentum, left_face_primitive[4]));
      // NOTE(cmat): Construct right state.
      // NOTE(cmat): If the right cell is not a ghost state, it has a gradient.
      V5F right_state     = { };
      V5F right_primitive = { }; // NOTE(cmat): cell-center primitive (unreconstructed) state, needed by the viscous flux.
      V3F right_grad[5]   = { };
      if (adjacent < (mesh->cells.len + mesh->halos.len)) {
        right_primitive = (V5F){
          .x1 = state->rho[adjacent],
          .x2 = state->rho_v1[adjacent] / state->rho[adjacent],
          .x3 = state->rho_v2[adjacent] / state->rho[adjacent],
          .x4 = state->rho_v3[adjacent] / state->rho[adjacent],
          .x5 = fl_state_get_pressure(state, adjacent),
        };
        for Iter_Index(it_state, 5) {
          right_grad[it_state] = v3f(grad->states[it_state].grad_x[adjacent], grad->states[it_state].grad_y[adjacent], grad->states[it_state].grad_z[adjacent]);
        }
        // NOTE(cmat): Construct right primitive state with gradient.
        F32 right_face_primitive[5] = { };
        V3F center_delta_r = v3f_sub(face_center, mesh->cells.center[adjacent]);
        for Iter_Index(it_state, 5) {
          F32 limiter                    = euler->limiter.states[it_state][adjacent];
          right_face_primitive[it_state] = right_primitive.dat[it_state] + limiter * v3f_dot(right_grad[it_state], center_delta_r);
        }
        // NOTE(cmat): Convert right primitive state to conservative state.
        V3F right_momentum  = v3f_mul(right_face_primitive[0], v3f(right_face_primitive[1], right_face_primitive[2], right_face_primitive[3]));
        right_state         = v5f(right_face_primitive[0], right_momentum.x, right_momentum.y, right_momentum.z, fl_state_energy_from_pressure(state, right_face_primitive[0], right_momentum, right_face_primitive[4]));
      // NOTE(cmat): If the right cell is a ghost state, it has a gradient of zero. We fallback to first order,
      // - both for the inviscid reconstruction and for the viscous gradients (no diffusive correction across
      // - ghost faces unless the boundary condition supplies a wall-gradient here).
      } else {
        right_state     = fl_state_get(state, adjacent);
        right_primitive = (V5F){
          .x1 = right_state.x1,
          .x2 = right_state.x2 / right_state.x1,
          .x3 = right_state.x3 / right_state.x1,
          .x4 = right_state.x4 / right_state.x1,
          .x5 = fl_state_get_pressure_from_conservative(right_state, state->gamma),
        };
        // right_grad stays zero.
      }
      FL_Flux flux_inviscid = fl_flux_hllc(left_state, right_state, normal, state->gamma);
      FL_Flux flux_viscous  = fl_flux_viscous(left_primitive, left_grad, right_primitive, right_grad, normal, area, mesh->cells.volume[it_cell], euler->mu, euler->k_thermal, euler->gas_constant, state->gamma, euler->prandtl);
      V5F     net_flux      = v5f_sub(flux_inviscid.state, flux_viscous.state);
      cell_residual         = v5f_sub(cell_residual, v5f_mul(area, net_flux));
      spectral_sum         += (F64)area * (F64)flux_inviscid.lambda_max;
      spectral_sum_visc    += (F64)flux_viscous.lambda_visc;
    }
    F32 volume     = mesh->cells.volume[it_cell];
    F32 volume_rcp = 1.f / volume;
    residual->rho     [it_cell] = cell_residual.x1 * volume_rcp;
    residual->rho_v1  [it_cell] = cell_residual.x2 * volume_rcp;
    residual->rho_v2  [it_cell] = cell_residual.x3 * volume_rcp;
    residual->rho_v3  [it_cell] = cell_residual.x4 * volume_rcp;
    residual->energy  [it_cell] = cell_residual.x5 * volume_rcp;
    if (compute_time_step) {
      // NOTE(cmat): Combined convective + diffusive stability limit.
      cell_time_step[it_cell] = (F64)volume / (spectral_sum + spectral_sum_visc);
    }
  }
  lane_barrier();
  profiler_end_function();
}


