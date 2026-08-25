fl->smagorinsky_cs        = 0.17f;  // Lilly's value; try 0.1-0.12 if too dissipative
fl->prandtl_turbulent     = 0.9f;   // standard turbulent Prandtl number

force_inline function FL_Flux fl_flux_viscous(V5F left_primitive, V3F left_grad[5], V3F left_center,
                                              V5F right_primitive, V3F right_grad[5], V3F right_center,
                                              V3F normal, F32 area, F32 left_volume, F32 right_volume,
                                              F32 mu, F32 k_thermal, F32 gas_constant, F32 gamma,
                                              F32 prandtl, F32 smagorinsky_cs, F32 prandtl_turbulent) {
  V3F center_delta = v3f_sub(right_center, left_center);
  F32 dist         = v3f_len(center_delta);
  F32 dist_rcp     = 1.f / dist;
  V3F e_hat        = v3f_mul(dist_rcp, center_delta);

  V3F left_velocity   = v3f(left_primitive.x2,  left_primitive.x3,  left_primitive.x4);
  V3F right_velocity  = v3f(right_primitive.x2, right_primitive.x3, right_primitive.x4);
  V3F face_velocity   = v3f_mul(.5f, v3f_add(left_velocity, right_velocity));

  V3F du_avg = v3f_mul              (.5f, v3f_add(left_grad[1], right_grad[1]));
  V3F dv_avg = v3f_mul              (.5f, v3f_add(left_grad[2], right_grad[2]));
  V3F dw_avg = v3f_mul              (.5f, v3f_add(left_grad[3], right_grad[3]));
  V3F du     = fl_flux_grad_correct (du_avg, left_velocity.x, right_velocity.x, e_hat, dist_rcp);
  V3F dv     = fl_flux_grad_correct (dv_avg, left_velocity.y, right_velocity.y, e_hat, dist_rcp);
  V3F dw     = fl_flux_grad_correct (dw_avg, left_velocity.z, right_velocity.z, e_hat, dist_rcp);

  F32 div_u = du.x + dv.y + dw.z;

  // NOTE(cmat): Smagorinsky SGS eddy viscosity.
  // Strain-rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i); |S| = sqrt(2*S_ij*S_ij).
  F32 Sxx = du.x;
  F32 Syy = dv.y;
  F32 Szz = dw.z;
  F32 Sxy = .5f * (du.y + dv.x);
  F32 Sxz = .5f * (du.z + dw.x);
  F32 Syz = .5f * (dv.z + dw.y);
  F32 S_mag2 = 2.f * (Sxx*Sxx + Syy*Syy + Szz*Szz) + 4.f * (Sxy*Sxy + Sxz*Sxz + Syz*Syz);
  F32 S_mag  = f32_sqrt(S_mag2);

  // NOTE(cmat): Filter width from local cell volume (cube root -> isotropic cell length scale).
  F32 rho_face   = .5f * (left_primitive.x1 + right_primitive.x1);
  F32 volume_avg = .5f * (left_volume + right_volume);
  F32 delta      = f32_pow(volume_avg, 1.f / 3.f);
  F32 mu_sgs     = rho_face * (smagorinsky_cs * smagorinsky_cs) * (delta * delta) * S_mag;

  F32 mu_eff = mu + mu_sgs;

  F32 tau_xx = 2.f * mu_eff * du.x - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_yy = 2.f * mu_eff * dv.y - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_zz = 2.f * mu_eff * dw.z - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_xy = mu_eff * (du.y + dv.x);
  F32 tau_xz = mu_eff * (du.z + dw.x);
  F32 tau_yz = mu_eff * (dv.z + dw.y);
  V3F tau_normal = v3f( tau_xx * normal.x + tau_xy * normal.y + tau_xz * normal.z,
                        tau_xy * normal.x + tau_yy * normal.y + tau_yz * normal.z,
                        tau_xz * normal.x + tau_yz * normal.y + tau_zz * normal.z);

  F32 left_rho       = left_primitive.x1;
  F32 right_rho      = right_primitive.x1;
  F32 left_pressure  = left_primitive.x5;
  F32 right_pressure = right_primitive.x5;
  V3F grad_T_l    = v3f_mul(1.f / (left_rho * left_rho * gas_constant), v3f_sub(v3f_mul(left_rho, left_grad[4]),  v3f_mul(left_pressure, left_grad[0])));
  V3F grad_T_r    = v3f_mul(1.f / (right_rho * right_rho * gas_constant), v3f_sub(v3f_mul(right_rho, right_grad[4]), v3f_mul(right_pressure, right_grad[0])));
  F32 T_L         = left_pressure  / (left_rho  * gas_constant);
  F32 T_R         = right_pressure / (right_rho * gas_constant);
  V3F grad_T_avg  = v3f_mul(.5f, v3f_add(grad_T_l, grad_T_r));
  V3F grad_T_face = fl_flux_grad_correct (grad_T_avg, T_L, T_R, e_hat, dist_rcp);

  // NOTE(cmat): Effective conductivity = laminar + turbulent (via eddy diffusivity & cp).
  F32 cp        = (gamma / (gamma - 1.f)) * gas_constant;
  F32 k_eff     = k_thermal + cp * mu_sgs / prandtl_turbulent;
  F32 heat_term = k_eff * v3f_dot(grad_T_face, normal);
  F32 work_term = v3f_dot(tau_normal, face_velocity);
  V5F viscous_state = v5f(0.f, tau_normal.x, tau_normal.y, tau_normal.z, work_term + heat_term);

  // NOTE(cmat): Stability limit uses mu_eff, since eddy viscosity also diffuses momentum.
  F32 visc_coeff       = f32_max(4.f / 3.f, gamma / prandtl);
  F32 lambda_visc_face = (mu_eff / rho_face) * visc_coeff * (area * area) / volume_avg;

  FL_Flux flux        = { };
  flux.state          = viscous_state;
  flux.lambda_viscous = lambda_visc_face;
  return flux;
}

F32 right_volume = (adjacent < mesh->cells.len)
  ? mesh->cells.volume[adjacent]
  : mesh->cells.volume[it_cell];  // ghost/halo fallback: no real volume, reuse local cell's

FL_Flux flux_viscous = fl_flux_viscous(
    left_primitive, left_grad, mesh->cells.center[it_cell],
    right_primitive, right_grad, mesh->cells.center[adjacent],
    normal, area, mesh->cells.volume[it_cell], right_volume,
    state->viscosity_mu, state->thermal_conductivity, state->gas_constant,
    state->gamma, state->prandtl_number,
    state->smagorinsky_cs, state->prandtl_turbulent);

typedef struct FL_Reference_Scales {
  F32 rho_ref;  // kg/m^3
  F32 p_ref;    // Pa
  F32 a_ref;    // m/s, derived speed of sound
  F32 L_ref;    // m, mesh length scale (world units / normalized units)
} FL_Reference_Scales;

function FL_Reference_Scales fl_reference_scales_derive(F32 rho_ref, F32 p_ref, F32 gamma, F32 L_ref) {
  FL_Reference_Scales ref = { };
  ref.rho_ref = rho_ref;
  ref.p_ref   = p_ref;
  ref.L_ref   = L_ref;
  ref.a_ref   = f32_sqrt(gamma * p_ref / rho_ref);
  return ref;
}

// NOTE(cmat): Normalizes a dimensional farfield state (SI units) into solver units,
// - given a chosen reference state. density* is relative to rho_ref by convention
// - (doesn't have to be 1, e.g. for stratified/multi-density inflow).
function FL_Boundary_Farfield fl_boundary_farfield_normalize(FL_Reference_Scales ref, F32 density, V3F velocity, F32 pressure) {
  FL_Boundary_Farfield out = { };
  out.density  = density / ref.rho_ref;
  out.velocity = v3f_div(velocity, ref.a_ref);
  out.pressure = pressure / (ref.rho_ref * ref.a_ref * ref.a_ref);
  return out;
}

// NOTE(cmat): Derives all normalized FL_State constants from dimensional (SI) gas
// - properties and the chosen reference scales. This is the single source of truth —
// - do not hand-type normalized constants elsewhere.
function void fl_state_set_normalized_constants(FL_State *fl, FL_Reference_Scales ref,
                                                 F32 gamma, F32 mu_dimensional, F32 k_thermal_dimensional,
                                                 F32 prandtl_number, F32 smagorinsky_cs, F32 prandtl_turbulent) {
  fl->gamma                = gamma;
  fl->gas_constant         = 1.f / gamma; // R* s.t. p* = rho* * R* * T* holds with p_ref = rho_ref * a_ref^2 / gamma
  fl->viscosity_mu         = mu_dimensional / (ref.rho_ref * ref.a_ref * ref.L_ref);
  fl->prandtl_number       = prandtl_number;
  fl->thermal_conductivity = fl->viscosity_mu / ((gamma - 1.f) * prandtl_number);
  // NOTE(cmat): both dimensionless model constants; scale-invariant, passed through as-is.
  fl->smagorinsky_cs       = smagorinsky_cs;
  fl->prandtl_turbulent    = prandtl_turbulent;
}

FL_Reference_Scales ref = fl_reference_scales_derive(
    /* rho_ref */ 1.225f,
    /* p_ref   */ 101325.f,
    /* gamma   */ 1.4f,
    /* L_ref   */ 2600.f);

FL_Boundary_Farfield farfield = fl_boundary_farfield_normalize(
    ref, /* density */ 1.225f, /* velocity */ v3f(3.f, 4.f, 0.f), /* pressure */ 101325.f);

fl_state_set_normalized_constants(fl, ref,
    /* gamma             */ 1.4f,
    /* mu (Pa*s)         */ 1.85e-5f,
    /* k (W/m*K)         */ 0.026f,
    /* prandtl_number    */ 0.71f,
    /* smagorinsky_cs    */ 0.17f,
    /* prandtl_turbulent */ 0.9f);
