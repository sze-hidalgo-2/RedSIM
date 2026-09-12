#if 0
force_inline function FL_Flux fl_flux_hllc(V5F UL, V5F UR, V3F n, F32 gamma) {
  FL_Flux flux = { };

  // NOTE(cmat): Density.
  F32 rL = UL.x1;
  F32 rR = UR.x1;

  // NOTE(cmat): Mass-Flow.
  V3F mL = UL.x234;
  V3F mR = UR.x234;

  // NOTE(cmat): Velocity.
  V3F uL = v3f_div(mL, rL);
  V3F uR = v3f_div(mR, rR);

  // NOTE(cmat): Project to normal vector.
  F32 unL = v3f_dot(uL, n);
  F32 unR = v3f_dot(uR, n);

  // NOTE(cmat): Velocity magnitude squared.
  F32 qL  = v3f_len2(uL);
  F32 qR  = v3f_len2(uR);

  // NOTE(cmat): Pressure.
  F32 pL  = (gamma - 1.f) * (UL.x5 - .5f * rL * qL);
  F32 pR  = (gamma - 1.f) * (UR.x5 - .5f * rR * qR);

  // NOTE(cmat): Speed of sound.
  F32 aL  = f32_sqrt(gamma * pL / rL);
  F32 aR  = f32_sqrt(gamma * pR / rR);

  // NOTE(cmat): Left and right wave speeds.
  F32 SL = f32_min(unL - aL, unR - aR);
  F32 SR = f32_max(unL + aL, unR + aR);

  // NOTE(cmat): Contact wave speed.
  F32 SM = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR));
  SM    /= (rL * (SL - unL) - rR * (SR - unR));

  // NOTE(cmat): Physical flux for left state.
  V5F FL = {
    .x1 = rL * unL,
    .x2 = mL.x * unL + pL * n.x,
    .x3 = mL.y * unL + pL * n.y,
    .x4 = mL.z * unL + pL * n.z,
    .x5 = (UL.x5 + pL) * unL,
  };

  // NOTE(cmat): Physical flux for right state.
  V5F FR = {
    .x1 = rR * unR,
    .x2 = mR.x * unR + pR * n.x,
    .x3 = mR.y * unR + pR * n.y,
    .x4 = mR.z * unR + pR * n.z,
    .x5 = (UR.x5 + pR) * unR,
  };

  // NOTE(cmat): Select the correct region.
  if (0.f <= SL) {
    flux.state = FL;
  } else if (0.f >= SR) {
    flux.state = FR;
  } else if (SM  >= 0.f)  {
    F32 rs = rL * (SL - unL) / (SL - SM);
    V3F us = v3f_add(uL, v3f_mul((SM - unL), n));
    F32 Es = rs * (UL.x5 / rL + (SM - unL) * (SM + pL / (rL * (SL - unL))));

    V5F U_star = {
      .x1 = rs,
      .x2 = rs * us.x,
      .x3 = rs * us.y,
      .x4 = rs * us.z,
      .x5 = Es,
    };

    flux.state = v5f_add(FL, v5f_mul(SL, v5f_sub(U_star, UL)));

  } else {
    F32 rs = rR * (SR - unR) / (SR - SM);
    V3F us = v3f_add(uR, v3f_mul((SM - unR), n));
    F32 Es = rs * (UR.x5 / rR + (SM - unR) * (SM + pR / (rR * (SR - unR))));

    V5F U_star = {
      .x1 = rs,
      .x2 = rs * us.x,
      .x3 = rs * us.y,
      .x4 = rs * us.z,
      .x5 = Es,
    };

    flux.state = v5f_add(FR, v5f_mul(SR, v5f_sub(U_star, UR)));
  }

  // NOTE(cmat): Maximum signal speed for CFL computation.
  flux.lambda_max = f32_max(f32_abs(SL), f32_abs(SR));

  return flux;
}

#else

force_inline function FL_Flux fl_flux_hllc(V5F UL, V5F UR, V3F n, F32 gamma) {
  FL_Flux flux = { };

  // NOTE(cmat): Density.
  F32 rL = UL.x1;
  F32 rR = UR.x1;

  F32 rL_rcp = 1.f / rL;
  F32 rR_rcp = 1.f / rR;

  // NOTE(cmat): Mass-Flow.
  V3F mL = UL.x234;
  V3F mR = UR.x234;

  // NOTE(cmat): Velocity.
  V3F uL = v3f_mul(rL_rcp, mL);
  V3F uR = v3f_mul(rR_rcp, mR);

  // NOTE(cmat): Project to normal vector.
  F32 unL = v3f_dot(uL, n);
  F32 unR = v3f_dot(uR, n);

  // NOTE(cmat): Velocity magnitude squared.
  F32 qL  = v3f_len2(uL);
  F32 qR  = v3f_len2(uR);

  // NOTE(cmat): Pressure.
  F32 pL  = (gamma - 1.f) * (UL.x5 - .5f * rL * qL);
  F32 pR  = (gamma - 1.f) * (UR.x5 - .5f * rR * qR);

  // NOTE(cmat): Speed of sound.
  F32 aL  = f32_sqrt(gamma * pL * rL_rcp);
  F32 aR  = f32_sqrt(gamma * pR * rR_rcp);

  // NOTE(cmat): Left and right wave speeds.
  F32 SL = f32_min(unL - aL, unR - aR);
  F32 SR = f32_max(unL + aL, unR + aR);

  // NOTE(cmat): Contact wave speed.
  F32 SM = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR));
  SM    /= (rL * (SL - unL) - rR * (SR - unR));

  // NOTE(cmat): Physical flux for left state.
  V5F FL = {
    .x1 = rL * unL,
    .x2 = mL.x * unL + pL * n.x,
    .x3 = mL.y * unL + pL * n.y,
    .x4 = mL.z * unL + pL * n.z,
    .x5 = (UL.x5 + pL) * unL,
  };

  // NOTE(cmat): Physical flux for right state.
  V5F FR = {
    .x1 = rR * unR,
    .x2 = mR.x * unR + pR * n.x,
    .x3 = mR.y * unR + pR * n.y,
    .x4 = mR.z * unR + pR * n.z,
    .x5 = (UR.x5 + pR) * unR,
  };

  // NOTE(cmat): Select the correct region.
  if (0.f <= SL) {
    flux.state = FL;

  } else if (0.f >= SR) {
    flux.state = FR;

  } else if (SM  >= 0.f)  {
    F32 SL_minus_unL_rcp = 1.f / (SL - SM);
    F32 rs = rL * (SL - unL) * SL_minus_unL_rcp;
    V3F us = v3f_add(uL, v3f_mul((SM - unL), n));
    F32 Es = rs * (UL.x5 * rL_rcp + (SM - unL) * (SM + pL / (rL * (SL - unL))));

    V5F U_star = {
      .x1 = rs,
      .x2 = rs * us.x,
      .x3 = rs * us.y,
      .x4 = rs * us.z,
      .x5 = Es,
    };

    flux.state = v5f_add(FL, v5f_mul(SL, v5f_sub(U_star, UL)));

  } else {
    F32 SR_minus_unR_rcp = 1.f / (SR - SM);
    F32 rs = rR * (SR - unR) * SR_minus_unR_rcp;
    V3F us = v3f_add(uR, v3f_mul((SM - unR), n));
    F32 Es = rs * (UR.x5 * rR_rcp + (SM - unR) * (SM + pR / (rR * (SR - unR))));

    V5F U_star = {
      .x1 = rs,
      .x2 = rs * us.x,
      .x3 = rs * us.y,
      .x4 = rs * us.z,
      .x5 = Es,
    };

    flux.state = v5f_add(FR, v5f_mul(SR, v5f_sub(U_star, UR)));
  }

  // NOTE(cmat): Maximum signal speed for CFL computation.
  flux.lambda_max = f32_max(f32_abs(SL), f32_abs(SR));
  return flux;
}

#endif

force_inline function V3F fl_flux_grad_correct(V3F grad_avg, F32 phi_L, F32 phi_R, V3F e_hat, F32 dist_rcp) {
  F32 directional_avg   = v3f_dot(grad_avg, e_hat);
  F32 directional_exact = (phi_R - phi_L) * dist_rcp;
  V3F grad_corrected    = v3f_add(grad_avg, v3f_mul(directional_exact - directional_avg, e_hat));
  return grad_corrected;
}

force_inline function FL_Flux fl_flux_viscous_smagorinsky_LES(V5F left_primitive, V3F left_grad[5], V3F left_center,
                                                              V5F right_primitive, V3F right_grad[5], V3F right_center,
                                                              V3F normal, F32 area, F32 left_volume, F32 right_volume, FL_Material *material) {

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

  // NOTE(cmat): Smagorinsky SGS large-eddy viscosity.
  // - We compute the strain-rate tensor S_ij here.
#if 0
  F32 Sxx = du.x;
  F32 Syy = dv.y;
  F32 Szz = dw.z;
  F32 Sxy = .5f * (du.y + dv.x);
  F32 Sxz = .5f * (du.z + dw.x);
  F32 Syz = .5f * (dv.z + dw.y);
  F32 S_mag2 = 2.f * (Sxx*Sxx + Syy*Syy + Szz*Szz) + 4.f * (Sxy*Sxy + Sxz*Sxz + Syz*Syz);
  F32 S_mag  = f32_sqrt(S_mag2);

#else
  // NOTE(cmat): Smagorinsky SGS large-eddy viscosity.
  F32 Sxx = du.x;
  F32 Syy = dv.y;
  F32 Szz = dw.z;
  F32 Sxy = .5f * (du.y + dv.x);
  F32 Sxz = .5f * (du.z + dw.x);
  F32 Syz = .5f * (dv.z + dw.y);
  F32 S_mag2 = 2.f * (Sxx*Sxx + Syy*Syy + Szz*Szz) + 4.f * (Sxy*Sxy + Sxz*Sxz + Syz*Syz);

  // EDIT(cmat/jfnk): regularize sqrt() so its derivative stays finite as S_mag2 -> 0.
  // sqrt(x) has infinite slope at x = 0, which forward-difference JFNK will sample directly
  // in near-quiescent / near-uniform flow regions. Floor scales with local strain so it
  // doesn't distort the physical SGS viscosity anywhere the flow is actually straining.
  F32 S_mag2_floor = 1e-20f * (S_mag2 + 1.f);
  F32 S_mag        = f32_sqrt(S_mag2 + S_mag2_floor);
#endif

  // NOTE(cmat): We filter width from local cell volume.
  F32 rho_face   = .5f * (left_primitive.x1 + right_primitive.x1);
  F32 volume_avg = .5f * (left_volume + right_volume);
  F32 delta      = cbrtf(volume_avg);
  F32 mu_sgs     = rho_face * material->smagorinsky_cs2 * (delta * delta) * S_mag;

  F32 mu_eff = material->viscosity_mu + mu_sgs;

  F32 tau_xx = 2.f * mu_eff * du.x - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_yy = 2.f * mu_eff * dv.y - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_zz = 2.f * mu_eff * dw.z - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_xy = mu_eff * (du.y + dv.x);
  F32 tau_xz = mu_eff * (du.z + dw.x);
  F32 tau_yz = mu_eff * (dv.z + dw.y);
  V3F tau_normal = v3f (tau_xx * normal.x + tau_xy * normal.y + tau_xz * normal.z,
                        tau_xy * normal.x + tau_yy * normal.y + tau_yz * normal.z,
                        tau_xz * normal.x + tau_yz * normal.y + tau_zz * normal.z);

  F32 left_rho       = left_primitive.x1;
  F32 right_rho      = right_primitive.x1;
  F32 left_pressure  = left_primitive.x5;
  F32 right_pressure = right_primitive.x5;
  V3F grad_T_l       = v3f_mul(1.f / (left_rho * left_rho * material->gas_constant), v3f_sub(v3f_mul(left_rho, left_grad[4]),  v3f_mul(left_pressure, left_grad[0])));
  V3F grad_T_r       = v3f_mul(1.f / (right_rho * right_rho * material->gas_constant), v3f_sub(v3f_mul(right_rho, right_grad[4]), v3f_mul(right_pressure, right_grad[0])));
  F32 T_L            = left_pressure  / (left_rho  * material->gas_constant);
  F32 T_R            = right_pressure / (right_rho * material->gas_constant);
  V3F grad_T_avg     = v3f_mul(.5f, v3f_add(grad_T_l, grad_T_r));
  V3F grad_T_face    = fl_flux_grad_correct (grad_T_avg, T_L, T_R, e_hat, dist_rcp);

  // NOTE(cmat): Effective conductivity = laminar + turbulent (from sgs eddy).
  F32 k_eff         = material->thermal_conductivity + material->cp * mu_sgs / material->prandtl_turbulent;
  F32 heat_term     = k_eff * v3f_dot(grad_T_face, normal);
  F32 work_term     = v3f_dot(tau_normal, face_velocity);
  V5F viscous_state = v5f(0.f, tau_normal.x, tau_normal.y, tau_normal.z, work_term + heat_term);

  // NOTE(cmat): Stability limit uses mu_eff, since eddy viscosity also diffuses momentum.
  F32 lambda_visc_face = (mu_eff / rho_face) * material->visc_coeff * (area * area) / volume_avg;

  FL_Flux flux        = { };
  flux.state          = viscous_state;
  flux.lambda_viscous = lambda_visc_face;
  return flux;
}

force_inline function FL_Flux fl_flux_viscous_wale_LES(V5F left_primitive, V3F left_grad[5], V3F left_center,
                                                       V5F right_primitive, V3F right_grad[5], V3F right_center,
                                                       V3F normal, F32 area, F32 left_volume, F32 right_volume, FL_Material *material) {

  V3F center_delta = v3f_sub(right_center, left_center);
  F32 dist         = v3f_len(center_delta);
  F32 dist_rcp     = 1.f / dist;
  V3F e_hat        = v3f_mul(dist_rcp, center_delta);

  V3F left_velocity   = v3f(left_primitive.x2,  left_primitive.x3,  left_primitive.x4);
  V3F right_velocity  = v3f(right_primitive.x2, right_primitive.x3, right_primitive.x4);
  V3F face_velocity   = v3f_mul(.5f, v3f_add(left_velocity, right_velocity));

  // NOTE(cmat/wale): Face-averaged velocity gradient tensor, corrected with the exact
  // directional derivative along the face-normal direction (same treatment as Smagorinsky).
  V3F du_avg = v3f_mul              (.5f, v3f_add(left_grad[1], right_grad[1]));
  V3F dv_avg = v3f_mul              (.5f, v3f_add(left_grad[2], right_grad[2]));
  V3F dw_avg = v3f_mul              (.5f, v3f_add(left_grad[3], right_grad[3]));
  V3F du     = fl_flux_grad_correct (du_avg, left_velocity.x, right_velocity.x, e_hat, dist_rcp);
  V3F dv     = fl_flux_grad_correct (dv_avg, left_velocity.y, right_velocity.y, e_hat, dist_rcp);
  V3F dw     = fl_flux_grad_correct (dw_avg, left_velocity.z, right_velocity.z, e_hat, dist_rcp);

  F32 div_u = du.x + dv.y + dw.z;

  // NOTE(cmat/wale): Full velocity gradient tensor g_ij = d(u_i)/d(x_j).
  // Row i = velocity component, column j = spatial direction.
  F32 g11 = du.x, g12 = du.y, g13 = du.z;
  F32 g21 = dv.x, g22 = dv.y, g23 = dv.z;
  F32 g31 = dw.x, g32 = dw.y, g33 = dw.z;

  // NOTE(cmat/wale): g^2 = g * g, matrix product (not elementwise).
  F32 g2_11 = g11*g11 + g12*g21 + g13*g31;
  F32 g2_12 = g11*g12 + g12*g22 + g13*g32;
  F32 g2_13 = g11*g13 + g12*g23 + g13*g33;
  F32 g2_21 = g21*g11 + g22*g21 + g23*g31;
  F32 g2_22 = g21*g12 + g22*g22 + g23*g32;
  F32 g2_23 = g21*g13 + g22*g23 + g23*g33;
  F32 g2_31 = g31*g11 + g32*g21 + g33*g31;
  F32 g2_32 = g31*g12 + g32*g22 + g33*g32;
  F32 g2_33 = g31*g13 + g32*g23 + g33*g33;

  F32 trace_g2       = g2_11 + g2_22 + g2_33;
  F32 trace_g2_third = trace_g2 * (1.f / 3.f);

  // NOTE(cmat/wale): Traceless symmetric part of g^2 — this is S^d_ij.
  F32 Sd_xx = g2_11 - trace_g2_third;
  F32 Sd_yy = g2_22 - trace_g2_third;
  F32 Sd_zz = g2_33 - trace_g2_third;
  F32 Sd_xy = .5f * (g2_12 + g2_21);
  F32 Sd_xz = .5f * (g2_13 + g2_31);
  F32 Sd_yz = .5f * (g2_23 + g2_32);

  F32 SdijSdij = Sd_xx*Sd_xx + Sd_yy*Sd_yy + Sd_zz*Sd_zz
               + 2.f * (Sd_xy*Sd_xy + Sd_xz*Sd_xz + Sd_yz*Sd_yz);

  // NOTE(cmat/wale): Resolved strain-rate tensor S_ij (plain double-contraction
  // convention, i.e. S_ij*S_ij with no factor of 2 folded in — this is what
  // the WALE formula wants, unlike the Smagorinsky |S| convention elsewhere).
  F32 Sxx = du.x;
  F32 Syy = dv.y;
  F32 Szz = dw.z;
  F32 Sxy = .5f * (du.y + dv.x);
  F32 Sxz = .5f * (du.z + dw.x);
  F32 Syz = .5f * (dv.z + dw.y);
  F32 SijSij = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.f * (Sxy*Sxy + Sxz*Sxz + Syz*Syz);

  // NOTE(cmat/wale): x^1.5, x^2.5, x^1.25 via sqrt products instead of powf —
  // cheaper, and this is on the residual/JVP hot path (once per face, every
  // RK stage and every Newton/GMRES Jacobian-vector product).
  F32 SijSij_sqrt   = f32_sqrt(SijSij);
  F32 SdijSdij_sqrt = f32_sqrt(SdijSdij);
  F32 Sd_num = SdijSdij * SdijSdij_sqrt;                     // SdijSdij^1.5
  F32 S_den  = SijSij * SijSij * SijSij_sqrt;                // SijSij^2.5
  F32 Sd_den = SdijSdij * f32_sqrt(SdijSdij_sqrt);           // SdijSdij^1.25  (FIXED: was SdijSdij_sqrt * sqrt(SdijSdij_sqrt) = x^0.75)

  // NOTE(cmat/wale): Guards literal 0/0 in perfectly uniform flow. Unlike
  // Smagorinsky's sqrt(S_mag2) floor, this isn't masking a derivative
  // singularity — the WALE expression is already C1 down to zero — it's
  // purely there so JFNK's forward difference never divides by an exact zero.
  F32 wale_eps = 1e-24f;

  // NOTE(cmat): Filter width from local cell volume, and face density,
  // matching the Smagorinsky model's convention.
  F32 rho_face   = .5f * (left_primitive.x1 + right_primitive.x1);
  F32 volume_avg = .5f * (left_volume + right_volume);
  F32 delta      = cbrtf(volume_avg);

  F32 nu_t   = (material->wale_cw * material->wale_cw) * (delta * delta) * Sd_num / (S_den + Sd_den + wale_eps);
  F32 mu_sgs = rho_face * nu_t;
  F32 mu_eff = material->viscosity_mu + mu_sgs;

  F32 tau_xx = 2.f * mu_eff * du.x - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_yy = 2.f * mu_eff * dv.y - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_zz = 2.f * mu_eff * dw.z - (2.f / 3.f) * mu_eff * div_u;
  F32 tau_xy = mu_eff * (du.y + dv.x);
  F32 tau_xz = mu_eff * (du.z + dw.x);
  F32 tau_yz = mu_eff * (dv.z + dw.y);
  V3F tau_normal = v3f (tau_xx * normal.x + tau_xy * normal.y + tau_xz * normal.z,
                        tau_xy * normal.x + tau_yy * normal.y + tau_yz * normal.z,
                        tau_xz * normal.x + tau_yz * normal.y + tau_zz * normal.z);

  F32 left_rho       = left_primitive.x1;
  F32 right_rho      = right_primitive.x1;
  F32 left_pressure  = left_primitive.x5;
  F32 right_pressure = right_primitive.x5;
  V3F grad_T_l       = v3f_mul(1.f / (left_rho * left_rho * material->gas_constant), v3f_sub(v3f_mul(left_rho, left_grad[4]),  v3f_mul(left_pressure, left_grad[0])));
  V3F grad_T_r       = v3f_mul(1.f / (right_rho * right_rho * material->gas_constant), v3f_sub(v3f_mul(right_rho, right_grad[4]), v3f_mul(right_pressure, right_grad[0])));
  F32 T_L            = left_pressure  / (left_rho  * material->gas_constant);
  F32 T_R            = right_pressure / (right_rho * material->gas_constant);
  V3F grad_T_avg     = v3f_mul(.5f, v3f_add(grad_T_l, grad_T_r));
  V3F grad_T_face    = fl_flux_grad_correct (grad_T_avg, T_L, T_R, e_hat, dist_rcp);

  // NOTE(cmat): Effective conductivity = laminar + turbulent (from sgs eddy).
  F32 k_eff         = material->thermal_conductivity + material->cp * mu_sgs / material->prandtl_turbulent;
  F32 heat_term     = k_eff * v3f_dot(grad_T_face, normal);
  F32 work_term     = v3f_dot(tau_normal, face_velocity);
  V5F viscous_state = v5f(0.f, tau_normal.x, tau_normal.y, tau_normal.z, work_term + heat_term);

  // NOTE(cmat): Stability limit uses mu_eff, since eddy viscosity also diffuses momentum.
  F32 lambda_visc_face = (mu_eff / rho_face) * material->visc_coeff * (area * area) / volume_avg;

  FL_Flux flux        = { };
  flux.state          = viscous_state;
  flux.lambda_viscous = lambda_visc_face;
  return flux;
}
