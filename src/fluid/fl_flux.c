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
  F32 Sxx = du.x;
  F32 Syy = dv.y;
  F32 Szz = dw.z;
  F32 Sxy = .5f * (du.y + dv.x);
  F32 Sxz = .5f * (du.z + dw.x);
  F32 Syz = .5f * (dv.z + dw.y);
  F32 S_mag2 = 2.f * (Sxx*Sxx + Syy*Syy + Szz*Szz) + 4.f * (Sxy*Sxy + Sxz*Sxz + Syz*Syz);
  F32 S_mag  = f32_sqrt(S_mag2);

  // NOTE(cmat): We filter width from local cell volume.
  F32 rho_face   = .5f * (left_primitive.x1 + right_primitive.x1);
  F32 volume_avg = .5f * (left_volume + right_volume);
  F32 delta      = cbrtf(volume_avg);
  F32 mu_sgs     = rho_face * (material->smagorinsky_cs * material->smagorinsky_cs) * (delta * delta) * S_mag;

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
  F32 cp            = (material->gamma / (material->gamma - 1.f)) * material->gas_constant;
  F32 k_eff         = material->thermal_conductivity + cp * mu_sgs / material->prandtl_turbulent;
  F32 heat_term     = k_eff * v3f_dot(grad_T_face, normal);
  F32 work_term     = v3f_dot(tau_normal, face_velocity);
  V5F viscous_state = v5f(0.f, tau_normal.x, tau_normal.y, tau_normal.z, work_term + heat_term);

  // NOTE(cmat): Stability limit uses mu_eff, since eddy viscosity also diffuses momentum.
  F32 visc_coeff       = f32_max(4.f / 3.f, material->gamma / material->prandtl_number);
  F32 lambda_visc_face = (mu_eff / rho_face) * visc_coeff * (area * area) / volume_avg;

  FL_Flux flux        = { };
  flux.state          = viscous_state;
  flux.lambda_viscous = lambda_visc_face;
  return flux;
}
