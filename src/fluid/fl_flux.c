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
