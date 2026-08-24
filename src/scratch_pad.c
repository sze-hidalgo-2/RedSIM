F32 L_ref   = /* characteristic length, e.g. domain bounding-box diagonal, or a fixed 1000m for a city */;
F32 rho_ref = farfield.density;
F32 U_ref   = v3f_len(farfield.velocity);   // or speed of sound if velocity is ~0
F32 p_ref   = rho_ref * U_ref * U_ref;
F32 t_ref   = L_ref / U_ref;

