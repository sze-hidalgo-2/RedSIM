function void fl_state_init(FL_State *fl, UG_Mesh *mesh) {
  Zero_Fill(fl);
  arena_init(&fl->arena);

  U64  total_len   = mesh->cells.len + mesh->ghosts.len;
  F32 *total_dat   = arena_push_count(&fl->arena, F32, 5 * total_len);

  fl->gamma        = 1.4f;
  fl->gas_constant = 287.05f; // J / (kg * K)

  fl->inner_len    = mesh->cells.len;
  fl->ghost_len    = mesh->ghosts.len;

  fl->rho          = total_dat + 0 * total_len;
  fl->rho_v1       = total_dat + 1 * total_len;
  fl->rho_v2       = total_dat + 2 * total_len;
  fl->rho_v3       = total_dat + 3 * total_len;
  fl->energy       = total_dat + 4 * total_len;
}

force_inline function V5F fl_state_get(FL_State *fl, U64 at) {
  V5F     state = v5f(fl->rho[at], fl->rho_v1[at], fl->rho_v2[at], fl->rho_v3[at], fl->energy[at]);
  return  state;
}

force_inline function void fl_state_set(FL_State *fl, U64 at, V5F state) {
  fl->rho     [at] = state.x1;
  fl->rho_v1  [at] = state.x2;
  fl->rho_v2  [at] = state.x3;
  fl->rho_v3  [at] = state.x4;
  fl->energy  [at] = state.x5;
}

function F32 fl_state_get_pressure(FL_State *fl, U64 at) {
  F32 rho            = fl->rho[at];
  V3F rho_v          = v3f(fl->rho_v1[at], fl->rho_v2[at], fl->rho_v3[at]);
  F32 energy         = fl->energy[at];
  F32 kinetic_energy = v3f_len2(rho_v) / (2.f * rho);
  F32 pressure       = (fl->gamma - 1.f) * (energy - kinetic_energy);

  return pressure;
}

function void fl_state_set_pressure(FL_State *fl, F32 pressure, U64 at) {
  F32 rho            = fl->rho[at];
  V3F rho_v          = v3f(fl->rho_v1[at], fl->rho_v2[at], fl->rho_v3[at]);
  F32 kinetic_energy = v3f_len2(rho_v) / (2.f * rho);

  fl->energy[at]     = pressure / (fl->gamma - 1.f) + kinetic_energy;
}

function F32 fl_state_get_temperature(FL_State *fl, U64 at) {
  F32 rho            = fl->rho[at];
  F32 pressure       = fl_state_get_pressure(fl, at);
  F32 temperature    = pressure / (rho * fl->gas_constant);

  return temperature;
}

function void fl_setup_sod(FL_State *fl, UG_Mesh *mesh) {
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    V3F center     = mesh->cells.center[it];
    F32 pressure   = center.x < 0 ? 1.f : 0.100f;  // 1.0 bar   | 0.100 bar
    fl->rho[it]    = center.x < 0 ? 1.f : 0.125f;  // 1kg / m^3 | 0.125kg / m^3
    fl->rho_v1[it] = 0.f;
    fl->rho_v2[it] = 0.f;
    fl->rho_v3[it] = 0.f;

    fl_state_set_pressure(fl, pressure, it);
  }

  lane_barrier();
}

function void fl_state_set_inner_from_farfield(FL_State *fl, FL_Boundary_Farfield *farfield) {
  for Iter_Range(it, lane_range(fl->inner_len)) {
    fl->rho[it]    = farfield->density;
    fl->rho_v1[it] = farfield->density * farfield->velocity.x;
    fl->rho_v2[it] = farfield->density * farfield->velocity.y;
    fl->rho_v3[it] = farfield->density * farfield->velocity.z;

    fl_state_set_pressure(fl, farfield->pressure, it);
  }

  lane_barrier();
}
