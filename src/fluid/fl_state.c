function void fl_state_init(FL_State *fl, FL_Material material, UG_Mesh *mesh, B32 store_ghost_halo, Arena *arena) {
  Zero_Fill(fl);

  U64 total_len = mesh->cells.len;
  if (store_ghost_halo) {
    total_len += mesh->halos.len + mesh->ghosts.len;
  }

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, 5 * total_len);
  }

  lane_broadcast_ptr(&total_dat, 0);

  fl->material     = material;

  fl->inner_len    = mesh->cells.len;
  fl->halo_len     = store_ghost_halo ? mesh->halos.len   : 0;
  fl->ghost_len    = store_ghost_halo ? mesh->ghosts.len  : 0;

  fl->rho          = total_dat + 0 * total_len;
  fl->rho_v1       = total_dat + 1 * total_len;
  fl->rho_v2       = total_dat + 2 * total_len;
  fl->rho_v3       = total_dat + 3 * total_len;
  fl->energy       = total_dat + 4 * total_len;

  lane_barrier();
}

function void fl_gradient_state_init(FL_Gradient_State *grad, UG_Mesh *mesh, B32 store_halo, Arena *arena) {
  Zero_Fill(grad);

  U64 total_len = mesh->cells.len;
  if (store_halo) {
    total_len += mesh->halos.len;
  }

  grad->inner_len    = mesh->cells.len;
  grad->halo_len     = store_halo ? mesh->halos.len   : 0;

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, 3 * 5 * total_len);
  }

  lane_broadcast_ptr(&total_dat, 0);

  for Iter_Index(it, 5) {
    grad->states[it].grad_x = total_dat; total_dat += total_len;
    grad->states[it].grad_y = total_dat; total_dat += total_len;
    grad->states[it].grad_z = total_dat; total_dat += total_len;
  }

  lane_barrier();
}

function void fl_limiter_state_init(FL_Limiter_State *limiter, UG_Mesh *mesh, B32 store_halo, Arena *arena) {
  Zero_Fill(limiter);

  U64 total_len = mesh->cells.len;
  if (store_halo) {
    total_len += mesh->halos.len;
  }

  limiter->inner_len    = mesh->cells.len;
  limiter->halo_len     = store_halo ? mesh->halos.len   : 0;

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, 5 * total_len);
  }

  lane_broadcast_ptr(&total_dat, 0);

  for Iter_Index(it, 5) {
    limiter->states[it] = total_dat;
    total_dat += total_len;
  }

  lane_barrier();
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

force_inline function F32 fl_state_get_pressure(FL_State *fl, U64 at) {
  F32 rho            = fl->rho[at];
  V3F rho_v          = v3f(fl->rho_v1[at], fl->rho_v2[at], fl->rho_v3[at]);
  F32 energy         = fl->energy[at];
  F32 kinetic_energy = v3f_len2(rho_v) / (2.f * rho);
  F32 pressure       = (fl->material.gamma - 1.f) * (energy - kinetic_energy);

  return pressure;
}

force_inline function void fl_state_set_pressure(FL_State *fl, F32 pressure, U64 at) {
  F32 rho            = fl->rho[at];
  V3F rho_v          = v3f(fl->rho_v1[at], fl->rho_v2[at], fl->rho_v3[at]);
  F32 kinetic_energy = v3f_len2(rho_v) / (2.f * rho);

  fl->energy[at]     = pressure / (fl->material.gamma - 1.f) + kinetic_energy;
}

force_inline function F32 fl_state_energy_from_pressure(FL_Material *mat, F32 density, V3F momentum, F32 pressure) {
  F32 kinetic_energy  = v3f_len2(momentum) / (2.f * density);
  F32 energy          = pressure / (mat->gamma - 1.f) + kinetic_energy;

  return energy;
}

force_inline function F32 fl_state_get_temperature(FL_State *fl, U64 at) {
  F32 rho            = fl->rho[at];
  F32 pressure       = fl_state_get_pressure(fl, at);
  F32 temperature    = pressure / (rho * fl->material.gas_constant);

  return temperature;
}

force_inline function F32 fl_gradient_q_criterion(FL_Gradient_State *grad, U32 at) {
  // NOTE(cmat): To compute the Q-criterion, we split the velocity gradient tensor
  // - into a symmetric and assymetric part.

  F32 ux = grad->v1.grad_x[at];
  F32 uy = grad->v1.grad_y[at];
  F32 uz = grad->v1.grad_z[at];

  F32 vx = grad->v2.grad_x[at];
  F32 vy = grad->v2.grad_y[at];
  F32 vz = grad->v2.grad_z[at];

  F32 wx = grad->v3.grad_x[at];
  F32 wy = grad->v3.grad_y[at];
  F32 wz = grad->v3.grad_z[at];

  F32 q_criterion = -0.5f * (ux * ux + vy * vy + wz * wz + 2.0f * (uy * vx + uz * wx + vz * wy));
  return q_criterion;
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

function void fl_state_set_inner_from_atmospheric(FL_State *fl, UG_Mesh *mesh, FL_Scale *scale, FL_Material *mat, FL_Boundary_Atmospheric *atm) {
  for Iter_Range(it, lane_range(fl->inner_len)) {
    V3F center     = mesh->cells.center[it];
    F32 z          = (center.z * scale->length) + scale->offset.z;

    F32 pressure   = fl_boundary_atmosphere_pressure (z, atm, mat->gas_constant_R);
    F32 density    = fl_boundary_atmosphere_density  (z, atm, mat->gas_constant_R);
    V3F velocity   = fl_boundary_atmosphere_velocity (z, atm);

    pressure       = fl_scale_normalize_pressure (scale, pressure);
    density        = fl_scale_normalize_density  (scale, density);
    velocity       = fl_scale_normalize_velocity (scale, velocity);

    fl->rho[it]    = density;
    fl->rho_v1[it] = density * velocity.x;
    fl->rho_v2[it] = density * velocity.y;
    fl->rho_v3[it] = density * velocity.z;

    fl_state_set_pressure(fl, pressure, it);
  }

  lane_barrier();
}


function void fl_material_init(FL_Material *material, F32 gamma, F32 viscosity, F32 prandtl_number, F32 gas_constant_R) {
  material->gamma                = gamma;
  material->gas_constant         = 1.f / gamma;
  material->gas_constant_R       = gas_constant_R;
  material->viscosity_mu         = viscosity;
  material->prandtl_number       = prandtl_number;
  material->thermal_conductivity = viscosity / ((gamma - 1.f) * prandtl_number);
  material->smagorinsky_cs       = 0.17f;
  material->prandtl_turbulent    = 0.9f;

  // NOTE(cmat): Precomputed values for smagorinsky LES.
  material->cp                   = (material->gamma / (material->gamma - 1.f)) * material->gas_constant;
  material->visc_coeff           = f32_max(4.f / 3.f, material->gamma / material->prandtl_number);
  material->smagorinsky_cs2      = material->smagorinsky_cs * material->smagorinsky_cs;
}

function void fl_scale_init(FL_Scale *scale, UG_Mesh *mesh, F32 density, F32 pressure, F32 gamma) {
  Zero_Fill(scale);

  scale->length       = mesh->grid.scale;
  scale->offset       = mesh->grid.offset;
  scale->density      = density;
  scale->pressure     = pressure;
  scale->sound_speed  = f32_sqrt((gamma * scale->pressure) / scale->density);
}

function void fl_scale_normalize_farfield(FL_Scale *scale, FL_Boundary_Farfield *farfield) {
  farfield->density  = farfield->density / scale->density;
  farfield->velocity = v3f_div(farfield->velocity, scale->sound_speed);
  farfield->pressure = farfield->pressure / (scale->density * scale->sound_speed * scale->sound_speed);
}

function void fl_scale_normalize_material(FL_Scale *scale, FL_Material *material) {
  F32 gamma            = material->gamma;
  F32 viscosity_scaled = material->viscosity_mu / (scale->density * scale->sound_speed * scale->length);
  F32 prandtl_number   = material->prandtl_number;
  F32 gas_constant_R   = material->gas_constant_R;
  fl_material_init(material, gamma, viscosity_scaled, prandtl_number, gas_constant_R);
}

function F32 fl_scale_normalize_time(FL_Scale *scale, F32 time) {
  F32 result = time * (scale->sound_speed / scale->length);
  return result;
}

function F32 fl_scale_normalize_density(FL_Scale *scale, F32 density) {
  F32 result = density / scale->density;
  return result;
}

function V3F fl_scale_normalize_velocity(FL_Scale *scale, V3F velocity) {
  V3F result = v3f_div(velocity, scale->sound_speed);
  return result;
}

function F32 fl_scale_normalize_pressure(FL_Scale *scale, F32 pressure) {
  F32 result = pressure / (scale->density * scale->sound_speed * scale->sound_speed);
  return result;
}

function F32 fl_scale_denormalize_density(FL_Scale *scale, F32 density) {
  F32 result = density * scale->density;
  return result;
}

function F32 fl_scale_denormalize_pressure(FL_Scale *scale, F32 pressure) {
  F32 result = pressure * scale->density * scale->sound_speed * scale->sound_speed;
  return result;
}

function F32 fl_scale_denormalize_energy(FL_Scale *scale, F32 energy) {
  F32 result = energy * scale->density * scale->sound_speed * scale->sound_speed;
  return result;
}

function F32 fl_scale_denormalize_velocity(FL_Scale *scale, F32 velocity) {
  F32 result = velocity * scale->sound_speed;
  return result;
}

function F32 fl_scale_denormalize_temperature(FL_Scale *scale, F32 temperature) {
  F32 result = temperature * scale->pressure / scale->density;
  return result;
}

function F32 fl_scale_denormalize_q_criterion(FL_Scale *scale, F32 q) {
  F32 result = q * (scale->sound_speed / scale->length) * (scale->sound_speed / scale->length);
  return result;
}

function F32 fl_scale_denormalize_time(FL_Scale *scale, F32 time) {
  F32 result = time * (scale->length / scale->sound_speed);
  return result;
}

