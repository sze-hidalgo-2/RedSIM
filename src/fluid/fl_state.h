typedef struct FL_Material {
  F32   gamma;
  F32   gas_constant;
  F32   viscosity_mu;
  F32   thermal_conductivity;
  F32   prandtl_number;
  F32   smagorinsky_cs;
  F32   prandtl_turbulent;
} FL_Material;

typedef struct FL_State {
  FL_Material   material;

  U64           inner_len;
  U64           halo_len;
  U64           ghost_len;

  union {
    struct {
      F32  *rho;
      F32  *rho_v1;
      F32  *rho_v2;
      F32  *rho_v3;
      F32  *energy;
    };

    F32 *states[5];
  };
} FL_State;

typedef struct FL_Gradient {
  union {
    struct {
      F32  *grad_x;
      F32  *grad_y;
      F32  *grad_z;
    };

    F32 *grad_dat[3];
  };
} FL_Gradient;

typedef struct FL_Gradient_State {
  U64         inner_len;
  U64         halo_len;

  union {
    struct {
      FL_Gradient rho;
      FL_Gradient v1;
      FL_Gradient v2;
      FL_Gradient v3;
      FL_Gradient pressure;
    };

    FL_Gradient states[5];
  };

} FL_Gradient_State;

typedef struct FL_Limiter_State {
  U64         inner_len;
  U64         halo_len;

  union {
    struct {
      F32 *rho;
      F32 *v1;
      F32 *v2;
      F32 *v3;
      F32 *pressure;
    };

    F32 *states[5];
  };
} FL_Limiter_State;

typedef struct FL_Scale {
  F32 length;
  F32 density;
  F32 pressure;
  F32 sound_speed;
} FL_Scale;

function void fl_state_init                     (FL_State *fl, FL_Material material, UG_Mesh *mesh, B32 store_ghost_halo, Arena *arena);
function void fl_state_set                      (FL_State *fl, U64 at, V5F state);
function V5F  fl_state_get                      (FL_State *fl, U64 at);
function F32  fl_state_get_pressure             (FL_State *fl, U64 at);
function F32  fl_state_get_temperature          (FL_State *fl, U64 at);
function void fl_state_set_pressure             (FL_State *fl, F32 pressure, U64 at);
function void fl_state_set_inner_from_farfield  (FL_State *fl, FL_Boundary_Farfield *farfield);

