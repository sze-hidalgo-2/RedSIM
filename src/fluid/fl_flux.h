typedef struct FL_Flux {
  V5F state;
  union {
    F32 lambda_max;
    F32 lambda_viscous;
  };
} FL_Flux;

function FL_Flux fl_flux_hllc(V5F UL, V5F UR, V3F n, F32 gamma);
