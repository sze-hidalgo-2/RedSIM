#include "alice/core/core_build.h"
#include "alice/core/core_build.c"

#if OS_LINUX
#include "alice/linux/linux_system.c"
#elif OS_MACOS
#include "alice/macos/macos_system.c"
#endif

#include "ipc/ipc_build.h"
#include "ipc/ipc_build.c"

#include "ugrid/ug_build.h"
#include "ugrid/ug_build.c"

#include "ugrid_format/ugf_build.h"
#include "ugrid_format/ugf_build.c"

#include "fluid/fl_build.h"
#include "fluid/fl_build.c"

#include "fluid/fl_scalar.c"

#include "fluid_format/flf_build.h"
#include "fluid_format/flf_build.c"

#include "rs_emission.h"
#include "rs_emission.c"

#include "rs_nox_ratios.h"
#include "rs_nox_ratios.c"

#include "rs_meteo.h"
#include "rs_meteo.c"

function void redsim_group_entry(void *user_data) {
  profiler_begin_function();
  log_zone_start("Thread Group Entry");

  Arena permanent_arena = { };
  arena_init(&permanent_arena);

  // NOTE(cmat): Load mesh on rank 0 and partition it (still multithreaded, just single rank).
  // NOTE(cmat): Once we've loaded the mesh on rank 0, distribute to other ranks and partition again
  // - on each rank for each thread group.
  UG_Mesh mesh = { };
  if (ipc_rank_index() == 0) {
    Arena partition_arena = { };
    arena_init(&partition_arena);

    UG_Mesh mesh_global = { };

    // NOTE(cmat): Load grid from file.
    Str08 su2_file = str08_from_cstring((char *)sys_context()->command_line.argv[1]);
    ugf_grid_init_from_su2(&mesh_global.grid, &partition_arena, su2_file);

    // NOTE(cmat): Compute mesh based on grid: adjacency + geometry.
    ug_mesh_init_from_grid(&mesh_global, &partition_arena);

    // NOTE(cmat): Partition mesh by rank count.
    UG_Partition partition = { };
    ug_partition_rcb(&partition, &partition_arena, &mesh_global, ipc_rank_count());

    UG_Mesh_Array mesh_array = { };
    ug_mesh_array_init(&mesh_array, &partition_arena, partition.blocks_len);

    // NOTE(cmat): Create sub-mesh for current rank
    // - Allocated on permanent, since we'll be using this one on this rank.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Create sub-mesh for each other rank.
    // - Allocated on partition storage, since we'll free after distributing.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(1, partition.blocks_len), &partition_arena);

    // NOTE(cmat): Compute cells to send between block for rank 0 mesh (permanent storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Compute cells to send between block for the other ranks (partition storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(1, partition.blocks_len), &partition_arena);

    // NOTE(cmat): Broadcast mesh array to all ranks.
    ug_mesh_ipc_distribute(&mesh_array);

    // NOTE(cmat): Assign our own mesh to rank 0.
    lane_barrier();
    mesh = mesh_array.dat[0];

    lane_barrier();
    arena_destroy(&partition_arena);
  } else {
    ug_mesh_ipc_receive(&permanent_arena, &mesh, 0);
  }

  // NOTE(cmat): Now, each rank has its own mesh.

  // NOTE(cmat): Compute gradients for each cell.
  ug_mesh_compute_cells_gradient(&mesh, &permanent_arena);

  // NOTE(cmat): Reoder cells by groups: Interior or boundary. [ interior cells | boundary cells ]
  // - This allows us to compute interior cells while waiting for halo cells to be distributed,
  // - needed only by boundary cells.
  ug_mesh_reorder_by_groups(&mesh);

  // NOTE(cmat): Reorder every group in the mesh to improve cache locality.
  ug_mesh_optimize_reorder(&mesh, mesh.groups.cells_interior);
  ug_mesh_optimize_reorder(&mesh, mesh.groups.cells_boundary);

  ug_mesh_spatial_grid(&permanent_arena, &mesh, 100);

  // NOTE(cmat): Simulation window - unchanged from before: 2014-11-06 Thursday 00:00 UTC
  // - through 2014-11-30 Sunday 23:00 UTC (600 hours / 25 days). The met station CSVs below
  // - cover the full year 2014, not just this window, so lookups index by calendar date/hour
  // - rather than by row order.
  U32 sim_start_year  = 2014;
  U32 sim_start_month = 11;
  U32 sim_start_day   = 6;
  U32 sim_start_hour  = 0;

  // NOTE(cmat): Domain location - Madrid. Used only for the solar-position calculation below;
  // - there's no measured sun-angle/zenith data, so it has to be computed astronomically.
  F64 madrid_lat_deg = 40.4168;
  F64 madrid_lon_deg = -3.7038;

  // NOTE(cmat): Load the full-year meteorological station CSVs (radiation, temperature, wind).
  // - Loads on lane 0 and broadcasts internally, so this is safe to call from every lane.
  RS_Meteo_Radiation_Table   meteo_radiation   = rs_meteo_radiation_table_load  (&permanent_arena, str08_lit("madrid/3129_GLOBAL_RADIATION-2014.csv"));
  RS_Meteo_Temperature_Table meteo_temperature = rs_meteo_temperature_table_load(&permanent_arena, str08_lit("madrid/3195_TEMPERATURE_2014.csv"));
  RS_Meteo_Wind_Table        meteo_wind        = rs_meteo_wind_table_load       (&permanent_arena, str08_lit("madrid/3195_WIND-2014.csv"));

  // NOTE(cmat): Initial (t=0) boundary-condition state, sourced from the tables/solar-geometry
  // - above - re-derived every step in the main loop below as simulated time advances.
  RS_Sim_Time       sim_time_0 = rs_sim_time_from_elapsed(sim_start_year, sim_start_month, sim_start_day, sim_start_hour, 0.0);
  RS_Solar_Position sun_0      = rs_solar_position(madrid_lat_deg, madrid_lon_deg, sim_time_0);

  F64 radiation_0_w_m2 = rs_meteo_radiation_lookup  (&meteo_radiation,   sim_time_0.year, sim_time_0.month, sim_time_0.day, sim_time_0.hour);
  F64 temperature_0_c  = rs_meteo_temperature_lookup(&meteo_temperature, sim_time_0.year, sim_time_0.month, sim_time_0.day, sim_time_0.hour);
  F64 wind_dir_0_deg = 0.0, wind_speed_0_m_s = 0.0;
  rs_meteo_wind_lookup(&meteo_wind, sim_time_0.year, sim_time_0.month, sim_time_0.day, sim_time_0.hour, &wind_dir_0_deg, &wind_speed_0_m_s);

  FL_Solver_Euler solver    = {};
  FL_Boundary_Map boundary  = {};

  FL_Boundary_Atmospheric atm = {
    .temperature_ground = (F32)(temperature_0_c + 273.15), // NOTE(cmat): station T (degC, from the tenths-of-degC T00..T23 columns) -> Kelvin.
    .pressure_ground    = 94000.f,  // ~940 hPa station pressure at Madrid's ~667 m elevation
                                     // (NOT sea-level 101325 Pa — Madrid sits high enough that this matters)
                                     // NOTE(cmat): no station-pressure CSV was provided, so this stays a fixed assumption.
    .gravity            = 9.81f,
    .lapse_rate         = 0.0065f,  // standard tropospheric lapse rate, fine for a shallow domain
    .wind_angle         = (F32)rs_meteo_wind_angle_math_rad(wind_dir_0_deg), // NOTE(cmat): station DIR_hh (tens-of-deg, direction wind comes FROM) -> math angle (rad, direction wind blows TOWARD).
    .wind_d             = 0.f,
    .wind_z0            = 0.5f,    // open/low-vegetation terrain — bump toward 0.5-1.0 if this is a dense urban domain
    .wind_z_ref         = 25.f,    // NOTE(cmat): station anemometer height - 25 m, as given.
    .wind_u_ref         = (F32)wind_speed_0_m_s, // NOTE(cmat): station speed_hh (tenths of m/s) -> m/s.
    .wind_z_cap         = 250.f,
  };

  FL_Boundary_Radiation_Wall wall = {
    .solar_irradiance     = (F32)radiation_0_w_m2, // NOTE(cmat): station RGLOhh - already W/m^2 (global/horizontal irradiance), no conversion needed.
    .gamma_coeff          = 0.85f,    // concrete/stone emissivity (this field doubles as ε in h_rad,
                                       // so it needs to be a real material emissivity, not a small
                                       // ground-heat-flux fraction — 0.35 was too low for that role)
    .albedo               = 0.20f,    // typical light concrete/stone urban albedo
    .sky_view_factor      = 0.4f,     // unchanged — depends on your street-canyon/domain geometry
    .diffuse_fraction     = 0.12f,    // NOTE(cmat): the dataset only reports total (global) irradiance, no direct/diffuse
                                       // split, so this stays a fixed physically-typical assumption.
    .cos_zenith           = (F32)((sun_0.cos_zenith > 0.001) ? sun_0.cos_zenith : 0.001), // NOTE(cmat): computed for Madrid at t=0 (no measured sun-angle data exists); clamped away from 0 to avoid a divide-by-zero in fl_boundary_radiation_heat_flux at night.
    .thermal_conductivity = 0.026f,   // unchanged (this is air's k; currently unused by the equilibrium formula anyway)
    .temperature_min      = 293.15f,  // ~20 °C floor on the *wall* equilibrium temperature - a generic safety clamp, not tied to this specific window
    .temperature_max      = 310.0f, // 343.15f,  // ~70 °C ceiling on the *wall* equilibrium temperature - same, a generic safety clamp
    .domain_center        = v3f_mul(.5f, v3f_add(mesh.bounds_global.min, mesh.bounds_global.max)).xy,
    .domain_radius        = v3f_mul(.5f, v3f_sub(mesh.bounds_global.max, mesh.bounds_global.min)).xy,
  };


  FL_Material material = {};
  fl_material_init(&material, 1.4f, 1.81e-5f, 0.71f, 287.05f);

  // FL_Boundary_Farfield farfield_old = farfield;
  FL_Material          material_old = material;

  FL_Scale ref_scale = {};
  // fl_scale_init(&ref_scale, &mesh, farfield.density, farfield.pressure, 1.4f);
  fl_scale_init(&ref_scale, &mesh, 1.2f, atm.pressure_ground, material.gamma);

  // NOTE(cmat): Normalize all simulated quantities.
  log_info("Normalizing Qualtities");

  Log_Zone_Scope("Reference Scale") {
    log_info("length:      %10.2e", ref_scale.length);
    log_info("density:     %10.2e", ref_scale.density);
    log_info("pressure:    %10.2e", ref_scale.density);
    log_info("sound speed: %10.2e", ref_scale.sound_speed);
  }

  // fl_scale_normalize_farfield(&ref_scale, &farfield);
  fl_scale_normalize_material(&ref_scale, &material);

#if 0
  Log_Zone_Scope("Farfield Boundary") {
    log_info("density:     %10.2e -> %10.2e",               farfield_old.density, farfield.density);
    log_info("velocity x:  %10.2e -> %10.2e",               farfield_old.velocity.x, farfield.velocity.x);
    log_info("velocity y:  %10.2e -> %10.2e",               farfield_old.velocity.y, farfield.velocity.y);
    log_info("velocity z:  %10.2e -> %10.2e",               farfield_old.velocity.z, farfield.velocity.z);
    log_info("pressure:    %10.2e -> %10.2e",               farfield_old.pressure, farfield.pressure);
  }
#endif

  Log_Zone_Scope("Material") {
    log_info("gamma:                %10.2e -> %10.2e" , material_old.gamma                , material.gamma);
    log_info("gas_constant:         %10.2e -> %10.2e" , material_old.gas_constant         , material.gas_constant);
    log_info("molecular viscosity:  %10.2e -> %10.2e" , material_old.viscosity_mu         , material.viscosity_mu);
    log_info("thermal conductivity: %10.2e -> %10.2e" , material_old.thermal_conductivity , material.thermal_conductivity);
    log_info("prandtl number:       %10.2e -> %10.2e" , material_old.prandtl_number,        material.prandtl_number);
    log_info("smagorinsky cs:       %10.2e -> %10.2e" , material_old.smagorinsky_cs,        material.smagorinsky_cs);
    log_info("prandtl turbulent:    %10.2e -> %10.2e" , material_old.prandtl_turbulent,     material.prandtl_turbulent);
  }

  log_info("time scaling: %10.2e", fl_scale_denormalize_time(&ref_scale, 1.f))

  // NOTE(cmat): Init boundary map.
  log_info("Initializing boundary");
  fl_boundary_map_init(&boundary, &permanent_arena, 3);
  if (lane_index() == 0) {
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary) { .type = FL_Boundary_Type_Radiation_Wall,  .radiation_wall = wall };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary) { .type = FL_Boundary_Type_Radiation_Wall,  .radiation_wall = wall };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary) { .type = FL_Boundary_Type_Atmospheric,     .atmospheric    = atm  };
  }

  // NOTE(cmat): Init solver.
  lane_barrier();

  V3F gravity = v3f(0, 0, -atm.gravity);
  gravity     = v3f_mul(ref_scale.length / (ref_scale.sound_speed * ref_scale.sound_speed), gravity);

  log_info("Initializing solver");
  fl_solver_euler_init(&solver, &boundary, ref_scale, material, gravity, &mesh, &permanent_arena);

  // NOTE(cmat): Initial condition.
  lane_barrier();
  log_info("Initializing flow with farfield");

  // fl_state_set_inner_from_farfield(&solver.flow_1, &farfield);
  fl_state_set_inner_from_atmospheric(&solver.flow_1, &mesh, &ref_scale, &material, &atm);

  // NOTE(cmat): Iterate and solve.
  lane_barrier();


  // NOTE(cmat): Scalar
#if 1

  FL_Scalar_Boundary_Map scalar_boundary = {};
  fl_scalar_boundary_map_init(&scalar_boundary, &permanent_arena, 3);

  if (lane_index() == 0) {
    *fl_scalar_boundary_map_by_index(&scalar_boundary, 0) = (FL_Scalar_Boundary) { .type = FL_Scalar_Boundary_Type_Zero_Gradient };
    *fl_scalar_boundary_map_by_index(&scalar_boundary, 1) = (FL_Scalar_Boundary) { .type = FL_Scalar_Boundary_Type_Zero_Gradient };
    *fl_scalar_boundary_map_by_index(&scalar_boundary, 2) = (FL_Scalar_Boundary) { .type = FL_Scalar_Boundary_Type_Background_Emission, .dirichlet_value = 0.0f, .background_emission_max_height = 75.f };
  }

  lane_barrier();

  FL_Scalar_Material scalar_material = {};
  fl_scalar_material_init(&scalar_material,
      fl_scale_normalize_diffusivity(&ref_scale, 0.001f),
      fl_scale_normalize_diffusivity(&ref_scale, 0.001f),
      fl_scale_normalize_diffusivity(&ref_scale, 0.001f));

  FL_Solver_Scalar scalar_solver = {};
  fl_solver_scalar_init(
    &scalar_solver,
    &scalar_boundary,
    scalar_material,
    &mesh,
    solver.flow_1.rho_v1,
    solver.flow_1.rho_v2,
    solver.flow_1.rho_v3,
    solver.flow_1.rho,
    &permanent_arena,
    ref_scale
  );

  // fl_solver_scalar_set_uniform(&scalar_solver, 0.001f);
  fl_solver_scalar_set_uniform(&scalar_solver, 0.0f);
  lane_barrier();

#endif

  // NOTE(cmat): Load traffic-emission line sources from CSV (x0,y0,x1,y1,value_1,value_2).
  // - 0.2f: release height above the road surface, matching the old single-point example.
  // - Loads on lane 0 and broadcasts internally, so this is safe to call from every lane.
  CSV_Emission_Line_Array emission_lines = csv_emission_lines_load(&permanent_arena, str08_lit("madrid/Traffic_Emissions_2014.csv"), 0.5f);

  // NOTE(cmat): Load the NOx time-of-day/day-of-week ratio table
  // - (Traffic_Emission_2014_NOX_ratios.csv). Row 0 must line up with the simulation's
  // - start date/hour (2014;11;6;Thursday;0), one row per hour with no gaps - so `time`
  // - (elapsed simulated seconds, accumulated in the loop below) indexes straight into it.
  RS_NOX_Ratio_Table nox_ratios = rs_nox_ratio_table_load(&permanent_arena, str08_lit("madrid/Traffic_Emission_2014_NOX_ratios.csv"));

  // NOTE(cmat): emission_scale = day_weight * emission_const. Split out so day_weight can
  // - be re-looked-up from nox_ratios every step as simulated time advances.
  F64 emission_const = (1000.0 * 1.9e+9) / (24.0 * 3600.0 * 25.0);

  F32 *scalar_emission_unit = 0; // NOTE(cmat): per-cell line contribution, NOT yet scaled by day_weight.
  F32 *scalar_emission      = 0; // NOTE(cmat): scalar_emission_unit * emission_scale(current day_weight).
  if (lane_index() == 0) {
    scalar_emission_unit = arena_push_count(&permanent_arena, F32, mesh.cells.len);
    scalar_emission      = arena_push_count(&permanent_arena, F32, mesh.cells.len);
  }
  lane_broadcast_ptr(&scalar_emission_unit, 0);
  lane_broadcast_ptr(&scalar_emission, 0);

  for Iter_Range(it, lane_range(mesh.cells.len)) {
    scalar_emission_unit[it] = 0.f;
    scalar_emission[it]      = 0.f;
  }
  lane_barrier();

  // NOTE(cmat): Trace every line through the mesh and spread its value_1 across the
  // - cells it crosses, weighted by (t_exit - t_enter) so each line's total contribution
  // - sums back to value_1. Done single-threaded on lane 0, same as the CSV load above -
  // - scalar_emission_unit is shared (broadcast above), so accumulating from multiple lanes
  // - here without atomics would race. NOT scaled by day_weight here - day_weight changes
  // - with simulated time, so that scaling is applied every step in the loop below instead.
  if (lane_index() == 0) {
    U32 lines_missed = 0;
    for Iter_Index(it_line, emission_lines.len) {
      CSV_Emission_Line *line = &emission_lines.dat[it_line];

      V3F a = v3f_mul(f32_div_safe(1.f, ref_scale.length), v3f_sub(v3f_sub(line->a, v3f(441918, 4474610, 0)), ref_scale.offset));
      V3F b = v3f_mul(f32_div_safe(1.f, ref_scale.length), v3f_sub(v3f_sub(line->b, v3f(441918, 4474610, 0)), ref_scale.offset));

      UG_Segment_Trace trace = ug_mesh_trace_segment(&permanent_arena, &mesh, a, b);
      if (trace.len == 0) { lines_missed += 1; continue; }

      for Iter_Index(it_hit, trace.len) {
        UG_Segment_Hit *hit    = &trace.dat[it_hit];
        F32             weight = hit->t_exit - hit->t_enter; // NOTE(cmat): fraction of the line inside this cell.
        scalar_emission_unit[hit->cell] += (F32)(line->value_1 * weight);
      }
    }
    log_info("emission lines: %llu loaded, %u missed the mesh entirely", emission_lines.len, lines_missed);
  }
  lane_barrier();

  // NOTE(cmat): day_weight for t=0 (simulation start, 2014-11-06 Thursday 00:00) - looked
  // - up again every step in the main loop below as `time` advances.
  F64 day_weight     = rs_nox_ratio_table_day_weight(&nox_ratios, 0.0);
  F64 emission_scale = day_weight * emission_const;

  if (lane_index() == 0) {
    for Iter_Index(it_cell, mesh.cells.len) {
      scalar_emission[it_cell] = (F32)(scalar_emission_unit[it_cell] * emission_scale);
    }
  }
  lane_barrier();

  fl_solver_scalar_source_set(&scalar_solver, scalar_emission);

  // NOTE(cmat): Export results.
  FLF_Ensight_Export export = { };
  flf_ensight_export_init(&export, str08_lit("karman"), &mesh, &permanent_arena);

  // NOTE(cmat): Compute current gradient + residual for variables using the gradient.
  fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 1);
  flf_ensight_export_flow(&export, &ref_scale, 0.0f, &solver.flow_1, &solver.gradient, solver.cell_time_step, scalar_solver.phi_1.phi);

#if 0
  F32 time = 0;
  for Iter_Index(it, 100) {
    F32 time_step = fl_solver_euler_solve_implicit(&solver, fl_scale_normalize_time(&ref_scale, 100.f));
    time += fl_scale_denormalize_time(&ref_scale, time_step);

    // NOTE(cmat): Compute current gradient + residual for variables using the gradient.
    fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 1);
    flf_ensight_export_flow(&export, &ref_scale, time, &solver.flow_1, &solver.gradient, solver.cell_time_step);
  }
#else
  // NOTE(cmat): Simulate the full requested window - 2014-11-06 Thursday 00:00 through
  // - 2014-11-30 Sunday 23:00 - which is exactly the `nox_ratios.len` hourly rows loaded
  // - above (600 hours = 25 days * 24h). Run in ~10s physical steps until that many hours
  // - of simulated time have elapsed, rather than a fixed iteration count.
  F64 sim_total_seconds = (F64)nox_ratios.len * 3600.0;

  F32 time              = 0;
  U64 last_exported_hour = 0; // NOTE(cmat): hour 0 was already exported above (t=0 initial state).

  log_info("simulating %llu hours (%.0f s) of traffic-emission time, exporting once per hour",
      nox_ratios.len, sim_total_seconds);

  while ((F64)time < sim_total_seconds) {
    // NOTE(cmat): Exchange halos, fill ghosts, compute gradients. Compute & discard residual for now.
    fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 0);
    fl_solver_scalar_solve_implicit(&scalar_solver, fl_scale_normalize_time(&ref_scale, 10.f));

    F32 time_step = fl_solver_euler_solve_implicit(&solver, fl_scale_normalize_time(&ref_scale, 10.f));
    if (time_step <= 0.f) {
      log_info("solver returned a non-positive time step (%f) - stopping early at t=%.0f s", time_step, (F64)time);
      break;
    }
    time += fl_scale_denormalize_time(&ref_scale, time_step);

    // NOTE(cmat): Re-look-up day_weight for the new simulated time (nox_ratios is
    // - hour-of-day/day-of-week dependent) and rescale the emission source accordingly.
    // - Done every physical step regardless of export cadence, so the forcing itself
    // - stays accurate even though we only *export* once per hour.
    day_weight     = rs_nox_ratio_table_day_weight(&nox_ratios, (F64)time);
    emission_scale = day_weight * emission_const;
    if (lane_index() == 0) {
      for Iter_Index(it_cell, mesh.cells.len) {
        scalar_emission[it_cell] = (F32)(scalar_emission_unit[it_cell] * emission_scale);
      }
    }
    lane_barrier();
    fl_solver_scalar_source_set(&scalar_solver, scalar_emission);

    // NOTE(cmat): Re-derive the atmospheric/radiation boundary conditions for the new
    // - simulated time: continuous solar geometry (Madrid) from an astronomical formula
    // - (no measured sun-angle data exists), and station-measured temperature/wind/radiation
    // - from the CSVs loaded above (hourly-stepped, same cadence as day_weight above).
    RS_Sim_Time       sim_now = rs_sim_time_from_elapsed(sim_start_year, sim_start_month, sim_start_day, sim_start_hour, (F64)time);
    RS_Solar_Position sun_now = rs_solar_position(madrid_lat_deg, madrid_lon_deg, sim_now);

    F64 radiation_w_m2 = rs_meteo_radiation_lookup  (&meteo_radiation,   sim_now.year, sim_now.month, sim_now.day, sim_now.hour);
    F64 temperature_c  = rs_meteo_temperature_lookup(&meteo_temperature, sim_now.year, sim_now.month, sim_now.day, sim_now.hour);
    F64 wind_dir_deg = 0.0, wind_speed_m_s = 0.0;
    rs_meteo_wind_lookup(&meteo_wind, sim_now.year, sim_now.month, sim_now.day, sim_now.hour, &wind_dir_deg, &wind_speed_m_s);

    atm.temperature_ground = (F32)(temperature_c + 273.15);
    atm.wind_angle         = (F32)rs_meteo_wind_angle_math_rad(wind_dir_deg);
    atm.wind_u_ref         = (F32)wind_speed_m_s;

    wall.solar_irradiance  = (F32)radiation_w_m2;
    wall.cos_zenith        = (F32)((sun_now.cos_zenith > 0.001) ? sun_now.cos_zenith : 0.001); // NOTE(cmat): clamp away from 0 - see the note on wall.cos_zenith's initial value above.

    if (lane_index() == 0) {
      *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary) { .type = FL_Boundary_Type_Radiation_Wall,  .radiation_wall = wall };
      *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary) { .type = FL_Boundary_Type_Radiation_Wall,  .radiation_wall = wall };
      *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary) { .type = FL_Boundary_Type_Atmospheric,     .atmospheric    = atm  };
    }
    lane_barrier();

    // NOTE(cmat): Only export once we've crossed into a new simulated hour - not every step.
    U64 current_hour = (U64)((F64)time / 3600.0);
    if (current_hour > last_exported_hour) {
      last_exported_hour = current_hour;

      // NOTE(cmat): Compute current gradient + residual for variables using the gradient.
      fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 1);
      flf_ensight_export_flow(&export, &ref_scale, time, &solver.flow_1, &solver.gradient, solver.cell_time_step, scalar_solver.phi_1.phi);

      log_info("exported hour %llu/%llu (t=%.0f s, day_weight=%.6e, T=%.1fK, GHI=%.0fW/m2, wind=%.1fm/s@%.0fdeg_from, cos_zenith=%.3f)",
          current_hour, nox_ratios.len, (F64)time, day_weight,
          atm.temperature_ground, wall.solar_irradiance, atm.wind_u_ref, wind_dir_deg, wall.cos_zenith);
    }
  }
#endif

  log_zone_end();
  profiler_end_function();
}

link_function void redsim_entry_point(void) {
  profiler_begin_function();
  log_info("RedSIM 1.0 | Build Hash: %S", Build_Hash_Str08);
  
  // NOTE(cmat): Check command syntax.
  if (sys_context()->command_line.argc != 2) {
    log_info("Command Syntax: ./redsim_cpu [mesh_file]");
  } else {
    log_sys_context();      // NOTE(cmat): Log system information.
    log_ipc_context();      // NOTE(cmat): Log IPC context.
    log_sys_numa_layout();  // NOTE(cmat): Log NUMA layout.

    U32 thread_count = 0;
    if (sys_numa_layout()->nodes_len > 1) {
      // NOTE(cmat): We are launching thread groups / numa node!
      U64 numa_index = ipc_rank_local_node_index();
      thread_count = sys_numa_layout()->nodes_dat[numa_index].cpus_len;
    } else {
      thread_count = sys_context()->cpu_logical_cores;
    }

    Thread_Group thread_group = { };
    log_info("Launching global thread group with %u threads", thread_count);

    thread_group_init(&thread_group, str08_lit("Sim_Group"), thread_count);

    U64 numa_index = 0;
    if (sys_numa_layout()->nodes_len > 1) {
      numa_index = ipc_rank_local_node_index();
    }

    thread_group_launch   (&thread_group, redsim_group_entry, numa_index, 0);
    thread_group_wait_all (&thread_group);
    thread_group_destroy  (&thread_group);
  }

  profiler_end_function();
}

link_function void sys_entry_point(void) {
  // NOTE(cmat): Initialize IPC communication first.
  ipc_init();

  // NOTE(cmat): Check if RedSIM was launched correctly.
  if (sys_numa_layout()->nodes_len > 1 && ipc_rank_local_node_count() != sys_numa_layout()->nodes_len) {
    log_fatal("Rank count per compute node does not match NUMA domain count: numa = %llu | local ranks = %llu",
              sys_numa_layout()->nodes_len, ipc_rank_local_node_count());
  }

  // NOTE(cmat): Bind main thread to appropriate NUMA node and cpu within that node.
  SYS_CPU bind_to_cpu = 0;
  if (sys_numa_layout()->nodes_len > 1) {
    U64 numa_index            = ipc_rank_local_node_index();
    SYS_NUMA_Node *numa_node  = &sys_numa_layout()->nodes_dat[numa_index];
    bind_to_cpu               = numa_node->cpus_dat[0];
  } else {
    bind_to_cpu = 0;
  }

  sys_thread_bind_to_cpu(bind_to_cpu);

  // NOTE(cmat): Initialize profiler, logger.
  if (ipc_rank_index() == 0) {
    // NOTE(cmat): We only profile for rank 0.
    // - Even though we only profile a single rank, we can still see
    // - the bottlenecks at collective communication calls.
    profiler_startup("spall_trace.spall");
    profiler_init_for_thread();

    // NOTE(cmat): We only generate logs for rank 0.
    // NOTE(cmat): We log both to stdout and files.
    logger_push_hook(logger_write_entry_standard_stream, logger_format_entry_detailed);
  }

  redsim_entry_point();

  // NOTE(cmat): Stop profiling.
  if (ipc_rank_index() == 0){ 
    profiler_quit_for_thread();
    profiler_shutdown();
  }

  // NOTE(cmat): Shutdown IPC.
  ipc_shutdown();
}
