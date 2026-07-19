typedef struct UG_Partition {
} UG_Partition;

function void ug_partition_rcb(UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Partitioning mesh: RCB");



  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}
