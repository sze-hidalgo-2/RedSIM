#!/bin/bash
echo "zoltan build started"

# NOTE(cmat): Exit on error.
set -eu

# NOTE(cmat): Set working directory to the build.sh folder.
cd "$(dirname "$0")"

compiler="mpicc"

include_dir="-IUtilities/shared/ -IUtilities/Timer -Idriver -Igraph -Ihsfc -Imatlab -Iorder -Iphg -Isimple -Iutil -Iall -Ich -Ifdriver -Iha -Iinclude -Imatrix -Ipar -Ircb -Itimer -Izz -Icoloring -Ifort -Ihier -Ilb -Iparams -Ireftree -Itpls"
source_files="
all/all_allo.c
coloring/coloring.c coloring/color_test.c
coloring/bucket.c coloring/g2l_hash.c
graph/graph.c ha/divide_machine.c
ha/get_processor_name.c ha/ha_ovis.c
hier/hier.c hier/hier_free_struct.c
hsfc/hsfc_box_assign.c hsfc/hsfc.c
hsfc/hsfc_hilbert.c
hsfc/hsfc_point_assign.c lb/lb_balance.c
lb/lb_box_assign.c lb/lb_copy.c
lb/lb_eval.c lb/lb_free.c
lb/lb_init.c lb/lb_invert.c
lb/lb_migrate.c lb/lb_part2proc.c
lb/lb_point_assign.c lb/lb_remap.c
lb/lb_set_fn.c lb/lb_set_method.c
lb/lb_set_part_sizes.c
matrix/matrix_build.c
matrix/matrix_distribute.c
matrix/matrix_operations.c
matrix/matrix_sym.c matrix/matrix_utils.c
order/order.c order/order_struct.c
order/order_tools.c order/hsfcOrder.c
order/perm.c par/par_average.c
par/par_bisect.c par/par_median.c
par/par_median_randomized.c
par/par_stats.c par/par_sync.c
par/par_tflops_special.c
params/assign_param_vals.c
params/bind_param.c params/check_param.c
params/free_params.c params/key_params.c
params/print_params.c params/set_param.c
tpls/build_graph.c tpls/postprocessing.c
tpls/preprocessing.c tpls/scatter_graph.c
tpls/third_library.c tpls/verify_graph.c
tpls/parmetis_interface.c phg/phg_build.c
phg/phg_build_calls.c phg/phg.c
phg/phg_lookup.c phg/phg_verbose.c
phg/phg_coarse.c phg/phg_comm.c
phg/phg_distrib.c phg/phg_gather.c
phg/phg_hypergraph.c phg/phg_match.c
phg/phg_order.c phg/phg_parkway.c
phg/phg_patoh.c phg/phg_plot.c
phg/phg_rdivide.c phg/phg_refinement.c
phg/phg_scale.c phg/phg_serialpartition.c
phg/phg_util.c phg/phg_tree.c
phg/phg_Vcycle.c rcb/box_assign.c
rcb/create_proc_list.c rcb/inertial1d.c
rcb/inertial2d.c rcb/inertial3d.c
rcb/point_assign.c rcb/rcb_box.c
rcb/rcb.c rcb/rcb_util.c
rcb/rib.c rcb/rib_util.c
rcb/shared.c reftree/reftree_build.c
reftree/reftree_coarse_path.c
reftree/reftree_hash.c
reftree/reftree_part.c
simple/block.c
simple/cyclic.c simple/random.c
timer/timer_params.c
Utilities/Communication/comm_exchange_sizes.c
Utilities/Communication/comm_invert_map.c
Utilities/Communication/comm_do.c
Utilities/Communication/comm_do_reverse.c
Utilities/Communication/comm_info.c
Utilities/Communication/comm_create.c
Utilities/Communication/comm_resize.c
Utilities/Communication/comm_sort_ints.c
Utilities/Communication/comm_destroy.c
Utilities/Communication/comm_invert_plan.c
Utilities/Communication/comm_default.c
Utilities/Timer/zoltan_timer.c
Utilities/Timer/timer.c
Utilities/DDirectory/DD_Memory.c
Utilities/DDirectory/DD_Find.c
Utilities/DDirectory/DD_Destroy.c
Utilities/DDirectory/DD_Set_Neighbor_Hash_Fn3.c
Utilities/DDirectory/DD_Remove.c
Utilities/DDirectory/DD_Create.c
Utilities/DDirectory/DD_Update.c
Utilities/DDirectory/DD_Stats.c
Utilities/DDirectory/DD_Hash2.c
Utilities/DDirectory/DD_Print.c
Utilities/DDirectory/DD_Set_Neighbor_Hash_Fn2.c
Utilities/DDirectory/DD_Set_Hash_Fn.c
Utilities/DDirectory/DD_Set_Neighbor_Hash_Fn1.c
Utilities/Memory/mem.c
Utilities/shared/zoltan_align.c
Utilities/shared/zoltan_id.c zz/zz_coord.c
zz/zz_gen_files.c zz/zz_hash.c
zz/murmur3.c zz/zz_map.c
zz/zz_heap.c zz/zz_init.c
zz/zz_obj_list.c zz/zz_rand.c
zz/zz_set_fn.c zz/zz_sort.c
zz/zz_struct.c zz/zz_back_trace.c
zz/zz_util.c
"

compiler_flags="-O3 -fno-strict-aliasing"
linker_flags=""
$compiler -c $compiler_flags $include_dir $source_files $linker_flags
ar rcs libzoltan.a *.o
# mv libzoltan.a ../../../build/
rm *.o
