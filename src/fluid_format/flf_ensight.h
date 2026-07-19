typedef struct FLF_Ensight_Export {
  Arena arena;
  Str08 case_file_path;
  Str08 data_folder_path;
  Str08 geo_file_path;

  U64   timestep_count;
  U64   cell_count;
} FLF_Ensight_Export;

function FLF_Ensight_Export flf_ensight_export_start          (Str08 folder_path, UG_Mesh *mesh);
function void               flf_ensight_export_end            (FLF_Ensight_Export *export);
function void               flf_ensight_export_cell_variable  (FLF_Ensight_Export *export, Str08 variable_name, U32 variable_buffer_len, F32 *variable_buffer_dat);
function void               flf_ensight_export_flow           (FLF_Ensight_Export *export, F32 time, FL_State *state);

