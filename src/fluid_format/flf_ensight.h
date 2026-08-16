typedef struct FLF_Ensight_Export {
  Str08 case_file_path;
  Str08 data_folder_path;
  Str08 geo_file_path;

  U64   timestep_count;
  U64  *part_cell_count;
} FLF_Ensight_Export;


