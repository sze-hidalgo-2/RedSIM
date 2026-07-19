// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Caller Meta-Data

typedef struct {
  Str08 function_name;
  Str08 file_name;
  U32   line;
} Caller_Metadata;

#define Caller_Metadata_Get ((Caller_Metadata) { \
  .function_name = str08_lit(__FUNCTION__),      \
  .file_name     = str08_lit(__FILE__),          \
  .line          = __LINE__                      \
})

