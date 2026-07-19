// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

#include "thirdparty/spall.h"  // TODO(cmat): Get rid of this dependency.

#define FFC_IMPL
#include "thirdparty/ffc.h"

#include "core_common.c"
#include "core_math.c"
#include "core_arena.c"
#include "core_string.c"
#include "core_format.c"
#include "core_meta.c"
#include "core_thread.c"
#include "core_log.c"
#include "core_random.c"
#include "core_array.c"

#if BUILD_PROFILE
# include "core_profile.c"
#endif
