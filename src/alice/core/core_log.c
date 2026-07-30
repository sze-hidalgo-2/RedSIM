// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Logging API

global Logger_State Logger = {
  .filter       = Logger_Filter_Build_Active,
  .zone_depth   = 0,
  .hook_count   = 0,
  .write_hooks  = { },
  .format_hooks = { },
  .mutex        = { },
};

function void logger_entry(Logger_Entry *entry) {
  thread_storage local_storage U08 entry_buffer[Logger_Max_Entry_Length] = {};

  // TODO(cmat): Make this more granular. May be we can enable loging per thread, etc.
  if (lane_index() == 0) {
    Mutex_Scope(&Logger.mutex) {
      // TODO(cmat): This is such a mess. Why are we checking this here again?
      If_Likely ((Logger.filter & logger_filter_flag_from_entry_type(entry->type))) {

        // NOTE(cmat): Zone-ing.
        if (entry->type == Logger_Entry_Zone_Start) {
            Logger.zone_depth += 1;
        } else if (entry->type == Logger_Entry_Zone_End) {
            Assert(Logger.zone_depth, "unmatched log_zone_start / log_zone_end calls");
            if (Logger.zone_depth) Logger.zone_depth -= 1;
        }

        // NOTE(cmat): Call all format/write hooks.
        for Iter_Index (it, Logger.hook_count) {
            Logger.format_hooks[it](entry, entry_buffer, Logger.zone_depth);
            if (*entry_buffer)
            Logger.write_hooks[it](entry->type, str08_from_cstring((char *)entry_buffer));
        }
      }
    }
  }
}

function void logger_set_filter(Logger_Filter_Flag filter) { 
  Mutex_Scope(&Logger.mutex) {
    Logger.filter = filter;
  }
}

function B32 logger_filter_type(Logger_Entry_Type type) {
  B32 result = 0;
  Mutex_Scope(&Logger.mutex) {
    result = (Logger.filter & logger_filter_flag_from_entry_type(type));
  }

  return result;
}

function void logger_push_hook(Logger_Write_Entry_Hook *write, Logger_Format_Entry_Hook *format) {
  Mutex_Scope(&Logger.mutex) {
    Assert(Logger.hook_count < Logger_Max_Hooks, "exceeded hook count");
    Logger.write_hooks[Logger.hook_count] = write;
    Logger.format_hooks[Logger.hook_count] = format;
    Logger.hook_count++;
  }
}

// NOTE(cmat): Default hooks.
// TODO(cmat): Cleanup code once we introduce better string formatting stuff.
// #--
// 

function void logger_format_entry_minimal(Logger_Entry *entry, U08 *entry_buffer, U32 zone_depth) {
  local_storage Str08 type_lookup[] = {
   str08_lit(""), // NOTE(cmat): Info.
   str08_lit(" [Debug]: "),
   str08_lit(" [Warning]: "),
   str08_lit(" [Error]: "),
   str08_lit(" [Fatal]: "),
   str08_lit(" [Zone_Start]: "),
   str08_lit(" [Zone_End]: "),
 };

  Str08 type_str = type_lookup[entry->type];  
  Assert_Compiler(Logger_Entry_Count == Stack_Array_Len(type_lookup));

  U32 buffer_at = 0;
  U32 zone_indent = u32_min(5, zone_depth);
  if (entry->type == Logger_Entry_Zone_Start && zone_indent) zone_indent -= 1;
  for (U32 index = 0; index < zone_indent; ++index) {
    entry_buffer[buffer_at++] = ' ';
    entry_buffer[buffer_at++] = ' ';
  }

  Assert(buffer_at <= Logger_Max_Entry_Length, "logger buffer overflow");

  if (entry->type == Logger_Entry_Zone_Start) {
   stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "# %s\n", entry->message);

  } else if (entry->type == Logger_Entry_Zone_End) {
    entry_buffer[0] = 0;
  } else if (entry->type == Logger_Entry_Debug) {
   stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "%.*s<%.*s:%d, %.*s>: %s\n",
                   (I32)type_str.len, type_str.txt,
                   (I32)entry->caller.file_name.len, entry->caller.file_name.txt, entry->caller.line, (I32)entry->caller.function_name.len, entry->caller.function_name.txt,
                   entry->message);
  } else {
   stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "%.*s%s\n",
                   (I32)type_str.len, type_str.txt, entry->message);
  }
}

function void logger_format_entry_detailed(Logger_Entry *entry, U08 *entry_buffer, U32 zone_depth) {
    local_storage Str08 type_lookup[] = {
     str08_lit(""), // NOTE(cmat): Info.
     str08_lit(" [Debug]"),
     str08_lit(" [Warning]"),
     str08_lit(" [Error]"),
     str08_lit(" [Fatal]"),
     str08_lit(" [Zone_Start]"),
     str08_lit(" [Zone_End]"),
   };
  
   Str08 type_str = type_lookup[entry->type];  
   Assert_Compiler(Logger_Entry_Count == Stack_Array_Len(type_lookup));

   U32 buffer_at = 0;
   U32 zone_indent = u32_min(5, zone_depth);
   if (entry->type == Logger_Entry_Zone_Start && zone_indent) zone_indent -= 1;
   for (U32 index = 0; index < zone_indent; ++index) {
     entry_buffer[buffer_at++] = '|';
     entry_buffer[buffer_at++] = ' ';
   }
   
   Assert(buffer_at <= Logger_Max_Entry_Length, "logger buffer overflow");
   
   if (entry->type == Logger_Entry_Zone_Start) {
     stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "# %s\n", entry->message);
   } else if (entry->type == Logger_Entry_Zone_End) {
     entry_buffer[0] = 0;
   } else if (entry->type == Logger_Entry_Debug) {
     stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "(%02d:%02d:%02d %s %02d/%02d/%02d)%.*s <%.*s:%d, %.*s>: %s\n",
                     entry->time.hours > 12 ? entry->time.hours - 12 : entry->time.hours, entry->time.minutes, entry->time.seconds,
                     entry->time.hours > 12 ? "PM" : "AM",
                     entry->time.day, entry->time.month, entry->time.year,
                     (I32)type_str.len, type_str.txt,
                     (I32)entry->caller.file_name.len, entry->caller.file_name.txt, entry->caller.line, (I32)entry->caller.function_name.len, entry->caller.function_name.txt,
                     entry->message);
   } else {
     stbsp_snprintf((char *)entry_buffer + buffer_at, Logger_Max_Entry_Length - buffer_at, "(%02d:%02d:%02d %s %02d/%02d/%02d)%.*s: %s\n",
                     entry->time.hours > 12 ? entry->time.hours - 12 : entry->time.hours, entry->time.minutes, entry->time.seconds,
                     entry->time.hours > 12 ? "PM" : "AM",
                     entry->time.day, entry->time.month, entry->time.year,
                     (I32)type_str.len, type_str.txt, entry->message);
   }
}
function void logger_write_entry_standard_stream(Logger_Entry_Type type, Str08 buffer) {
  switch (type) {
    case Logger_Entry_Error:
    case Logger_Entry_Fatal: {
        sys_stream_write(buffer, SYS_Stream_Standard_Error);
    } break;

    case Logger_Entry_Warning: {
        sys_stream_write(buffer, SYS_Stream_Standard_Error);
    } break;

    default: {
        sys_stream_write(buffer, SYS_Stream_Standard_Output);
    } break;
  }
}

function void log_message_ext(Logger_Entry_Type type, Caller_Metadata caller, char *format, ...) {
  if (logger_filter_type(type)) {
    Logger_Entry entry = { 
      .type   = type, 
      .time   = sys_local_time_utc(), 
      .caller = caller,
    }; 

    va_list args;
    va_start(args, format);
    entry.message[stbsp_vsnprintf((char *)entry.message, Logger_Max_Entry_Length, format, args)] = 0;
    va_end(args);

    logger_entry(&entry);
  }
}

// ------------------------------------------------------------
// #-- Common Utility Logging Functions

function void log_sys_context(void) {
  Log_Zone_Scope("System Context") {
    log_info("CPU: %S",              sys_context()->cpu_name);
    log_info("Logical Cores: %llu",  sys_context()->cpu_logical_cores);
    log_info("Page Size: %$$llu",    sys_context()->mmu_page_bytes);
    log_info("RAM Capacity: %$$llu", sys_context()->ram_capacity_bytes);
  }
}

function void log_sys_numa_layout(void) {
  SYS_NUMA_Layout *numa = sys_numa_layout();

  Log_Zone_Scope("NUMA Layout - Nodes: %llu", numa->nodes_len) {
    for Iter_Index(node_it, numa->nodes_len) {
      Log_Zone_Scope("Node-ID: %llu - CPU Count: %llu", node_it, numa->nodes_dat[node_it].cpus_len) {
        for Iter_Index(cpu_it, numa->nodes_dat[node_it].cpus_len) {
          log_info("CPU-ID %llu", numa->nodes_dat[node_it].cpus_dat[cpu_it]);
        }
      }
    }
  }
}
