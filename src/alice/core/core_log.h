// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Logging API

enum {
    Logger_Max_Hooks        = 32,
    Logger_Max_Entry_Length = 1024,
};

typedef U32 Logger_Entry_Type;
enum {
    Logger_Entry_Info,
    Logger_Entry_Debug,
    Logger_Entry_Warning,
    Logger_Entry_Error,
    Logger_Entry_Fatal,
    
    Logger_Entry_Zone_Start,
    Logger_Entry_Zone_End,
    
    Logger_Entry_Count
};

typedef U32 Logger_Filter_Flag;
enum {
    Logger_Filter_Info     = 1 << 0,
    Logger_Filter_Debug    = 1 << 1,
    Logger_Filter_Warning  = 1 << 2,
    Logger_Filter_Error    = 1 << 3,
    Logger_Filter_Fatal    = 1 << 4,
    Logger_Filter_Zone     = 1 << 5,
    
    Logger_Filter_Build_Debug =
        Logger_Filter_Debug   |
        Logger_Filter_Info    |
        Logger_Filter_Warning |
        Logger_Filter_Error   |
        Logger_Filter_Fatal   |
        Logger_Filter_Zone,
    
    Logger_Filter_Build_Release =
        Logger_Filter_Info     |
        Logger_Filter_Warning  |
        Logger_Filter_Error    |
        Logger_Filter_Fatal    |
        Logger_Filter_Zone,
    
#if BUILD_DEBUG
    Logger_Filter_Build_Active = Logger_Filter_Build_Debug,
#else
    Logger_Filter_Build_Active = Logger_Filter_Build_Release,
#endif
};

force_inline function U32 logger_filter_flag_from_entry_type(U32 entry) {
    U32 result = 0;
    switch (entry) {
        case Logger_Entry_Info:       { result = Logger_Filter_Info; }    break;
        case Logger_Entry_Debug:      { result = Logger_Filter_Debug; }   break;
        case Logger_Entry_Warning:    { result = Logger_Filter_Warning; } break;
        case Logger_Entry_Error:      { result = Logger_Filter_Error; }   break;
        case Logger_Entry_Fatal:      { result = Logger_Filter_Fatal; }   break;
        case Logger_Entry_Zone_Start: { result = Logger_Filter_Zone; }    break;
        case Logger_Entry_Zone_End:   { result = Logger_Filter_Zone; }    break;
        Invalid_Default;
    }
    
    return result;
}

typedef struct Logger_Entry {
    Logger_Entry_Type type;
    U08               message[Logger_Max_Entry_Length];
    Local_Time        time;
    Caller_Metadata   caller;
} Logger_Entry;

// NOTE(cmat): Callback prototypes.
typedef void Logger_Write_Entry_Hook  (Logger_Entry_Type type, Str08 buffer);
typedef void Logger_Format_Entry_Hook (Logger_Entry *entry, U08 *entry_buffer, U32 zone_depth);

// NOTE(cmat): Each thread shares the same global logger.
// - Every log function is thread safe (they are all wrapped in a mutex).
typedef struct Logger_State {
  volatile Logger_Filter_Flag filter;
  U32                         zone_depth;
  U32                         hook_count;
  Logger_Write_Entry_Hook    *write_hooks [Logger_Max_Hooks];
  Logger_Format_Entry_Hook   *format_hooks[Logger_Max_Hooks];
  Mutex                       mutex;
} Logger_State;

function void  logger_set_filter                    (Logger_Filter_Flag filter);
function B32   logger_filter_type                   (Logger_Entry_Type type);
function void  logger_entry                         (Logger_Entry *entry);
function void  logger_push_hook                     (Logger_Write_Entry_Hook *write, Logger_Format_Entry_Hook *format);
function void  log_message_ext                      (Logger_Entry_Type type, Caller_Metadata caller_meta, char *format, ...);

// NOTE(cmat): Default hooks.
function void  logger_write_entry_standard_stream   (Logger_Entry_Type type, Str08 buffer);
function void  logger_format_entry_minimal          (Logger_Entry *entry, U08 *entry_buffer, U32 zone_depth);
function void  logger_format_entry_detailed         (Logger_Entry *entry, U08 *entry_buffer, U32 zone_depth);

#define log_message(type_, format_, ...) log_message_ext(type_, Caller_Metadata_Get, format_,##__VA_ARGS__);
#define log_info(format_, ...)           log_message(Logger_Entry_Info,       format_,##__VA_ARGS__);
#define log_debug(format_, ...)          log_message(Logger_Entry_Debug,      format_,##__VA_ARGS__);
#define log_warning(format_, ...)        log_message(Logger_Entry_Warning,    format_,##__VA_ARGS__);
#define log_fatal(format_, ...)          log_message(Logger_Entry_Fatal,      format_,##__VA_ARGS__);
#define log_zone_start(format_, ...)     log_message(Logger_Entry_Zone_Start, format_,##__VA_ARGS__);
#define log_zone_end()                   log_message(Logger_Entry_Zone_End, "");

#define Log_Zone_Scope(format_, ...)     Defer_Scope(log_message_ext(Logger_Entry_Zone_Start, Caller_Metadata_Get, format_,##__VA_ARGS__), \
log_message_ext(Logger_Entry_Zone_End,   Caller_Metadata_Get, ""))

// ------------------------------------------------------------
// #-- Common Utility Logging Functions

function void log_sys_context     (void);
function void log_sys_numa_layout (void);
