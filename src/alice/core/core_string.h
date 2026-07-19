// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

struct Arena;

// ------------------------------------------------------------
// #-- Character Operations

inline function B32 char_is_alpha      (I08 c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline function B32 char_is_digit      (I08 c) { return c >= '0' && c <= '9'; }
inline function B32 char_is_upper      (I08 c) { return c >= 'A' && c <= 'Z'; }
inline function B32 char_is_lower      (I08 c) { return c >= 'a' && c <= 'z'; }
inline function B32 char_is_whitespace (I08 c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
inline function B32 char_is_visible    (I08 c) { return c >= '!' && c <= '~'; }
inline function B32 char_is_linefeed   (I08 c) { return c == '\r' || c == '\n'; }
inline function I08 char_to_upper      (I08 c) { return !char_is_alpha(c) ? c : (c >  'Z' ? c - ('A' - 'a') : c); }
inline function I08 char_to_lower      (I08 c) { return !char_is_alpha(c) ? c : (c <= 'Z' ? c + ('A' - 'a') : c); }

// ------------------------------------------------------------
// #-- String Types

typedef struct Str08 { U64 len; U08 *txt; } Str08;
typedef struct Str16 { U64 len; U16 *txt; } Str16;
typedef struct Str32 { U64 len; U32 *txt; } Str32;
typedef struct Str64 { U64 len; U64 *txt; } Str64;

#define str08_lit(literal_) ((Str08) { .txt = (U08 *)(literal_), .len = sizeof(literal_) - 1 })

force_inline function Str08 str08(U64 len, U08 *txt) {
  return (Str08) { .len = len, .txt = txt };
}

// ------------------------------------------------------------
// #-- String Operations

function Str08  str08_slice                 (Str08 base, U64 start, U64 len);
function Str08  str08_from_cstring          (char *cstring);
function Str08  str08_trim                  (Str08 string);
function B32    str08_match                 (Str08 lhs, Str08 rhs);
function B32    str08_starts_with           (Str08 base, Str08 start);
function B32    str08_contains              (Str08 base, Str08 sub);
function B32    str08_match_any_case        (Str08 lhs, Str08 rhs);
function B32    str08_starts_with_any_case  (Str08 base, Str08 start);
function B32    str08_contains_any_case     (Str08 base, Str08 sub);
function U64    str08_hash                  (Str08 string);

function U64    cstring_len                 (char *cstring);

function F64    f64_from_str08              (Str08 string);
function I64    i64_from_str08              (Str08 string);
function U64    u64_from_str08              (Str08 string);

// ------------------------------------------------------------
// #-- String Scanner

typedef struct Scan_Error {
    struct Scan_Error  *next;
    U08                 line_at;
    U08                 char_at;
    Str08               message;
} Scan_Error;

typedef struct Scan {
    struct Arena   *arena;
    Str08           stream;
    U32             at;
    U32             line_at;
    U32             char_at;
    Scan_Error     *error_first;
    Scan_Error     *error_last;
} Scan;

function void        scan_init            (Scan *scan, struct Arena *arena, Str08 stream);
function void        scan_error_push      (Scan *scan, Str08 message);
function Scan_Error *scan_error           (Scan *scan);
function U08         scan_char            (Scan *scan);
function void        scan_move            (Scan *scan, U32 offset);
function void        scan_skip_whitespace (Scan *scan);
function void        scan_skip_line       (Scan *scan);
function B32         scan_end             (Scan *scan);
function Str08       scan_identifier      (Scan *scan);
function Str08       scan_str             (Scan *scan);
function U64         scan_u64             (Scan *scan);
function F64         scan_f64             (Scan *scan);
function B32         scan_require         (Scan *scan, Str08 match);
