// (C) Copyright 2025 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// ------------------------------------------------------------
// #-- Strings

function Str08 str08_slice(Str08 base, U64 start, U64 len) {
  Assert(base.len >= start + len, "invalid string slice");
  
  U64 clamped = 0;
  if (start < base.len) {
      clamped = u64_min(len, base.len - start);
  }

  return (Str08) { .len = clamped, .txt = base.txt + start };
}

function Str08 str08_from_cstring(char *cstring) {
  Str08 result = {
    .len = cstring_len(cstring),
    .txt = (U08 *)cstring
  };

  return result;
}

function Str08 str08_trim(Str08 string) {
  U64 start = 0;
  for Iter_Index(it, string.len) {
    if (char_is_whitespace(string.txt[it])) {
        start++;
    } else {
      break;
    }
  }

  U64 end = string.len;
  for Iter_Index(it, string.len) {
    if (char_is_whitespace(string.txt[string.len - it - 1])) {
      end--;
    } else {
      break;
    }
  }

  return str08_slice(string, start, end - start);
}

function B32 str08_match(Str08 lhs, Str08 rhs) {
  B32 result = lhs.len == rhs.len;
  if (result) {
    for Iter_Index(it, lhs.len) {
      if (lhs.txt[it] != rhs.txt[it]) {
        result = 0;
        break;
      }
    }
  }

  return result;
}

function B32 str08_match_any_case(Str08 lhs, Str08 rhs) {
  B32 result = lhs.len == rhs.len;
  if (result) {
    for Iter_Index(it, lhs.len) {
      if (char_to_lower(lhs.txt[it]) != char_to_lower(rhs.txt[it])) {
        result = 0;
        break;
      }
    }
  }

  return result;
}

function B32 str08_starts_with(Str08 base, Str08 start) {
  B32 result = base.len >= start.len;
  if (result) {
    result = str08_match(str08_slice(base, 0, start.len), start);
  }

  return result;
}

function B32 str08_starts_with_any_case(Str08 base, Str08 start) {
  B32 result = base.len >= start.len;
  if (result) {
    result = str08_match_any_case(str08_slice(base, 0, start.len), start);
  }

  return result;
}

function B32 str08_contains(Str08 base, Str08 sub) {
  B32 result = 0;
  if (sub.len <= base.len) {
    for Iter_Index(it, base.len - sub.len) {
      result = (str08_match(str08_slice(base, it, sub.len), sub));
      if (result) break;
    }
  }

  return result;
}

function B32 str08_contains_any_case(Str08 base, Str08 sub) {
  B32 result = 0;
  if (sub.len <= base.len) {
    for Iter_Index(it, base.len - sub.len) {
      result = (str08_match_any_case(str08_slice(base, it, sub.len), sub));
      if (result) break;
    }
  }

  return result;
}

function I64 str08_find(Str08 base, Str08 sub) {
  I64 result = -1;
  if (sub.len <= base.len) {
    for Iter_Index(it, base.len - sub.len + 1) {
      if (str08_match(str08_slice(base, it, sub.len), sub)) {
        result = (I64)it;
        break;
      }
    }
  }

  return result;
}

// NOTE(cmat): djb2, Dan Bernstein
function U64 str08_hash(Str08 string) {
  U64 hash = 5381;
  for Iter_Index(it, string.len) {
    hash = ((hash << 5) + hash) + string.txt[it];
  }

  return hash;
}

function U64 cstring_len(char *cstring) {
  U64 len = 0;
  while(*cstring++) {
    len++;
  }
  
  return len;
}

function F64 f64_from_str08(Str08 string) {
  ffc_outcome outcome = { };
  F64 result = ffc_parse_double_simple(string.len, (char *)string.txt, &outcome);
  return result;
}

function I64 i64_from_str08(Str08 string) {
  ffc_outcome outcome = { };
  I64 result = ffc_parse_i64_simple(string.len, (char *)string.txt, 10, &outcome);
  return result;
}

function U64 u64_from_str08(Str08 string) {
  ffc_outcome outcome = { };
  U64 result = ffc_parse_u64_simple(string.len, (char *)string.txt, 10, &outcome);
  return result;
}

// ------------------------------------------------------------
// #-- String Scanner

function void scan_init(Scan *scan, Arena *arena, Str08 stream) {
  Zero_Fill(scan);
  scan->arena   = arena;
  scan->stream  = stream;
  scan->line_at = 1;
  scan->char_at = 0;
}

function void scan_error_push(Scan *scan, Str08 message) {
  Scan_Error *error = arena_push_type(scan->arena, Scan_Error);
  Queue_Push(scan->error_first, scan->error_last, error);
  
  error->line_at = scan->line_at;
  error->char_at = scan->char_at;
  error->message = arena_push_str08(scan->arena, message);
}

function Scan_Error *scan_error(Scan *scan) {
  return scan->error_first;
}

function U08 scan_char(Scan *scan) {
  U08 result = scan->at < scan->stream.len ? scan->stream.txt[scan->at] : 0;
  return result;
}

function void scan_move(Scan *scan, U32 offset) {
  for Iter_Index(it, offset) {
    if (scan->at >= scan->stream.len) {
      break;
    }

    if (scan->stream.txt[scan->at] == '\n') {
      scan->line_at   += 1;
      scan->char_at   = 0;
    }

    scan->at       += 1;
    scan->char_at  += 1;
  }
}

function void scan_skip_whitespace(Scan *scan) {
  while (char_is_whitespace(scan_char(scan))) {
    scan_move(scan, 1);
  }
}

function void scan_skip_line(Scan *scan) {
  for (;;) {
    U08 c = scan_char(scan);
    if (char_is_linefeed(c) || c == 0) {
      break;
    }

    scan_move(scan, 1);
  }
}

function B32 scan_end(Scan *scan) {
  scan_skip_whitespace(scan);
  B32 result = scan_char(scan) == 0;
  return result;
}


function Str08 scan_identifier(Scan *scan) {
  Str08 result = { };
  scan_skip_whitespace(scan);

  U64 start = scan->at;
  U08 c = scan_char(scan);
  if (char_is_alpha(c) || c == '_') {
    for (;;) {
      scan_move(scan, 1);
      c = scan_char(scan);
      if (!(char_is_alpha(c) || char_is_digit(c) || c == '_')) {
        break;
      }
    }

    result = str08_slice(scan->stream, start, scan->at - start);
  } else {
    scan_error_push(scan, str08_lit("expected identifier"));
  }

  return result;
}

function Str08 scan_str(Scan *scan) {
  Str08 result = { .len = 0 };
  scan_require(scan, str08_lit("\""));

  U64 start = scan->at;
  B32 str08_closed  = 0;
  if (!scan_error(scan)) {
    for (;;) {
      U08 c = scan_char(scan);

      if (c == 0) {
        break;
      } else if (c == '"') {
        scan_move(scan, 1);
        str08_closed = 1;
        break;
      }

      scan_move(scan, 1);
    }
  }


  if (!str08_closed) {
    scan_error_push(scan, str08_lit("unclosed string"));
  } else {
    result = str08_slice(scan->stream, start, scan->at - start - 1);
  }

  return result;
}

function U64 scan_u64(Scan *scan) {
  U64 result = 0;
  scan_skip_whitespace(scan);

  U64 start = scan->at;
  U08 c     = scan_char(scan);
  if (char_is_digit(c)) {
    for (;;) {
      scan_move(scan, 1);
      c = scan_char(scan);
      if (!char_is_digit(c)) {
        break;
      }
    }

    result = u64_from_str08(str08_slice(scan->stream, start, scan->at - start));
  }

  return result;
}

function F64 scan_f64(Scan *scan) {
  F64 result = 0;
  scan_skip_whitespace(scan);
  U64 start_at = scan->at;

  U08 c = scan_char(scan);
  if (c != '+' && c != '-' && !char_is_digit(c)) {
    scan_error_push(scan, str08_lit("expected f64"));
  } else {
    if (c == '+' || c == '-') {
      scan_move(scan, 1);
    }

    for (;;) { 
      c = scan_char(scan);
      if (!c || !char_is_digit(c)) { break; }
      scan_move(scan, 1);
    }

    if (c == '.') {
      scan_move(scan, 1);
      for (;;) { 
        c = scan_char(scan);
        if (!c || !char_is_digit(c)) { break; }
        scan_move(scan, 1);
      }
    }
    
    if (c == 'e' || c == 'E') {
      scan_move(scan, 1);
      c = scan_char(scan);
      if (c == '+' || c == '-') {
        scan_move(scan, 1);
      }

      for (;;) { 
        c = scan_char(scan);
        if (!c || !char_is_digit(c)) { break; }
        scan_move(scan, 1);
      }
    }

    Str08 slice = str08_slice(scan->stream, start_at, scan->at - start_at);
    result = f64_from_str08(slice);
  }

  return result;
}

function B32 scan_require(Scan *scan, Str08 match) {
  B32 result = 0;

  Scratch scratch = { };
  Scratch_Scope(&scratch, 0) {
    scan_skip_whitespace(scan);

    result = str08_match(match, str08_slice(scan->stream, scan->at, match.len));
    if (!result) {
      Str08 message = str08_format(scratch.arena, "expected \"%S\"", match);
      scan_error_push(scan, message);
    } else {
      scan_move(scan, match.len);
    }
  }

  return result;
}
