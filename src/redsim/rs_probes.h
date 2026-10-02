#include <stdio.h>

// NOTE(cmat): Probe-location loader + hourly NOx probe CSV writer.
// - Probe file format (TAB separated, no header, one probe per line):
//     name  lon  lat  height  x  y  z  "description"
//   Only column 1 (name) and columns 5, 6, 7 (x, y, z - domain coordinates in meters, i.e. already
//   relative to the UTM origin v3f(441918, 4474610, 0) used for the emission lines) are read.
// - Lines that don't parse (blank lines, a header, ...) are skipped.

typedef struct RS_Probe {
  Str08 name;
  V3F   position; // NOTE(cmat): columns 5,6,7 as given in the file (meters, relative to the UTM origin).
  I64   cell;     // NOTE(cmat): index of the cell on THIS rank that contains the probe, -1 if this rank doesn't own it.
} RS_Probe;

typedef struct RS_Probe_Array {
  U64       len;
  RS_Probe *dat;
} RS_Probe_Array;

typedef struct RS_Probe_Writer {
  FILE *file; // NOTE(cmat): only non-null on lane 0.
} RS_Probe_Writer;

// NOTE(cmat): Loads on lane 0 and broadcasts to every lane - safe to call from every lane.
function RS_Probe_Array rs_probes_load(Arena *arena, Str08 file_path);

// NOTE(cmat): Finds the cell (owned by this rank) containing each probe. Lane 0 does the work -
// - call from every lane, followed by a lane_barrier().
function void rs_probes_locate(RS_Probe_Array *probes, UG_Mesh *mesh, Arena *arena, FL_Scale *scale, F64 search_step_m);

// NOTE(cmat): Opens "<stem>.csv" (single rank) or "<stem>_rank<N>.csv" (several ranks) and writes the
// - header: hour,elapsed_s,utc,<probe names...>. Every rank writes every probe column; a probe this
// - rank doesn't own is left empty, so per-rank files can be merged by overlaying them.
function void rs_probes_csv_open(RS_Probe_Writer *writer, RS_Probe_Array *probes, const char *stem);

// NOTE(cmat): Appends one row (flushed immediately). `phi` is the per-cell scalar array.
function void rs_probes_csv_write_row(RS_Probe_Writer *writer, RS_Probe_Array *probes, F32 *phi,
                                      U64 hour_index, F64 elapsed_seconds, RS_Sim_Time time);

function void rs_probes_csv_close(RS_Probe_Writer *writer);
