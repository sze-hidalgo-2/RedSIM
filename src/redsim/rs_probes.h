#include <stdio.h>

// NOTE(cmat): Probe-location loader + hourly NOx probe CSV writer.
// - Probe file format (TAB separated, no header, one probe per line):
//     name  lon  lat  height  x  y  z  "description"
//   Only column 1 (name) and columns 5, 6, 7 (x, y, z - domain coordinates in meters, i.e. already
//   relative to the UTM origin v3f(441918, 4474610, 0) used for the emission lines) are read.
// - Lines that don't parse (blank lines, a header, ...) are skipped.
// - Output is ONE csv, written by rank 0: every probe's value is gathered across ranks with
//   ipc_rank_sum_f64 (the owning rank contributes the value, every other rank contributes 0).

typedef struct RS_Probe {
  Str08 name;
  V3F   position; // NOTE(cmat): columns 5,6,7 as given in the file (meters, relative to the UTM origin).
  I64   cell;     // NOTE(cmat): index of the cell on THIS rank that contains the probe, -1 if this rank doesn't own it.
  U32   claims;   // NOTE(cmat): number of ranks (all ranks, set by rs_probes_locate) that found the probe. 0 = outside the mesh.
} RS_Probe;

typedef struct RS_Probe_Array {
  U64       len;
  RS_Probe *dat;
} RS_Probe_Array;

typedef struct RS_Probe_Writer {
  FILE *file; // NOTE(cmat): only non-null on rank 0, lane 0.
} RS_Probe_Writer;

// NOTE(cmat): Loads on lane 0 and broadcasts to every lane - safe to call from every lane.
function RS_Probe_Array rs_probes_load(Arena *arena, Str08 file_path);

// NOTE(cmat): Finds the cell (owned by this rank) containing each probe, then counts across all ranks how many
// - found it (collective: call from every lane of every rank).
function void rs_probes_locate(RS_Probe_Array *probes, UG_Mesh *mesh, Arena *arena, FL_Scale *scale, F64 search_step_m);

// NOTE(cmat): Rank 0 / lane 0 opens `path` and writes the header: hour,elapsed_s,utc,<probe names...>.
function void rs_probes_csv_open(RS_Probe_Writer *writer, RS_Probe_Array *probes, const char *path);

// NOTE(cmat): Collective (every lane of every rank, after the scalar solve): gathers every probe's value across
// - ranks and appends one row on rank 0 (flushed). `phi` is the per-cell scalar array. A probe outside the mesh
// - is left empty.
function void rs_probes_csv_write_row(RS_Probe_Writer *writer, RS_Probe_Array *probes, F32 *phi,
                                      U64 hour_index, F64 elapsed_seconds, RS_Sim_Time time);

function void rs_probes_csv_close(RS_Probe_Writer *writer);
