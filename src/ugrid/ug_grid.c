function void ug_grid_init(UG_Grid *grid) {
  Zero_Fill(grid);
  arena_init(&grid->arena);
}
