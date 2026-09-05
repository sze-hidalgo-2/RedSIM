  UG_Mesh *mesh = euler->mesh;
  Range1_U64 groups[2] = { mesh->groups.cells_interior, mesh->groups.cells_boundary };
  for Iter_Index(g, 2) {
    Range1_U64 group = groups[g];
    U64 group_len    = range1_u64_len(group);
    for Iter_Range(it_local, lane_range(group_len)) {  // mirrors the runtime partition exactly
      U64 it_cell = group.min + it_local;
      U32 my_lane = lane_index(); // the thread currently owning it_cell under this group's partition
      for Iter_Index(it_face, 4) {
        U32 adjacent = mesh->cells.faces[it_cell].adjacent[it_face];
        UG_Face_Role role = UG_Face_Role_Independent;
        // NOTE: only cells that fall in THIS SAME group qualify for lane-local treatment.
        if (adjacent >= group.min && adjacent < group.max) {
          U64 adjacent_local = adjacent - group.min;
          U32 adjacent_lane  = lane_owner_of(adjacent_local, group_len); // same partition function lane_range() uses internally
          if (adjacent_lane == my_lane) {
            role = (adjacent > it_cell) ? UG_Face_Role_Owner : UG_Face_Role_Skip;
          }
        }
        euler->face_role[4 * it_cell + it_face] = (U08)role;
      }
    }
  }
  lane_barrier();
