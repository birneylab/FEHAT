timestamps = [[s for s in path.split(
        '-T') if s[0].isdigit()][-1][0:10] for path in sorted_frame_paths]