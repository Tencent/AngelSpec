def balance_packed_rows(rank_rows, lengths, rank_loads):
    if not rank_rows or len(rank_rows) != len(rank_loads):
        raise ValueError("Expected one load for each nonempty DP rank")
    counts = {len(rows) for rows in rank_rows}
    if len(counts) != 1 or next(iter(counts)) < 1:
        raise ValueError("Every DP rank must have the same positive row count")
    if any(load < 0 for load in rank_loads):
        raise ValueError("Rank loads must be nonnegative")
    rows = [list(row) for group in rank_rows for row in group]
    seen = set()
    for row in rows:
        if not row:
            raise ValueError("Packed rows must not be empty")
        for index in row:
            if index < 0 or index >= len(lengths) or lengths[index] <= 0 or index in seen:
                raise ValueError("Invalid or repeated packed sample index")
            seen.add(index)
    costs = [sum(lengths[index] for index in row) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: (-costs[i], tuple(rows[i])))
    assigned = [[] for _ in rank_rows]
    loads = list(rank_loads)
    count = next(iter(counts))
    for index in order:
        rank = min(
            (rank for rank in range(len(loads)) if len(assigned[rank]) < count),
            key=lambda rank: (loads[rank], rank),
        )
        assigned[rank].append(rows[index])
        loads[rank] += costs[index]
    return assigned, loads
