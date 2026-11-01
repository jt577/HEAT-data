"""Aggregate independent convergence trials, preserving the plotting normalization."""
from pathlib import Path
import hashlib
import json
import numpy as np


def discover_logs(directory):
    paths = sorted(directory.glob("trial_*/genetic_optimization_log.txt"))
    root = directory / "genetic_optimization_log.txt"
    if root.exists():
        paths.append(root)
    # A relocated original run must not be counted twice.
    unique, seen = [], set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            unique.append(path)
            seen.add(digest)
    if not unique:
        raise ValueError(f"No trial logs in {directory}")
    return unique


def aggregate(paths, load, curve, hit, horizon):
    grid = np.arange(horizon + 1)
    values, records = [], []
    for path in paths:
        completed = set()
        for line in path.read_text().splitlines():
            if line.startswith("generation_done |"):
                fields = dict(p.strip().split("=", 1) for p in line.split("|")[1:] if "=" in p)
                completed.add(int(fields["generation"]))
        candidates = [c for c in load(path) if c.generation in completed and c.generation <= horizon]
        generations, best = curve(candidates)
        last = int(generations[-1])
        if not np.array_equal(generations, np.arange(last + 1)):
            raise ValueError(f"Missing completed generations: {path}")
        reached = hit(candidates)
        if last < horizon and reached is None:
            raise ValueError(f"Unconverged run ends before plotting horizon: {path}")
        # Keep the achieved incumbent after an early stop; do not invent improvements.
        aligned = best[np.minimum(grid, last)]
        values.append(aligned)
        records.append(dict(source=str(path.relative_to(path.parents[2])),
                            last_completed_generation=last,
                            reference_hit_generation=reached,
                            carried_forward_from_generation=last if last < horizon else None,
                            final_gap=float(aligned[-1])))
    matrix = np.asarray(values)
    # Select one actual trajectory, never a pointwise splice of different runs.
    # Ties in first-hit generation are resolved by the final plotted gap;
    # if no run reaches the reference, select the lowest final gap.
    selected = min(range(len(records)), key=lambda i: (
        records[i]["reference_hit_generation"]
        if records[i]["reference_hit_generation"] is not None else float("inf"),
        records[i]["final_gap"], records[i]["source"]))
    return dict(generations=grid, selected_index=selected,
                selected=matrix[selected], minimum=matrix.min(axis=0), maximum=matrix.max(axis=0),
                mean=matrix.mean(axis=0),
                std=matrix.std(axis=0, ddof=1) if len(matrix)>1 else np.zeros_like(grid,dtype=float),
                trials=records, values=matrix)


def save_summary(output, series, normalization, reference):
    payload = dict(statistic="Single trial reaching the reference in the fewest generations",
                   selection_rule="Earliest reference-hit generation; ties resolved by lowest final plotted gap, then source path. If no run hits, use lowest final plotted gap.",
                   band="Pointwise minimum to maximum across all trials, including the highlighted trial",
                   normalization=normalization, reference_energy_eV_per_total_atom=reference,
                   reference_hit_tolerance_eV_per_total_atom=1e-4,
                   early_stop_policy="Carry forward the final incumbent only for runs that reached the reference tolerance",
                   methods={})
    for name, result in series.items():
        payload["methods"][name] = {k:result[k].tolist() for k in ["generations","selected","minimum","maximum","mean","std","values"]}
        payload["methods"][name].update(n=len(result["trials"]), trials=result["trials"],
                                         selected_index=result["selected_index"],
                                         selected_trial=result["trials"][result["selected_index"]])
        print("Highlighted", name, result["trials"][result["selected_index"]])
        print(name, "n=", len(result["trials"]), "hit generations=",
              [r["reference_hit_generation"] for r in result["trials"]],
              "final mean +/- SD=", result["mean"][-1], result["std"][-1])
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2)+"\n")
