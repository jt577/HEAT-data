"""Compare HEAT and ordinary-GA edge-dislocation convergence.

Candidate formation energies are read directly from each algorithm's
``genetic_optimization_log.txt``. The common ground-truth reference is read
from the GEAT log's ``ga_start`` record.
Plot the whole-system formation-energy difference divided by a fixed count of
48 defect-core atoms (the chosen eight-atom core per plane, across six planes).
This is not an energy decomposition restricted to those atoms. Reference-hit
detection retains the original per-total-atom criterion used in the searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convergence_ensemble import discover_logs, aggregate, save_summary

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
GEAT_GENETIC_LOG = BASE_DIR / "eat" / "genetic_optimization_log.txt"
ORDINARY_GENETIC_LOG = BASE_DIR / "ordinary" / "genetic_optimization_log.txt"
OUTPUT_PATH = BASE_DIR / "edge_convergence.png"

# Set an integer to cap both source series at that generation, inclusive.
# None uses a shared endpoint: when both methods hit the reference, the plot
# ends shortly after the later hit; otherwise it uses the latest available data.
MAX_GENERATION_OVERRIDE = 20
GENERATIONS_AFTER_BOTH_HIT = 2

ENERGY_Y_MIN = -0.1
# A hit means the per-total-atom gap is below 1e-4 eV/atom.
GROUND_TRUTH_HIT_TOLERANCE_EVPATOM = 1.0e-4

GEAT_COLOR = "#1f1f1f"
ORDINARY_COLOR = "#2171b5"
BEST_LINE_WIDTH = 3.0
REFERENCE_LINE_WIDTH = 2.0
CONVERGENCE_LINE_WIDTH = 2.8
CONVERGENCE_MARKER_SIZE = 90
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 17
LEGEND_LABEL_SIZE = 20
FIGURE_SIZE = (6.8, 4.8)

GROUND_TRUTH_LOG_FIELD = "ground_truth_reference_formation_energy_per_atom"
SCAFFOLD_ATOM_COUNT = 605
REFERENCE_ACTIVE_ATOM_COUNT = 364
CORE_ATOM_COUNT = 8 * 6


@dataclass(frozen=True)
class Candidate:
    generation: int
    formation_energy: float
    delta_energy: float
    delta_energy_per_core_atom: float


def best_energy_so_far(
    candidates: list[Candidate], *, per_core: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return one cumulative-best relaxed energy gap per available generation."""
    best_by_generation: dict[int, float] = {}
    for candidate in candidates:
        best_by_generation[candidate.generation] = min(
            candidate.delta_energy_per_core_atom if per_core else candidate.delta_energy,
            best_by_generation.get(candidate.generation, np.inf),
        )

    generations = np.asarray(sorted(best_by_generation), dtype=int)
    generation_minima = np.asarray(
        [best_by_generation[generation] for generation in generations],
        dtype=float,
    )
    return generations, np.minimum.accumulate(generation_minima)


def ground_truth_hit_generation(candidates: list[Candidate]) -> int | None:
    """Return the first generation whose cumulative gap reaches the tolerance."""
    generations, best = best_energy_so_far(candidates)
    hits = np.flatnonzero(best < GROUND_TRUTH_HIT_TOLERANCE_EVPATOM)
    return int(generations[hits[0]]) if len(hits) else None


def common_plot_horizon(geat_candidates, ordinary_candidates):
    """Choose one endpoint shared by both cumulative-best curves."""
    if MAX_GENERATION_OVERRIDE is not None:
        return int(MAX_GENERATION_OVERRIDE)
    geat_hit = ground_truth_hit_generation(geat_candidates)
    ordinary_hit = ground_truth_hit_generation(ordinary_candidates)
    if geat_hit is not None and ordinary_hit is not None:
        return max(geat_hit, ordinary_hit) + int(GENERATIONS_AFTER_BOTH_HIT)
    return max(
        max(candidate.generation for candidate in geat_candidates),
        max(candidate.generation for candidate in ordinary_candidates),
    )


def best_energy_through_horizon(candidates, horizon):
    """Clip a cumulative-best series and hold its last value to the horizon."""
    clipped = [candidate for candidate in candidates if candidate.generation <= horizon]
    if not clipped:
        raise ValueError(f"No candidate records at or before generation {horizon}")
    generations, best = best_energy_so_far(clipped, per_core=True)
    if generations[-1] < horizon:
        generations = np.append(generations, int(horizon))
        best = np.append(best, best[-1])
    return generations, best


def parse_log_record(line: str) -> tuple[str, dict[str, str]]:
    """Parse one ``event | key=value`` genetic-log record."""
    pieces = [piece.strip() for piece in line.split("|") if piece.strip()]
    if not pieces:
        return "", {}
    fields = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key.strip()] = value.strip()
    return pieces[0], fields


def reference_energy_from_log(genetic_log: Path) -> float:
    """Read the hard ground-truth formation energy from ``ga_start``."""
    with genetic_log.open(encoding="utf-8") as handle:
        for line in handle:
            event, fields = parse_log_record(line)
            if event != "ga_start":
                continue
            if GROUND_TRUTH_LOG_FIELD not in fields:
                break
            value = float(fields[GROUND_TRUTH_LOG_FIELD])
            if not np.isfinite(value):
                break
            return value
    raise ValueError(
        f"Could not find {GROUND_TRUTH_LOG_FIELD} in the ga_start record of {genetic_log}"
    )


def load_results(genetic_log: Path, ground_truth_energy: float) -> list[Candidate]:
    """Load hard-relaxed member formation energies from a genetic log."""
    candidates = []
    with genetic_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            event, fields = parse_log_record(line)
            if event not in {"member_done", "elite_copied", "imported_seed_used"}:
                continue
            if "generation" not in fields or "formation_energy_per_atom" not in fields:
                continue
            try:
                generation = int(fields["generation"])
                formation_energy = float(fields["formation_energy_per_atom"])
                # HEAT's realized count excludes null sites; every ordinary-GA
                # site is a hard atom. Both counts exclude the fixed scaffold.
                active_atom_count = int(
                    fields["realized_atom_count"]
                    if "realized_atom_count" in fields else fields["n_sites"]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Malformed generation or formation energy in {genetic_log} "
                    f"on line {line_number}"
                ) from exc
            if not np.isfinite(formation_energy):
                continue
            candidates.append(
                Candidate(
                    generation=generation,
                    formation_energy=formation_energy,
                    delta_energy=formation_energy - float(ground_truth_energy),
                    delta_energy_per_core_atom=(
                        formation_energy * (SCAFFOLD_ATOM_COUNT + active_atom_count)
                        - float(ground_truth_energy)
                        * (SCAFFOLD_ATOM_COUNT + REFERENCE_ACTIVE_ATOM_COUNT)
                    ) / CORE_ATOM_COUNT,
                )
            )

    if not candidates:
        raise ValueError(
            f"No records with generation and formation_energy_per_atom found in {genetic_log}"
        )

    if MAX_GENERATION_OVERRIDE is not None:
        generation_limit = int(MAX_GENERATION_OVERRIDE)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.generation <= generation_limit
        ]
    if not candidates:
        raise ValueError(
            f"No candidates remained after the generation cap for {genetic_log}"
        )
    return candidates


def style_axis(axis):
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main():
    heat_paths = discover_logs(BASE_DIR / "eat")
    conventional_paths = discover_logs(BASE_DIR / "ordinary")
    reference = reference_energy_from_log(heat_paths[0])
    horizon = int(MAX_GENERATION_OVERRIDE)
    load = lambda path: load_results(path, reference)
    curve = lambda candidates: best_energy_so_far(candidates, per_core=True)
    series = {name: aggregate(paths, load, curve, ground_truth_hit_generation, horizon)
              for name, paths in [("HEAT", heat_paths), ("Conventional", conventional_paths)]}
    output = OUTPUT_PATH
    fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    for name, color in [("HEAT", GEAT_COLOR), ("Conventional", ORDINARY_COLOR)]:
        result = series[name]
        x = result["generations"]
        axis.fill_between(x, result["minimum"], result["maximum"], color=color, alpha=0.18, linewidth=0)
        axis.plot(x, result["selected"], color=color, linewidth=BEST_LINE_WIDTH, label=name)
    axis.axhline(0, color="#c62828", linestyle="--", linewidth=REFERENCE_LINE_WIDTH, zorder=1)
    for name, color in [("HEAT", GEAT_COLOR), ("Conventional", ORDINARY_COLOR)]:
        result = series[name]
        hit_generation = result["trials"][result["selected_index"]]["reference_hit_generation"]
        if hit_generation is not None:
            axis.scatter([hit_generation], [0.0], color=color, s=CONVERGENCE_MARKER_SIZE, zorder=5)
    axis.set_ylabel(r"$E_{\rm best}-E_{\rm ref}$ (eV/core atom)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xlabel("Generation", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.set_xlim(-0.7, horizon+0.7)
    axis.set_ylim(bottom=ENERGY_Y_MIN)
    axis.set_xticks(np.arange(0, horizon+1, max(1, int(np.ceil(horizon/7)))))
    style_axis(axis)
    axis.legend(frameon=False, fontsize=LEGEND_LABEL_SIZE)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    save_summary(output.with_name(output.stem+"_statistics"), series, "eV/core atom", reference)
    print("Saved", output)


if __name__ == "__main__":
    main()
