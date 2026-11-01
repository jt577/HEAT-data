"""Compare GEAT and ordinary-GA H-sublattice convergence by generation.

Candidate formation energies are read directly from each algorithm's
``genetic_optimization_log.txt``. Each curve highlights one trial, selected by earliest reference attainment,
with a band spanning all trials. The plotted quantity is the formation-energy gap per H atom to the common relaxed ground-state reference.
Logged energies and reference-hit detection retain their original per-total-atom
normalization; only the plotted energies are converted.
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
OUTPUT_DIR = BASE_DIR

# The H-sublattice genetic log does not include the reference energy in its
# ga_start record. This is the common relaxed-ground-truth value previously
# recorded for these runs. Update it here if the reference calculation changes.
GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM = -0.2228974413906268
SCAFFOLD_ATOM_COUNT = 90  # 30 Mg + 60 B
REFERENCE_H_ATOM_COUNT = 240

# Set an integer to cap both curves at that generation, inclusive. None shows
# every completed generation available in each source.
MAX_GENERATION_OVERRIDE = 20
# Once both methods have reached the reference, retain this many additional
# generations after the later hit and stop the common plot there.
GENERATIONS_AFTER_BOTH_HIT = 2
# Preserve the GA's zero-based numbering: generation 0 is the initial
# population, and generation 1 is the first evolved population.
GENERATION_NUMBERING = "zero_based"

ENERGY_Y_MIN = -0.01
ENERGY_Y_MAX = None
# A hit means the per-total-atom gap is below 1e-4 eV/atom.
GROUND_TRUTH_HIT_TOLERANCE_EVPATOM = 1.0e-4

GEAT_COLOR = "#1f1f1f"
ORDINARY_COLOR = "#2171b5"
BEST_LINE_WIDTH = 2.8
FIGURE_SIZE = (4.0, 4.2)
HORIZONTAL_REFERENCE_LINE_WIDTH = 3.0
VERTICAL_REFERENCE_LINE_WIDTH = 2.6
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14

@dataclass(frozen=True)
class Candidate:
    generation: int
    formation_energy: float
    delta_energy: float
    h_atom_count: int


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


def load_results(genetic_log: Path) -> tuple[float, list[Candidate]]:
    """Load hard-relaxed member formation energies from a genetic log."""
    ground_truth_energy = float(GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM)
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
                # HEAT reports realized H atoms separately from its soft sites.
                # In the ordinary GA, every site is a fully occupied H atom.
                h_atom_count = int(fields.get("realized_atom_count", fields["n_sites"]))
            except ValueError as exc:
                raise ValueError(
                    f"Malformed generation or formation energy in {genetic_log} "
                    f"on line {line_number}"
                ) from exc
            if not np.isfinite(formation_energy):
                continue
            if h_atom_count <= 0:
                raise ValueError(f"Cannot normalize by H count {h_atom_count}")
            candidates.append(
                Candidate(
                    generation=generation,
                    formation_energy=formation_energy,
                    delta_energy=formation_energy - ground_truth_energy,
                    h_atom_count=h_atom_count,
                )
            )

    if not candidates:
        raise ValueError(
            f"No records with generation and formation_energy_per_atom found in {genetic_log}"
        )
    if MAX_GENERATION_OVERRIDE is not None:
        generation_limit = int(MAX_GENERATION_OVERRIDE)
        candidates = [
            candidate for candidate in candidates
            if candidate.generation <= generation_limit
        ]
    if not candidates:
        raise ValueError("No candidate rows remained after the generation cap")
    return ground_truth_energy, candidates


def best_energy_so_far(
    candidates: list[Candidate], *, per_h: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return one cumulative-best energy gap for each available generation."""
    best_by_generation: dict[int, float] = {}
    for candidate in candidates:
        delta_energy = candidate.delta_energy
        if per_h:
            reference_per_h = GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM * (
                SCAFFOLD_ATOM_COUNT + REFERENCE_H_ATOM_COUNT
            ) / REFERENCE_H_ATOM_COUNT
            delta_energy = candidate.formation_energy * (
                SCAFFOLD_ATOM_COUNT + candidate.h_atom_count
            ) / candidate.h_atom_count - reference_per_h
        best_by_generation[candidate.generation] = min(
            delta_energy,
            best_by_generation.get(candidate.generation, np.inf),
        )

    generations = np.asarray(sorted(best_by_generation), dtype=int)
    generation_minima = np.asarray(
        [best_by_generation[generation] for generation in generations], dtype=float
    )
    return generations, np.minimum.accumulate(generation_minima)


def ground_truth_hit_generation(candidates: list[Candidate]) -> int | None:
    """Return the first generation whose cumulative gap reaches the tolerance."""
    generations, best = best_energy_so_far(candidates)
    hits = np.flatnonzero(best < GROUND_TRUTH_HIT_TOLERANCE_EVPATOM)
    return int(generations[hits[0]]) if len(hits) else None


def common_plot_horizon(
    geat_candidates: list[Candidate],
    ordinary_candidates: list[Candidate],
) -> int:
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


def best_energy_through_horizon(
    candidates: list[Candidate], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a cumulative-best series and hold its last value to the horizon."""
    clipped = [candidate for candidate in candidates if candidate.generation <= horizon]
    if not clipped:
        raise ValueError(f"No candidate records at or before generation {horizon}")
    generations, best = best_energy_so_far(clipped, per_h=True)
    if generations[-1] < horizon:
        generations = np.append(generations, int(horizon))
        best = np.append(best, best[-1])
    return generations, best


def style_axis(axis):
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main():
    heat_paths = discover_logs(BASE_DIR / "eat")
    conventional_paths = discover_logs(BASE_DIR / "ordinary")
    reference = GROUND_TRUTH_FORMATION_ENERGY_PER_ATOM
    horizon = int(MAX_GENERATION_OVERRIDE)
    load = lambda path: load_results(path)[1]
    curve = lambda candidates: best_energy_so_far(candidates, per_h=True)
    series = {name: aggregate(paths, load, curve, ground_truth_hit_generation, horizon)
              for name, paths in [("HEAT", heat_paths), ("Conventional", conventional_paths)]}
    output = OUTPUT_DIR / "h_sublattice_convergence.png"
    fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    for name, color in [("HEAT", GEAT_COLOR), ("Conventional", ORDINARY_COLOR)]:
        result = series[name]
        x = result["generations"]
        axis.fill_between(x, result["minimum"], result["maximum"], color=color, alpha=0.18, linewidth=0)
        axis.plot(x, result["selected"], color=color, linewidth=BEST_LINE_WIDTH, label=name)
    axis.axhline(0, color="#c62828", linestyle="--", linewidth=HORIZONTAL_REFERENCE_LINE_WIDTH, zorder=1)
    for name, color in [("HEAT", GEAT_COLOR), ("Conventional", ORDINARY_COLOR)]:
        result = series[name]
        hit_generation = result["trials"][result["selected_index"]]["reference_hit_generation"]
        if hit_generation is not None:
            axis.scatter([hit_generation], [0.0], color=color, s=34, zorder=5)
    axis.set_ylabel(r"$E_{\rm best}-E_{\rm ref}$ (eV/H atom)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xlabel("Generation", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.set_xlim(-0.7, horizon+0.7)
    axis.set_ylim(bottom=ENERGY_Y_MIN)
    axis.set_xticks(np.arange(0, horizon+1, max(1, int(np.ceil(horizon/7)))))
    style_axis(axis)
    axis.legend(frameon=False, fontsize=TICK_LABEL_SIZE)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    save_summary(output.with_name(output.stem+"_statistics"), series, "eV/H atom", reference)
    print("Saved", output)


if __name__ == "__main__":
    main()
