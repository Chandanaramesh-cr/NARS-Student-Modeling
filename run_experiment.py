# run_experiment.py
# Runs NARS + both baselines on all three student profiles.
# Saves results to results.json and writes four figures.
#
# Usage:
#   python3 run_experiment.py

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from nars_engine          import NARSStudentModel
from bkt_baseline         import BKTStudentModel
from rule_based_baseline  import RuleBasedStudentModel
from student_simulator    import CONCEPTS, GROUND_TRUTH, answer
from evaluation           import full_evaluation_report

random.seed(42)
np.random.seed(42)

OUT      = "results"
N_ROUNDS = 15   # quiz answers per concept

PROFILE_COLORS = {
    "strong":        "#16A34A",
    "struggling":    "#EA580C",
    "misconception": "#7C3AED",
}
MODEL_COLORS = {
    "NARS":       "#2563EB",
    "BKT":        "#EA580C",
    "Rule-Based": "#6B7280",
}

os.makedirs(OUT, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def run_one(ModelClass, profile, n_rounds=N_ROUNDS):
    """
    Run a single model on one student profile.
    Returns (model_instance, series_by_concept).
    series_by_concept[concept] = list of belief values after each observation.
    """
    model  = ModelClass(CONCEPTS)
    series = {c: [] for c in CONCEPTS}

    for concept in CONCEPTS:
        for _ in range(n_rounds):
            correct = answer(profile, concept)
            model.update(concept, correct)
            # snapshot belief for this concept right after the update
            if hasattr(model, "belief"):
                val = model.belief(concept)
                # NARS returns (f, c) tuple; extract expectation
                if isinstance(val, tuple):
                    f, c = val
                    val  = c * (f - 0.5) + 0.5
            series[concept].append(val)

    return model, series


# ── run everything ────────────────────────────────────────────────────────────

all_results = {}

for profile in ["strong", "struggling", "misconception"]:
    gt = GROUND_TRUTH[profile]
    profile_results = {}

    for label, ModelClass in [("NARS", NARSStudentModel),
                               ("BKT",  BKTStudentModel),
                               ("Rule-Based", RuleBasedStudentModel)]:
        model, series = run_one(ModelClass, profile)
        final = model.knowledge_state()
        report = full_evaluation_report(label, series, final, gt)
        profile_results[label] = {
            "metrics": report,
            "final":   final,
            "series":  series,
        }
        print(f"[{profile:14s}] {label:12s}  MAE={report['mae']:.3f}  "
              f"conv={report['convergence']:.1f}  "
              f"misc={report['misconception']:.2f}  "
              f"stab={report['stability']:.4f}")

    all_results[profile] = profile_results


# ── save results.json (no series — too large) ────────────────────────────────

json_out = {}
for profile, pr in all_results.items():
    json_out[profile] = {}
    for model_label, data in pr.items():
        json_out[profile][model_label] = {
            "metrics": data["metrics"],
            "final":   data["final"],
        }

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(json_out, f, indent=2)
print("\nSaved results.json")


# ── Figure 1 — convergence curves for fraction_addition ──────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.suptitle("Fig 1 — Convergence on fraction_addition", fontsize=13, fontweight="bold")

concept = "fraction_addition"
for ax, profile in zip(axes, ["strong", "struggling", "misconception"]):
    gt_line = GROUND_TRUTH[profile][concept]
    ax.axhline(gt_line, color="black", linestyle="--", linewidth=1.4, label="Ground truth")
    for label in ["NARS", "BKT", "Rule-Based"]:
        series = all_results[profile][label]["series"][concept]
        ax.plot(series, color=MODEL_COLORS[label], linewidth=1.8, label=label)
    ax.set_title(profile.capitalize(), fontsize=11)
    ax.set_xlabel("Observations", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Belief (expectation)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_convergence.png"), dpi=150)
plt.close()
print("Saved fig1_convergence.png")


# ── Figure 2 — bar chart of all four metrics across profiles ─────────────────

metrics_labels = ["mae", "convergence", "misconception", "stability"]
metric_titles  = ["MAE ↓", "Convergence speed ↓", "Misconception accuracy ↑", "Stability (σ) ↓"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Fig 2 — Metric comparison across all profiles", fontsize=13, fontweight="bold")

for ax, metric, title in zip(axes.flat, metrics_labels, metric_titles):
    profiles = ["strong", "struggling", "misconception"]
    x        = np.arange(len(profiles))
    width    = 0.25

    for i, label in enumerate(["NARS", "BKT", "Rule-Based"]):
        vals = [all_results[p][label]["metrics"][metric] for p in profiles]
        ax.bar(x + i * width, vals, width, label=label,
               color=MODEL_COLORS[label], alpha=0.85)

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.capitalize() for p in profiles], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_metrics.png"), dpi=150)
plt.close()
print("Saved fig2_metrics.png")


# ── Figure 3 — NARS vs BKT on misconception student, all concepts ────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Fig 3 — NARS vs BKT on misconception student (per concept)", fontsize=12)

for ax, concept in zip(axes.flat, CONCEPTS):
    gt_line = GROUND_TRUTH["misconception"][concept]
    ax.axhline(gt_line, color="black", linestyle="--", linewidth=1.3, label="Ground truth")
    for label in ["NARS", "BKT"]:
        s = all_results["misconception"][label]["series"][concept]
        ax.plot(s, color=MODEL_COLORS[label], linewidth=1.6, label=label)
    ax.set_title(concept.replace("_", " "), fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_misconception_concepts.png"), dpi=150)
plt.close()
print("Saved fig3_misconception_concepts.png")


# ── Figure 4 — heatmap of final beliefs vs ground truth ──────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Fig 4 — Final belief heatmap vs ground truth", fontsize=12)

for ax, profile in zip(axes, ["strong", "struggling", "misconception"]):
    gt = GROUND_TRUTH[profile]
    rows  = ["Ground truth", "NARS", "BKT", "Rule-Based"]
    data  = []
    data.append([gt[c] for c in CONCEPTS])
    for label in ["NARS", "BKT", "Rule-Based"]:
        final = all_results[profile][label]["final"]
        data.append([final[c] for c in CONCEPTS])

    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(CONCEPTS)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CONCEPTS], fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(profile.capitalize(), fontsize=10)
    for i in range(len(rows)):
        for j in range(len(CONCEPTS)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")

plt.colorbar(im, ax=axes[-1], fraction=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_misconception_heatmap.png"), dpi=150)
plt.close()
print("Saved fig4_misconception_heatmap.png")

print("\nDone. All outputs are in the results/ folder.")
