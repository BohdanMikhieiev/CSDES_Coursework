import json
import math
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as ex
from assembly_sim import ModelConfig, run_simulation

BASE = ModelConfig()
MODIFIED = replace(BASE, balanced_types=True)
SYSTEMS = (("базова", BASE), ("модифікована", MODIFIED))

PILOT_SEEDS = range(101, 106)
T_MAX = 60_000
CHECKPOINTS = list(range(1_000, T_MAX + 1, 1_000))
EPS_TRANSIENT = {"L_Q1": 0.1, "L_Q21": 0.1, "L_Q22": 0.1, "L_Q4": 0.1, "N": 0.2}
WARMUP = 40_000
T_OBS = 40_000
EPS_RESULT = 0.05
BETA = 0.95
SIGMA_RUNS = 20
HORIZONS = ((1_000, 100), (10_000, 100), (100_000, 60), (1_000_000, 20))
CAUSE_HORIZON, CAUSE_RUNS = 10_000, 60
SAMPLE_SPACING, SAMPLE_SIZE = 500, 4_000

FIG_DIR = "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
results = {}


def save_figure(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


RESPONSES = (
    ("L_Q1", "L_Q1, деталей"),
    ("L_Q21", "L_Q21, деталей"),
    ("L_Q22", "L_Q22, деталей"),
    ("L_Q4", "L_Q4, деталей"),
    ("N", "N, деталей"),
)


def draw_transient_figure(pilots, transient, checkpoints, filename="fig_4_1_transient.png"):
    fig, axes = plt.subplots(len(RESPONSES), 2, figsize=(8.0, 13.0))
    for row, (name, ylabel) in enumerate(RESPONSES):
        for col, label in enumerate(("базова", "модифікована")):
            ax = axes[row][col]
            for i, p in enumerate(pilots[label], start=1):
                ax.plot(checkpoints, [p[t][name] for t in checkpoints], lw=1,
                        label=f"запуск {i}")
            t_trans = transient[label][name]["T"]
            if t_trans is not None:
                ax.axvline(t_trans, color="black", ls=":", lw=1, label=f"T_пер = {t_trans}")
                title = f"{label} система"
            else:
                title = f"{label} система (T_пер не досягнуто)"
            if all(p[t][name] == 0 for p in pilots[label] for t in checkpoints):
                ax.set_ylim(-0.1, 1.0)          # черга завжди порожня: щоб вісь не розтягувалась
            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylabel)
            if row == len(RESPONSES) - 1:
                ax.set_xlabel("час моделювання t, умовних одиниць")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    save_figure(fig, filename)


print("Перехідний період")
pilots = {}
results["transient"] = {}
for label, cfg in SYSTEMS:
    pilots[label] = ex.run_pilots(cfg, PILOT_SEEDS, T_MAX, CHECKPOINTS)
    results["transient"][label] = {}
    for name, eps in EPS_TRANSIENT.items():
        t = ex.transient_time(pilots[label], name, eps, CHECKPOINTS)
        final = ex.spread(pilots[label], name, T_MAX)
        results["transient"][label][name] = {"eps": eps, "T": t, "spread_at_end": final}
        print(f"{label:13s} {name:6s} eps={eps}: T = {t}, розкид при t={T_MAX}: {final:.3f}")

draw_transient_figure(pilots, results["transient"], CHECKPOINTS)

print("\nКількість замірів")
sigma_runs = {}
results["replications"] = {}
for label, cfg in SYSTEMS:
    runs = ex.replicate(cfg, SIGMA_RUNS, WARMUP, T_OBS, seed0=1_000)
    sigma_runs[label] = {name: ex.std(ex.run_means(runs, name)) for name in ex.OUTPUTS}
    results["replications"][label] = {}
    for name in ex.OUTPUTS:
        s = sigma_runs[label][name]
        n_req = ex.required_replications(s, EPS_RESULT, BETA)
        results["replications"][label][name] = {"sigma": s, "n": n_req}
        print(f"{label:13s} {name:7s} σ = {s:8.4f}  n = {n_req}")
n_stable = max(results["replications"]["модифікована"][nm]["n"] for nm in ex.OUTPUTS)
N_RUNS = max(n_stable, 20)
print(f"Прийнято кількість замірів n = {N_RUNS}")
results["n_runs"] = N_RUNS

print("\nЗалежність середнього N від часу моделювання")
results["horizon"] = {}
horizon_stats = {}
for label, cfg in SYSTEMS:
    results["horizon"][label] = []
    horizon_stats[label] = []
    for t_h, n_h in HORIZONS:
        runs = [run_simulation(cfg, t_h, seed=50_000 + k) for k in range(n_h)]
        row = {"T": t_h, "runs": n_h}
        for name in ("L_Q21", "L_Q22", "N"):
            vals = ex.run_means(runs, name)
            row[name] = {"mean": ex.mean(vals), "sigma": ex.std(vals)}
        results["horizon"][label].append(row)
        horizon_stats[label].append(row)
        print(f"{label:13s} T={t_h:8d} n={n_h:3d}  N = {row['N']['mean']:8.2f} (σ {row['N']['sigma']:7.2f})"
              f"  L_Q21 = {row['L_Q21']['mean']:7.2f}  L_Q22 = {row['L_Q22']['mean']:7.2f}")

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ts = [h[0] for h in HORIZONS]
for label, marker in (("базова", "o"), ("модифікована", "s")):
    ax.errorbar(ts, [r["N"]["mean"] for r in horizon_stats[label]],
                yerr=[r["N"]["sigma"] for r in horizon_stats[label]], marker=marker,
                capsize=3, label=f"{label} система")
ref = horizon_stats["базова"][1]["N"]["mean"]
ax.plot(ts, [ref * math.sqrt(t / ts[1]) for t in ts], "k--", lw=1, label="пропорційно √t")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("час моделювання T, умовних одиниць")
ax.set_ylabel("середня кількість деталей N, шт.")
ax.grid(alpha=0.3, which="both")
ax.legend()
save_figure(fig, "fig_4_2_horizon.png")

print("\nПричини накопичення (час моделювання", CAUSE_HORIZON, ")")
cause_scenarios = [
    ("базова система", BASE),
    ("без браку (p_бр = 0)", replace(BASE, p_defect=0.0)),
    ("швидша обробка D1 (t_обрD1 = 3,5)", replace(BASE, t_preprocess=3.5)),
    ("менше навантаження (t_надх = 60)", replace(BASE, mean_interarrival=60.0)),
    ("чергування типів деталей", MODIFIED),
]
results["causes"] = []
for label, cfg in cause_scenarios:
    runs = [run_simulation(cfg, CAUSE_HORIZON, seed=70_000 + k) for k in range(CAUSE_RUNS)]
    row = {"scenario": label}
    for name in ("L_Q21", "L_Q22", "N", "rho_D1", "rho_D3"):
        vals = ex.run_means(runs, name)
        row[name] = {"mean": ex.mean(vals), "sigma": ex.std(vals)}
    results["causes"].append(row)
    print(f"{label:38s} N = {row['N']['mean']:7.2f} (σ {row['N']['sigma']:6.2f})  "
          f"L_Q21 = {row['L_Q21']['mean']:6.2f}  L_Q22 = {row['L_Q22']['mean']:6.2f}  "
          f"ρD1 = {row['rho_D1']['mean']:.3f}  ρD3 = {row['rho_D3']['mean']:.3f}")

print("\nРезультати експериментів:", N_RUNS, "запусків, перехідний період", WARMUP,
      ", збір", T_OBS, "")
final_runs = {label: ex.replicate(cfg, N_RUNS, WARMUP, T_OBS, seed0=10_000)
              for label, cfg in SYSTEMS}
results["final"] = {}
for label, _ in SYSTEMS:
    results["final"][label] = {}
    for name in ex.OUTPUTS:
        vals = ex.run_means(final_runs[label], name)
        sig_time = ex.mean([r.statistics[name].std() for r in final_runs[label]])
        s = ex.std(vals)
        results["final"][label][name] = {"mean": ex.mean(vals), "sigma_runs": s,
                                         "eps": ex.achieved_accuracy(s, N_RUNS, BETA),
                                         "sigma_time": sig_time}
        print(f"{label:13s} {name:7s} середнє = {ex.mean(vals):9.4f}  σ(запусків) = {s:8.4f}  "
              f"ε = {ex.achieved_accuracy(s, N_RUNS, BETA):8.4f}  σ(в часі) = {sig_time:8.4f}")

print("\nДисперсійний аналіз (фактор - модифікація)")
results["anova"] = {}
for name in ex.OUTPUTS:
    groups = [ex.run_means(final_runs[label], name) for label, _ in SYSTEMS]
    a = ex.anova_one_way(groups)
    results["anova"][name] = {"D_f": a.d_factor, "D_r": a.d_residual, "F": a.f_value,
                              "F_crit": a.f_critical, "df": [a.df_factor, a.df_residual],
                              "significant": bool(a.significant)}
    print(f"{name:7s} D_ф = {a.d_factor:12.4f}  D_з = {a.d_residual:10.5f}  F = {a.f_value:12.2f}  "
          f"F_крит = {a.f_critical:.3f}  вплив {'суттєвий' if a.significant else 'не виявлено'}")

# ------------------------------------------------ 6. ідентифікація закону розподілу
print("\nІдентифікація закону розподілу черг")
cases = (("базова", BASE, "q1", "Q1"), ("базова", BASE, "q4", "Q4"),
         ("модифікована", MODIFIED, "q1", "Q1"), ("модифікована", MODIFIED, "q21", "Q21"),
         ("модифікована", MODIFIED, "q4", "Q4"))
results["chi2"] = []
fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))
for ax, (label, cfg, attr, title) in zip(axes.flat, cases):
    sample = ex.sample_states(cfg, attr, SAMPLE_SIZE, WARMUP, SAMPLE_SPACING, seed=90_000)
    pmf = ex.fit_zero_modified_geometric(sample)
    test = ex.chi_square_test(sample, pmf, n_params=2)
    p0 = sum(1 for v in sample if v == 0) / len(sample)
    results["chi2"].append({"system": label, "queue": title, "mean": ex.mean(sample),
                            "sigma": ex.std(sample), "p0": p0, "chi2": test.chi2, "df": test.df,
                            "critical": test.critical, "accepted": bool(test.accepted),
                            "rows": [[r[0], r[1], r[2]] for r in test.rows]})
    print(f"{label:13s} {title:4s} середнє {ex.mean(sample):.3f}  σ {ex.std(sample):.3f}  "
          f"χ² = {test.chi2:.2f} (df {test.df}, крит. {test.critical:.2f}) "
          f"{'гіпотезу не відхилено' if test.accepted else 'гіпотезу відхилено'}")
    top = min(max(sample), 8)
    xs = list(range(top + 1))
    ax.bar([x - 0.2 for x in xs], [sum(1 for v in sample if v == x) / len(sample) for x in xs],
           width=0.4, label="спостережено")
    ax.bar([x + 0.2 for x in xs], [pmf(x) for x in xs], width=0.4, label="гіпотеза")
    ax.set_title(f"{label}, {title}")
    ax.set_xlabel("довжина черги, деталей")
    ax.set_ylabel("частота")
axes.flat[0].legend()
axes.flat[5].axis("off")
fig.tight_layout()
save_figure(fig, "fig_4_3_distributions.png")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)