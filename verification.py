from dataclasses import replace

from assembly_sim import ModelConfig, run_simulation

T_MOD = 100_000
SEED = 1


def expected_loads(c: ModelConfig):
    rate = c.batch_size / c.mean_interarrival
    pairs = rate * min(c.p_processed, 1 - c.p_processed)
    attempts = pairs / (1 - c.p_defect)
    d1_flow = rate * c.p_processed + c.p_defect * attempts
    return (min(1.0, c.t_preprocess * d1_flow),
            min(1.0, c.t_assembly * attempts),
            min(1.0, c.mean_regulation * pairs))


SCENARIOS = [
    ("1. Початкові значення", ModelConfig()),
    ("2. t_надх = 20", ModelConfig(mean_interarrival=20.0)),
    ("3. t_обрD1 = 10", ModelConfig(t_preprocess=10.0)),
    ("4. t_обрD3 = 9", ModelConfig(t_assembly=9.0)),
    ("5. t_обрD4 = 12", ModelConfig(mean_regulation=12.0)),
    ("6. p_бр = 0", ModelConfig(p_defect=0.0)),
    ("7. p_бр = 0,10", ModelConfig(p_defect=0.10)),
    ("8. n_д = 2", ModelConfig(batch_size=2)),
    ("9. p_обр = 0 (усі деталі без обробки)", ModelConfig(p_processed=0.0)),
    ("10. p_обр = 1 (усі деталі оброблюються)", ModelConfig(p_processed=1.0)),
]


def main() -> None:
    print(f"Час моделювання T_mod = {T_MOD}, seed = {SEED}\n")
    print("Вихідні змінні")
    print(f"{'Сценарій':<42}{'L_Q1':>7}{'L_Q21':>8}{'L_Q22':>8}{'L_Q4':>7}{'N':>8}"
          f"{'ρD1':>7}{'ρD3':>7}{'ρD4':>7}{'збірок':>8}{'браку':>7}{'баланс':>8}")
    results = []
    for name, cfg in SCENARIOS:
        r = run_simulation(cfg, T_MOD, seed=SEED)
        s = r.statistics
        balanced = (r.parts_arrived - r.assemblies - r.n_processed == r.parts_on_site_end)
        results.append((name, cfg, r))
        print(f"{name:<42}{s['L_Q1'].mean():>7.3f}{s['L_Q21'].mean():>8.2f}"
              f"{s['L_Q22'].mean():>8.2f}{s['L_Q4'].mean():>7.3f}{s['N'].mean():>8.2f}"
              f"{s['rho_D1'].mean():>7.3f}{s['rho_D3'].mean():>7.3f}{s['rho_D4'].mean():>7.3f}"
              f"{r.assemblies:>8}{r.defects:>7}{('так' if balanced else 'НІ'):>8}")

    trace = []
    run_simulation(ModelConfig(), 200.0, seed=SEED, trace=trace)
    fragment = [line for line in trace if 111.0 <= float(line[2:line.index("|")]) <= 196.0]
    with open("event_log_fragment.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(fragment) + "\n")
    print(f"\nФрагмент протоколу подій ({len(fragment)} рядків) записано у event_log_fragment.txt")

    print("\nПорівняння завантажень з аналітичними значеннями (імітація / аналітично)")
    for name, cfg, r in results:
        s = r.statistics
        e1, e3, e4 = expected_loads(cfg)
        print(f"{name:<42}"
              f"{s['rho_D1'].mean():.3f} / {e1:.3f}   "
              f"{s['rho_D3'].mean():.3f} / {e3:.3f}   "
              f"{s['rho_D4'].mean():.3f} / {e4:.3f}")


if __name__ == "__main__":
    main()