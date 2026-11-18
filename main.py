from assembly_sim import ModelConfig, run_simulation


def main() -> None:
    config = ModelConfig()
    t_mod = 10_000.0
    result = run_simulation(config, t_mod, seed=1)
    stats = result.statistics

    print(f"Час моделювання: {t_mod:.0f} умовних одиниць часу (1 одиниця = 1 хв)")
    print(f"Надійшло деталей: {result.parts_arrived}, зібрано виробів: {result.assemblies}, "
          f"браку: {result.defects}, залишило ділянку: {result.n_processed}\n")
    print(f"{'Змінна':<10}{'середнє':>10}{'σ':>10}")
    for name in stats.NAMES:
        print(f"{name:<10}{stats[name].mean():>10.3f}{stats[name].std():>10.3f}")


if __name__ == "__main__":
    main()