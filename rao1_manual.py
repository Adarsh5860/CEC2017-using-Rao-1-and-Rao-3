"""
Rao-1 Algorithm — Manual Values Step-by-Step Verification
=========================================================
Edit the values below (population, random values, objective function)
and run this script to see each step of Rao-1 in detail.

Usage:
    python rao1_manual.py
"""

import numpy as np

# ╔══════════════════════════════════════════════════════════════════╗
# ║              EDIT YOUR MANUAL VALUES BELOW                      ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 1. Initial Population ──
# Each row is one candidate solution [x1, x2]
# Add or remove rows to change population size
population = np.array([
    [20.0,  30.0],     # Candidate 1
    [10.0, -50.0],     # Candidate 2
    [-30.0, 40.0],     # Candidate 3
    [50.0, -20.0],     # Candidate 4
    [60.0,  10.0],     # Candidate 5
], dtype=float)

# ── 2. Random values (r1) for each candidate in each iteration ──
# Shape: [num_iterations][num_candidates] — each entry is [r1_for_x1, r1_for_x2]
# Iteration 1: one r1 per candidate
# Iteration 2: one r1 per candidate
random_values = [
    # ── Iteration 1 ──
    [
        [0.35, 0.72],   # r1 for Candidate 1
        [0.50, 0.40],   # r1 for Candidate 2
        [0.15, 0.88],   # r1 for Candidate 3
        [0.60, 0.25],   # r1 for Candidate 4
        [0.90, 0.10],   # r1 for Candidate 5
    ],
    # ── Iteration 2 ──
    [
        [0.45, 0.30],   # r1 for Candidate 1
        [0.80, 0.55],   # r1 for Candidate 2
        [0.20, 0.65],   # r1 for Candidate 3
        [0.70, 0.40],   # r1 for Candidate 4
        [0.33, 0.77],   # r1 for Candidate 5
    ],
]

# ── 3. Number of iterations to run ──
NUM_ITERATIONS = 2

# ── 4. Objective Function ──
# Change this to match the function you're verifying
# Default: Sphere function f(x) = sum(xi^2)
def objective_function(x):
    """Sphere function: f(x) = x1^2 + x2^2"""
    return np.sum(x ** 2)


# ╔══════════════════════════════════════════════════════════════════╗
# ║              DO NOT EDIT BELOW THIS LINE                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_rao1_manual():
    pop = population.copy()
    pop_size = len(pop)
    D = pop.shape[1]

    print("=" * 70)
    print("  RAO-1 ALGORITHM — MANUAL STEP-BY-STEP VERIFICATION")
    print("=" * 70)

    # ── Compute initial fitness ──
    fitness = np.array([objective_function(pop[i]) for i in range(pop_size)])

    print(f"\n{'─' * 70}")
    print("  INITIAL POPULATION")
    print(f"{'─' * 70}")
    print(f"  {'Candidate':<12} {'Values':<30} {'f(x)':<15}")
    print(f"  {'─'*12} {'─'*30} {'─'*15}")
    for i in range(pop_size):
        vals = ", ".join(f"{v:>8.4f}" for v in pop[i])
        print(f"  C{i+1:<11} [{vals}]  {fitness[i]:<15.6f}")

    best_idx = int(np.argmin(fitness))
    worst_idx = int(np.argmax(fitness))
    print(f"\n  ✦ Best:  C{best_idx+1} → f(x) = {fitness[best_idx]:.6f}")
    print(f"  ✦ Worst: C{worst_idx+1} → f(x) = {fitness[worst_idx]:.6f}")

    # ── Run iterations ──
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'━' * 70}")
        print(f"  ITERATION {iteration + 1}")
        print(f"{'━' * 70}")

        best_idx = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))
        best = pop[best_idx].copy()
        worst = pop[worst_idx].copy()

        best_str = ", ".join(f"{v:>8.4f}" for v in best)
        worst_str = ", ".join(f"{v:>8.4f}" for v in worst)
        print(f"\n  X_best  (C{best_idx+1}): [{best_str}]  f = {fitness[best_idx]:.6f}")
        print(f"  X_worst (C{worst_idx+1}): [{worst_str}]  f = {fitness[worst_idx]:.6f}")

        diff = best - worst
        diff_str = ", ".join(f"{v:>8.4f}" for v in diff)
        print(f"  (X_best - X_worst):  [{diff_str}]")

        for i in range(pop_size):
            x = pop[i].copy()
            r1 = np.array(random_values[iteration][i])

            print(f"\n  ┌─ Candidate C{i+1} ─────────────────────────────────────")
            x_str = ", ".join(f"{v:>8.4f}" for v in x)
            r1_str = ", ".join(f"{v:>8.4f}" for v in r1)
            print(f"  │  x     = [{x_str}]")
            print(f"  │  r1    = [{r1_str}]")

            # Rao-1 formula: x_new = x + r1 * (best - worst)
            perturbation = r1 * diff
            x_new = x + perturbation

            pert_str = ", ".join(f"{v:>8.4f}" for v in perturbation)
            xnew_str = ", ".join(f"{v:>8.4f}" for v in x_new)
            print(f"  │")
            print(f"  │  Formula: x_new = x + r1 × (X_best - X_worst)")
            print(f"  │  r1 × (X_best - X_worst) = [{pert_str}]")
            print(f"  │  x_new = [{xnew_str}]")

            f_old = fitness[i]
            f_new = objective_function(x_new)

            print(f"  │")
            print(f"  │  f(x_old) = {f_old:.6f}")
            print(f"  │  f(x_new) = {f_new:.6f}")

            if f_new <= f_old:
                pop[i] = x_new
                fitness[i] = f_new
                print(f"  │  ✅ ACCEPTED (f_new ≤ f_old) → Update C{i+1}")
            else:
                print(f"  │  ❌ REJECTED (f_new > f_old) → Keep old C{i+1}")
            print(f"  └──────────────────────────────────────────────────")

        # ── Summary after iteration ──
        print(f"\n  {'─' * 60}")
        print(f"  POPULATION AFTER ITERATION {iteration + 1}")
        print(f"  {'─' * 60}")
        print(f"  {'Candidate':<12} {'Values':<30} {'f(x)':<15}")
        print(f"  {'─'*12} {'─'*30} {'─'*15}")
        for i in range(pop_size):
            vals = ", ".join(f"{v:>8.4f}" for v in pop[i])
            print(f"  C{i+1:<11} [{vals}]  {fitness[i]:<15.6f}")

        best_idx = int(np.argmin(fitness))
        print(f"\n  ✦ Best after Iteration {iteration+1}: C{best_idx+1} → f(x) = {fitness[best_idx]:.6f}")

    # ── Final Summary ──
    print(f"\n{'═' * 70}")
    print("  FINAL RESULT")
    print(f"{'═' * 70}")
    best_idx = int(np.argmin(fitness))
    best_vals = ", ".join(f"{v:>8.4f}" for v in pop[best_idx])
    print(f"  Best Solution: C{best_idx+1} = [{best_vals}]")
    print(f"  Best Fitness:  f(x) = {fitness[best_idx]:.6f}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    run_rao1_manual()
