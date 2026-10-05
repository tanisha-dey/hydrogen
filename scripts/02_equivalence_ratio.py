import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Hydrogen-Air Combustion
# Equivalence Ratio Study
# --------------------------------------------------

# Equivalence ratios to investigate
phi_values = np.array([
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.5
])

# Initial conditions
temperature = 300.0
pressure = ct.one_atm

results = []


# --------------------------------------------------
# Run simulations
# --------------------------------------------------

for phi in phi_values:

    # Load gas mechanism
    gas = ct.Solution("gri30.yaml")

    # Set hydrogen-air mixture
    gas.set_equivalence_ratio(
        phi=phi,
        fuel="H2",
        oxidizer="O2:1, N2:3.76"
    )

    # Set initial conditions
    gas.TP = temperature, pressure

    # Equilibrium calculation
    gas.equilibrate("HP")

    # Store results
    results.append({
        "Equivalence Ratio": phi,
        "Flame Temperature (K)": gas.T
    })


# --------------------------------------------------
# Create results table
# --------------------------------------------------

df = pd.DataFrame(results)

print("\nHYDROGEN-AIR EQUIVALENCE RATIO STUDY")
print("=" * 50)
print(df.to_string(index=False))
print("=" * 50)


# --------------------------------------------------
# Save results
# --------------------------------------------------

df.to_csv(
    "results/equivalence_ratio_results.csv",
    index=False
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Equivalence Ratio"],
    df["Flame Temperature (K)"],
    marker="o"
)

plt.xlabel("Equivalence Ratio (φ)")
plt.ylabel("Adiabatic Flame Temperature (K)")
plt.title("Hydrogen-Air Combustion: Flame Temperature vs Equivalence Ratio")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/flame_temperature_vs_equivalence_ratio.png",
    dpi=300
)

plt.show()