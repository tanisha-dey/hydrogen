import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# TRANSIENT IGNITION vs EQUIVALENCE RATIO
# --------------------------------------------------

T0 = 1000.0
P0 = ct.one_atm

phi_values = [
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3
]

results = []


for phi in phi_values:

    print(f"Running phi = {phi:.1f}")

    # ----------------------------------------------
    # Create gas
    # ----------------------------------------------

    gas = ct.Solution("gri30.yaml")

    gas.set_equivalence_ratio(
        phi=phi,
        fuel="H2",
        oxidizer="O2:1, N2:3.76"
    )

    gas.TP = T0, P0


    # ----------------------------------------------
    # Reactor
    # ----------------------------------------------

    reactor = ct.IdealGasReactor(gas)

    network = ct.ReactorNet([reactor])


    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    times = []
    temperatures = []

    end_time = 0.01
    n_steps = 2000

    time_values = np.linspace(
        0,
        end_time,
        n_steps
    )

    for time in time_values:

        network.advance(time)

        times.append(time)
        temperatures.append(reactor.T)


    times = np.array(times)
    temperatures = np.array(temperatures)


    # ----------------------------------------------
    # Calculate dT/dt
    # ----------------------------------------------

    dTdt = np.gradient(
        temperatures,
        times
    )


    # Find ignition point
    ignition_index = np.argmax(dTdt)

    ignition_delay = times[ignition_index]

    max_temperature = np.max(
        temperatures
    )


    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append({
        "Equivalence Ratio": phi,
        "Ignition Delay (s)": ignition_delay,
        "Maximum Temperature (K)": max_temperature
    })


# --------------------------------------------------
# Results table
# --------------------------------------------------

df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("HYDROGEN IGNITION vs EQUIVALENCE RATIO")
print("=" * 70)

print(
    df.to_string(index=False)
)

print("=" * 70)


# --------------------------------------------------
# Save results
# --------------------------------------------------

df.to_csv(
    "results/ignition_vs_equivalence_ratio.csv",
    index=False
)


# --------------------------------------------------
# Plot ignition delay
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Equivalence Ratio"],
    df["Ignition Delay (s)"],
    marker="o"
)

plt.xlabel("Equivalence Ratio (φ)")
plt.ylabel("Ignition Delay (s)")
plt.title(
    "Hydrogen Ignition Delay vs Equivalence Ratio"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/ignition_delay_vs_equivalence_ratio.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Plot maximum temperature
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Equivalence Ratio"],
    df["Maximum Temperature (K)"],
    marker="o"
)

plt.xlabel("Equivalence Ratio (φ)")
plt.ylabel("Maximum Temperature (K)")
plt.title(
    "Maximum Temperature vs Equivalence Ratio"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/max_temperature_vs_equivalence_ratio.png",
    dpi=300
)

plt.show()