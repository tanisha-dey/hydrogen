import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LEAN vs STOICHIOMETRIC vs RICH
# TRANSIENT HYDROGEN-AIR COMBUSTION
# ============================================================

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

T0 = 1000.0          # Initial temperature [K]
P0 = ct.one_atm     # Initial pressure [Pa]

# Three representative combustion conditions
cases = {
    "Lean": 0.7,
    "Stoichiometric": 1.0,
    "Rich": 1.3
}

# Simulation settings
end_time = 0.002     # 2 milliseconds
n_steps = 5000


# ------------------------------------------------------------
# Store final results
# ------------------------------------------------------------

summary = []

# Store complete transient data for plotting
all_data = {}


# ============================================================
# RUN SIMULATIONS
# ============================================================

for case_name, phi in cases.items():

    print(f"Running {case_name} case: phi = {phi}")

    # --------------------------------------------------------
    # Create gas
    # --------------------------------------------------------

    gas = ct.Solution("gri30.yaml")

    gas.set_equivalence_ratio(
        phi=phi,
        fuel="H2",
        oxidizer="O2:1, N2:3.76"
    )

    gas.TP = T0, P0


    # --------------------------------------------------------
    # Create constant-volume reactor
    # --------------------------------------------------------

    reactor = ct.IdealGasReactor(
        gas,
        clone=False
    )

    network = ct.ReactorNet([reactor])

    # Numerical tolerances
    network.rtol = 1e-8
    network.atol = 1e-15


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = []
    temperatures = []
    pressures = []

    h2_values = []
    o2_values = []
    h2o_values = []
    oh_values = []


    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    time_values = np.linspace(
        0,
        end_time,
        n_steps
    )

    for time in time_values:

        network.advance(time)

        times.append(time)

        temperatures.append(
            reactor.T
        )

        pressures.append(
            reactor.phase.P / ct.one_atm
        )

        h2_values.append(
            reactor.phase["H2"].X[0]
        )

        o2_values.append(
            reactor.phase["O2"].X[0]
        )

        h2o_values.append(
            reactor.phase["H2O"].X[0]
        )

        oh_values.append(
            reactor.phase["OH"].X[0]
        )


    # Convert to NumPy arrays

    times = np.array(times)
    temperatures = np.array(temperatures)
    pressures = np.array(pressures)

    h2_values = np.array(h2_values)
    o2_values = np.array(o2_values)
    h2o_values = np.array(h2o_values)
    oh_values = np.array(oh_values)


    # --------------------------------------------------------
    # Calculate ignition delay
    # --------------------------------------------------------

    dTdt = np.gradient(
        temperatures,
        times
    )

    ignition_index = np.argmax(dTdt)

    ignition_delay = times[ignition_index]


    # --------------------------------------------------------
    # Calculate maximum values
    # --------------------------------------------------------

    max_temperature = np.max(
        temperatures
    )

    max_pressure = np.max(
        pressures
    )

    max_oh = np.max(
        oh_values
    )


    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary.append({

        "Case": case_name,

        "Equivalence Ratio": phi,

        "Ignition Delay (ms)":
            ignition_delay * 1000,

        "Maximum Temperature (K)":
            max_temperature,

        "Maximum Pressure (atm)":
            max_pressure,

        "Maximum OH":
            max_oh
    })


    # --------------------------------------------------------
    # Store transient data
    # --------------------------------------------------------

    all_data[case_name] = {

        "phi": phi,

        "time": times,

        "temperature": temperatures,

        "pressure": pressures,

        "H2": h2_values,

        "O2": o2_values,

        "H2O": h2o_values,

        "OH": oh_values
    }


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

df = pd.DataFrame(summary)


print("\n")
print("=" * 80)
print("LEAN vs STOICHIOMETRIC vs RICH")
print("TRANSIENT HYDROGEN-AIR COMBUSTION")
print("=" * 80)

print(
    df.to_string(index=False)
)

print("=" * 80)


# ============================================================
# SAVE SUMMARY
# ============================================================

df.to_csv(
    "results/lean_stoich_rich_summary.csv",
    index=False
)


# ============================================================
# PLOT 1 — TEMPERATURE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

for case_name, data in all_data.items():

    plt.plot(
        data["time"] * 1000,
        data["temperature"],
        label=f"{case_name} (φ={data['phi']})"
    )

plt.xlabel("Time (ms)")
plt.ylabel("Temperature (K)")
plt.title(
    "Hydrogen-Air Combustion: Temperature Comparison"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/lean_stoich_rich_temperature.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 2 — PRESSURE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

for case_name, data in all_data.items():

    plt.plot(
        data["time"] * 1000,
        data["pressure"],
        label=f"{case_name} (φ={data['phi']})"
    )

plt.xlabel("Time (ms)")
plt.ylabel("Pressure (atm)")
plt.title(
    "Hydrogen-Air Combustion: Pressure Comparison"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/lean_stoich_rich_pressure.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 3 — HYDROGEN CONSUMPTION
# ============================================================

plt.figure(figsize=(9, 5))

for case_name, data in all_data.items():

    plt.plot(
        data["time"] * 1000,
        data["H2"],
        label=f"{case_name} (φ={data['phi']})"
    )

plt.xlabel("Time (ms)")
plt.ylabel("H₂ Mole Fraction")
plt.title(
    "Hydrogen Consumption for Different Equivalence Ratios"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/lean_stoich_rich_h2.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 4 — WATER FORMATION
# ============================================================

plt.figure(figsize=(9, 5))

for case_name, data in all_data.items():

    plt.plot(
        data["time"] * 1000,
        data["H2O"],
        label=f"{case_name} (φ={data['phi']})"
    )

plt.xlabel("Time (ms)")
plt.ylabel("H₂O Mole Fraction")
plt.title(
    "Water Formation for Different Equivalence Ratios"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/lean_stoich_rich_h2o.png",
    dpi=300
)

plt.show()