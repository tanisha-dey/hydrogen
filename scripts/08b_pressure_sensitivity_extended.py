import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EXTENDED PRESSURE SENSITIVITY STUDY
# HYDROGEN-AIR TRANSIENT IGNITION
# ============================================================

T0 = 1000.0
phi = 1.0

pressure_values_atm = [
    1,
    3,
    5,
    7,
    10
]

# Give the chemistry more time to ignite
end_time = 0.02       # 20 milliseconds
n_steps = 10000

results = []


# ============================================================
# RUN SIMULATIONS
# ============================================================

for pressure_atm in pressure_values_atm:

    print(f"Running pressure = {pressure_atm} atm")

    # --------------------------------------------------------
    # Gas
    # --------------------------------------------------------

    gas = ct.Solution("gri30.yaml")

    gas.set_equivalence_ratio(
        phi=phi,
        fuel="H2",
        oxidizer="O2:1, N2:3.76"
    )

    gas.TP = (
        T0,
        pressure_atm * ct.one_atm
    )


    # --------------------------------------------------------
    # Reactor
    # --------------------------------------------------------

    reactor = ct.IdealGasReactor(
        gas,
        clone=False
    )

    network = ct.ReactorNet([reactor])

    network.rtol = 1e-8
    network.atol = 1e-15


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = []
    temperatures = []
    pressures = []

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    for time in np.linspace(
        0,
        end_time,
        n_steps
    ):

        network.advance(time)

        times.append(time)

        temperatures.append(
            reactor.T
        )

        pressures.append(
            reactor.phase.P / ct.one_atm
        )


    times = np.array(times)
    temperatures = np.array(temperatures)
    pressures = np.array(pressures)


    # ========================================================
    # IGNITION DETECTION
    # ========================================================

    dTdt = np.gradient(
        temperatures,
        times
    )

    ignition_index = np.argmax(dTdt)

    # Temperature rise from initial state
    temperature_rise = (
        temperatures - T0
    )

    max_temperature = np.max(
        temperatures
    )

    # Define ignition as a significant rapid temperature rise
    # Here we use a 100 K rise as a simple preliminary criterion
    ignition_indices = np.where(
        temperature_rise >= 100.0
    )[0]


    if len(ignition_indices) > 0:

        first_ignition_index = ignition_indices[0]

        ignition_delay = (
            times[first_ignition_index]
        )

        ignition_status = "Detected"

    else:

        ignition_delay = np.nan

        ignition_status = "Not detected"


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Initial Pressure (atm)": pressure_atm,

        "Ignition Status":
            ignition_status,

        "Ignition Delay (ms)":
            (
                ignition_delay * 1000
                if not np.isnan(ignition_delay)
                else np.nan
            ),

        "Maximum Temperature (K)":
            max_temperature,

        "Maximum Pressure (atm)":
            np.max(pressures)
    })


# ============================================================
# RESULTS
# ============================================================

df = pd.DataFrame(results)

print("\n")
print("=" * 85)
print("EXTENDED PRESSURE SENSITIVITY STUDY")
print("=" * 85)

print(
    df.to_string(index=False)
)

print("=" * 85)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/pressure_sensitivity_extended.csv",
    index=False
)


# ============================================================
# PLOT — IGNITION DELAY
# ============================================================

plt.figure(figsize=(8, 5))

detected = df[
    df["Ignition Status"] == "Detected"
]

plt.plot(
    detected["Initial Pressure (atm)"],
    detected["Ignition Delay (ms)"],
    marker="o"
)

plt.xlabel("Initial Pressure (atm)")
plt.ylabel("Ignition Delay (ms)")
plt.title(
    "Hydrogen Ignition Delay vs Initial Pressure"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/ignition_delay_vs_pressure_extended.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT — MAXIMUM TEMPERATURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["Initial Pressure (atm)"],
    df["Maximum Temperature (K)"],
    marker="o"
)

plt.xlabel("Initial Pressure (atm)")
plt.ylabel("Maximum Temperature (K)")
plt.title(
    "Maximum Temperature vs Initial Pressure"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/max_temperature_vs_pressure_extended.png",
    dpi=300
)

plt.show()