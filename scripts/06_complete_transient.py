import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# COMPLETE TRANSIENT HYDROGEN-AIR COMBUSTION
# ============================================================

# -----------------------------
# Initial conditions
# -----------------------------

T0 = 1000.0                  # Initial temperature [K]
P0 = ct.one_atm              # Initial pressure [Pa]
phi = 1.0                    # Stoichiometric mixture

# Simulation time
end_time = 0.002             # seconds
n_steps = 5000


# ============================================================
# CREATE GAS MIXTURE
# ============================================================

gas = ct.Solution("gri30.yaml")

gas.set_equivalence_ratio(
    phi=phi,
    fuel="H2",
    oxidizer="O2:1, N2:3.76"
)

gas.TP = T0, P0


# ============================================================
# CREATE REACTOR
# ============================================================

reactor = ct.IdealGasReactor(gas)

network = ct.ReactorNet([reactor])

# Numerical tolerances
network.rtol = 1e-8
network.atol = 1e-15


# ============================================================
# STORAGE
# ============================================================

times = []
temperatures = []
pressures = []

h2_values = []
o2_values = []
h2o_values = []
oh_values = []


# ============================================================
# RUN TRANSIENT SIMULATION
# ============================================================

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
        reactor.thermo.P / ct.one_atm
    )

    h2_values.append(
        reactor.thermo["H2"].X[0]
    )

    o2_values.append(
        reactor.thermo["O2"].X[0]
    )

    h2o_values.append(
        reactor.thermo["H2O"].X[0]
    )

    oh_values.append(
        reactor.thermo["OH"].X[0]
    )


# Convert to NumPy arrays

times = np.array(times)
temperatures = np.array(temperatures)
pressures = np.array(pressures)
h2_values = np.array(h2_values)
o2_values = np.array(o2_values)
h2o_values = np.array(h2o_values)
oh_values = np.array(oh_values)


# ============================================================
# IGNITION DELAY
# ============================================================

dTdt = np.gradient(
    temperatures,
    times
)

ignition_index = np.argmax(dTdt)

ignition_delay = times[ignition_index]


# ============================================================
# IMPORTANT RESULTS
# ============================================================

max_temperature = np.max(temperatures)
max_pressure = np.max(pressures)

max_oh = np.max(oh_values)

final_h2 = h2_values[-1]
final_o2 = o2_values[-1]
final_h2o = h2o_values[-1]


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "Time (s)": times,

    "Temperature (K)": temperatures,

    "Pressure (atm)": pressures,

    "H2": h2_values,

    "O2": o2_values,

    "H2O": h2o_values,

    "OH": oh_values

})


# ============================================================
# SAVE DATA
# ============================================================

results.to_csv(
    "results/complete_transient_phi_1.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("COMPLETE TRANSIENT HYDROGEN-AIR COMBUSTION")
print("=" * 65)

print(f"Initial temperature      : {T0:.2f} K")
print(f"Initial pressure         : {P0 / ct.one_atm:.2f} atm")
print(f"Equivalence ratio        : {phi:.2f}")

print("-" * 65)

print(f"Ignition delay           : {ignition_delay:.8f} s")
print(f"Ignition delay           : {ignition_delay * 1000:.4f} ms")

print(f"Maximum temperature      : {max_temperature:.2f} K")
print(f"Maximum pressure         : {max_pressure:.4f} atm")
print(f"Maximum OH mole fraction : {max_oh:.6e}")

print("-" * 65)

print(f"Final H2 mole fraction   : {final_h2:.6e}")
print(f"Final O2 mole fraction   : {final_o2:.6e}")
print(f"Final H2O mole fraction  : {final_h2o:.6e}")

print("=" * 65)


# ============================================================
# PLOT 1 — TEMPERATURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    times,
    temperatures
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title(
    "Transient Hydrogen-Air Combustion: Temperature"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/transient_temperature_phi_1.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 2 — PRESSURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    times,
    pressures
)

plt.xlabel("Time (s)")
plt.ylabel("Pressure (atm)")
plt.title(
    "Transient Hydrogen-Air Combustion: Pressure"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/transient_pressure_phi_1.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 3 — FUEL AND OXYGEN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    times,
    h2_values,
    label="H2"
)

plt.plot(
    times,
    o2_values,
    label="O2"
)

plt.xlabel("Time (s)")
plt.ylabel("Mole Fraction")
plt.title(
    "Hydrogen and Oxygen Consumption"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/h2_o2_consumption_phi_1.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 4 — PRODUCTS AND RADICAL
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    times,
    h2o_values,
    label="H2O"
)

plt.plot(
    times,
    oh_values,
    label="OH"
)

plt.xlabel("Time (s)")
plt.ylabel("Mole Fraction")
plt.title(
    "Water Formation and OH Radical"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/h2o_oh_formation_phi_1.png",
    dpi=300
)

plt.show()