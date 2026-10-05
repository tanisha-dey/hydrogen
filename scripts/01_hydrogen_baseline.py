import cantera as ct


# --------------------------------------------------
# Hydrogen Combustion - Baseline Case
# --------------------------------------------------

# Load combustion mechanism
gas = ct.Solution("gri30.yaml")

# Simulation conditions
temperature = 300.0       # K
pressure = ct.one_atm     # Pa
phi = 1.0                 # Stoichiometric

# Set hydrogen-air mixture
gas.set_equivalence_ratio(
    phi=phi,
    fuel="H2",
    oxidizer="O2:1, N2:3.76"
)

# Set initial temperature and pressure
gas.TP = temperature, pressure

# Calculate equilibrium state
gas.equilibrate("HP")


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("=" * 50)
print("HYDROGEN COMBUSTION - BASELINE SIMULATION")
print("=" * 50)

print(f"Initial temperature : {temperature:.2f} K")
print(f"Initial pressure    : {pressure / 101325:.2f} atm")
print(f"Equivalence ratio   : {phi:.2f}")

print("-" * 50)

print(f"Adiabatic flame temperature : {gas.T:.2f} K")

print("-" * 50)
print("Major products:")

for species, mole_fraction in zip(gas.species_names, gas.X):

    if mole_fraction > 1e-6:
        print(f"{species:10s} : {mole_fraction:.6f}")

print("=" * 50)