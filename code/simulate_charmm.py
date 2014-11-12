#!/usr/bin/env python
from __future__ import division, print_function

import sys

# OpenMM Imports
import simtk.unit as u
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd Imports
from chemistry.charmm.openmmloader import (OpenMMCharmmPsfFile as CharmmPsfFile,
                                           OpenMMCharmmCrdFile as CharmmCrdFile)
from chemistry.charmm.parameters import CharmmParameterSet
from chemistry.amber.openmmreporters import (
            AmberStateDataReporter as AKMAStateDataReporter)

# Load the CHARMM files
print('Loading CHARMM files...')
params = CharmmParameterSet('toppar_water_ions.str')
water_par = CharmmPsfFile('../pdb_files/water_3.psf')
water_top = app.PDBFile('../pdb_files/water_3.pdb')


# Compute the box dimensions from the coordinates and set the box lengths (only
# orthorhombic boxes are currently supported in OpenMM)
print('computing box dimensions')
coords = water_top.positions
min_crds = [coords[0][0], coords[0][1], coords[0][2]]
max_crds = [coords[0][0], coords[0][1], coords[0][2]]

for coord in coords:
    min_crds[0] = min(min_crds[0], coord[0])
    min_crds[1] = min(min_crds[1], coord[1])
    min_crds[2] = min(min_crds[2], coord[2])
    max_crds[0] = max(max_crds[0], coord[0])
    max_crds[1] = max(max_crds[1], coord[1])
    max_crds[2] = max(max_crds[2], coord[2])
#scale_fact = 1
water_par.setBox((max_crds[0]-min_crds[0]),
                 (max_crds[1]-min_crds[1]),
                 (max_crds[2]-min_crds[2]),
)

# Create the OpenMM system
print('Creating OpenMM System')
system = water_par.createSystem(params, nonbondedMethod=app.PME,
                                nonbondedCutoff=12.0*u.angstroms,
                                constraints=app.HBonds,
				switchDistance=10.0*u.angstroms
)
# Isotropic pressure coupling
barostat = mm.MonteCarloBarostat(1.0*u.bar, 300.0*u.kelvin, 25)
system.addForce(barostat)

# restrain protein
# First compute the system mass
#total_mass = sum([a.mass for a in twoRH1_solv.atom_list]) * u.dalton
#for i, atom in enumerate(twoRH1_solv.atom_list):
#    if atom.residue.resname in ('WAT', 'HOH', 'TIP3'):
#        continue # Skip these atoms
#    if atom.name in ('SOD', 'CLA'):
#        continue
#    system.setParticleMass(i, 0*u.dalton)

# Create the integrator to do Langevin dynamics
integrator = mm.LangevinIntegrator(
                        300*u.kelvin,       # Temperature of heat bath
                        1.0/u.picoseconds,  # Friction coefficient
                        2.0*u.femtoseconds, # Time step
)

# Define platform
platform = mm.Platform.getPlatformByName('CUDA')
prop = dict(CudaPrecision='mixed')

# Create the Simulation object
print('creating simulation object')
sim = app.Simulation(water_par.topology, system, integrator, platform, prop )

# Set the particle positions
print('setting particle positions')
sim.context.setPositions(water_top.positions)

# Minimize the energy
print('Minimizing energy')
sim.minimizeEnergy()

# Print out pdb of minimzed system
positions = sim.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(water_top.topology,positions, open('../simulation/water3_min_output.pdb', 'w'))

sim.reporters.append(AKMAStateDataReporter(sys.stdout, 1, step=True, potentialEnergy=True,
                              kineticEnergy=True, temperature=True,
                              volume=True, density=True, separator='\t' ))

sim.reporters.append(app.DCDReporter('../simulation/water3_sim.dcd', 1))
# equilibriate
sim.context.setVelocitiesToTemperature(300*u.kelvin)
print('Equilibirating...')
sim.step(100)

# Set up the reporters to report energies and coordinates every 100 steps

# Run dynamics
print('Running dynamics')
sim.step(1000)
