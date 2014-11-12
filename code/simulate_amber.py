import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import sys

print('reading pdb')
pdb = app.PDBFile('../pdb_files/water_3.pdb')
forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, 
				 nonbondedCutoff=12.0*u.angstroms,
				 constraints=app.HBonds,
				 switchDistance=10.0*u.angstroms
)

integrator = mm.LangevinIntegrator(
				   300*u.kelvin, 
				   1.0/u.picosecond, 
				   2.0*u.femtosecond
)

barostat = mm.MonteCarloBarostat(
				 1.0*u.bar,
				 300*u.kelvin,
				 25
)

system.addForce(barostat)

platform = mm.Platform.getPlatformByName('CUDA')
prop = dict(CudaPrecision='mixed')

simulation = app.Simulation(pdb.topology, system, integrator, platform, prop)

print('setting positions')
simulation.context.setPositions(pdb.positions)

print('minimizing')
simulation.minimizeEnergy()

# Print out pdb of minimzed system
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(pdb.topology,positions, open('../simulation/water3_min_amber.pdb', 'w'))


simulation.reporters.append(app.StateDataReporter(sys.stdout, 1, step=True, potentialEnergy=True,
                              kineticEnergy=True, temperature=True,
                              volume=True, density=True, separator='\t' ))

simulation.reporters.append(app.DCDReporter('../simulation/water3_amber.dcd', 1))

# equilibriate
simulation.context.setVelocitiesToTemperature(300*u.kelvin)
print('Equilibirating...')
simulation.step(100)

# Run dynamics
print('Running dynamics')
simulation.step(1000)
                           

