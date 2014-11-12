import os, os.path
import numpy as np
import numpy.random
import math
import copy
import scipy.special
from simtk import openmm
from simtk import unit
from simtk.openmm import app

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
# code from openmmtools testsystem.py

class TestSystem(object):
	"""Abstract base class for test systems, demonstrating how to implement a test system.
	Parameters
	----------
	Attributes
	----------
	system : simtk.openmm.System
	Openmm system with the harmonic oscillator
	positions : list
	positions of harmonic oscillator
	Notes
	-----
	Unimplemented methods will default to the base class methods, which raise a NotImplementedException.
	Examples
	--------
	Create a test system.
	>>> testsystem = TestSystem()
	Retrieve System object.
	>>> system = testsystem.system
	Retrieve the positions.
	>>> positions = testsystem.positions
	Serialize system and positions to XML (to aid in debugging).
	>>> (system_xml, positions_xml) = testsystem.serialize()
	"""
	def __init__(self, temperature=None, pressure=None):
		"""Abstract base class for test system.
		Parameters
		----------
		temperature : simtk.unit.Quantity, optional, units compatible with simtk.unit.kelvin
		The temperature of the system.
		pressure : simtk.unit.Quantity, optional, units compatible with simtk.unit.atmospheres
		The pressure of the system.
		"""
		# Create an empty system object.
		self._system = openmm.System()
		# Store positions.
		self._positions = unit.Quantity(np.zeros([0,3], np.float), unit.nanometers)
		return
		@property
		def system(self):
			"""The simtk.openmm.System object corresponding to the test system."""
			return copy.deepcopy(self._system)
		@system.setter
		def system(self, value):
			self._system = value
		@system.deleter
		def system(self):
			del self._system
		@property
		def positions(self):
			"""The simtk.unit.Quantity object containing the particle positions, with units compatible with simtk.unit.nanometers."""
			return copy.deepcopy(self._positions)
		@positions.setter
		def positions(self, value):
			self._positions = value
		@positions.deleter
		def positions(self):
			del self._positions
		@property
		def analytical_properties(self):
			"""A list of available analytical properties, accessible via 'get_propertyname(thermodynamic_state)' calls."""
			return [ method[4:] for method in dir(self) if (method[0:4]=='get_') ]
		def reduced_potential_expectation(self, state_sampled_from, state_evaluated_in):
			"""Calculate the expected potential energy in state_sampled_from, divided by kB * T in state_evaluated_in.
			Notes
			-----
			This is not called get_reduced_potential_expectation because this function
			requires two, not one, inputs.
			"""
			if hasattr(self, "get_potential_expectation"):
				U = self.get_potential_expectation(state_sampled_from)
				U_red = U / (kB * state_evaluated_in.temperature)
				return U_red
			else:
				raise AttributeError("Cannot return reduced potential energy because system lacks get_potential_expectation")
		def serialize(self):
			"""Return the System and positions in serialized XML form.
			Returns
			-------
			system_xml : str
			Serialized XML form of System object.
			state_xml : str
			Serialized XML form of State object containing particle positions.
			"""
			from simtk.openmm import XmlSerializer
			# Serialize System.
			system_xml = XmlSerializer.serialize(self._system)
			# Serialize positions via State.
			if self._system.getNumParticles() == 0:
			# Cannot serialize the State of a system with no particles.
				state_xml = None
			else:
				platform = openmm.Platform.getPlatformByName('Reference')
				integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
				context = openmm.Context(self._system, integrator, platform)
				context.setPositions(self._positions)
				state = context.getState(getPositions=True)
				del context, integrator
				state_xml = XmlSerializer.serialize(state)
			return (system_xml, state_xml)
		@property
		def name(self):
			"""The name of the test system."""
			return self.__class__.__name__
	

class WaterBox(TestSystem):
	"""
	Create a water box test system.
	Examples
	--------
	Create a default (TIP3P) waterbox.
	>>> waterbox = WaterBox()
	Control the cutoff.
	>>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)
	Use a different water model.
	>>> waterbox = WaterBox(model='tip4pew')
	Don't use constraints.
	>>> waterbox = WaterBox(constrained=False)
	"""
	def __init__(self, box_edge=2.5*unit.nanometers, cutoff=0.9*unit.nanometers, model='tip3p', switch=True, switch_width=0.5*unit.angstroms, constrained=True, dispersion_correction=True, nonbondedMethod=app.PME):
		"""
		Create a water box test system.
		Parameters
		----------
		box_edge : simtk.unit.Quantity with units compatible with nanometers, optional, default = 2.5 nm
		Edge length for cubic box [should be greater than 2*cutoff]
		cutoff : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.9 nm
		Nonbonded cutoff
		model : str, optional, default = 'tip3p'
		The name of the water model to use ['tip3p', 'tip4p', 'tip4pew', 'tip5p', 'spce']
		switch : bool, optional, default = True
		Turns the Lennard-Jones switching function on or off.
		switch_width : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.5 A
		Sets the width of the switch function for Lennard-Jones.
		constrained : bool, optional, default=True
		Sets whether water geometry should be constrained (rigid water implemented via SETTLE) or flexible.
		dispersion_correction : bool, optional, default=True
		Sets whether the long-range dispersion correction should be used.
		nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
		Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
		Examples
		--------
		Create a default waterbox.
		>>> waterbox = WaterBox()
		>>> [system, positions] = [waterbox.system, waterbox.positions]
		Use reaction-field electrostatics instead.
		>>> waterbox = WaterBox(nonbondedMethod=app.CutoffPeriodic)
		Control the cutoff.
		>>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)
		Use a different water model.
		>>> waterbox = WaterBox(model='spce')
		Use a five-site water model.
		>>> waterbox = WaterBox(model='tip5p')
		Turn off the switch function.
		>>> waterbox = WaterBox(switch=False)
		Set the switch width.
		>>> waterbox = WaterBox(switch=True, switch_width=0.8*unit.angstroms)
		Turn of long-range dispersion correction.
		>>> waterbox = WaterBox(dispersion_correction=False)
		"""
		import simtk.openmm.app as app
		supported_models = ['tip3p', 'tip4pew', 'tip5p', 'spce']
		if model not in supported_models:
			raise Exception("Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))
		# Load forcefield for solvent model.
		ff = app.ForceField(model + '.xml')
		# Create empty topology and coordinates.
		top = app.Topology()
		pos = unit.Quantity((), unit.angstroms)
		# Create new Modeller instance.
		m = app.Modeller(top, pos)
		# Add solvent to specified box dimensions.
		boxSize = unit.Quantity(numpy.ones([3]) * box_edge/box_edge.unit, box_edge.unit)
		m.addSolvent(ff, boxSize=boxSize, model=model)
		# Get new topology and coordinates.
		newtop = m.getTopology()
		newpos = m.getPositions()
		# Convert positions to numpy.
		positions = unit.Quantity(numpy.array(newpos / newpos.unit), newpos.unit)
		# Create OpenMM System.
		system = ff.createSystem(newtop, nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff, constraints=None, rigidWater=constrained, removeCMMotion=False)
		# Set switching function and dispersion correction.
		forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
		forces['NonbondedForce'].setUseSwitchingFunction(switch)
		forces['NonbondedForce'].setSwitchingDistance(cutoff - switch_width)
		forces['NonbondedForce'].setUseDispersionCorrection(dispersion_correction)
		self.ndof = 3*system.getNumParticles() - 3*constrained
		
		self.system, self.positions, self.topology = system, positions, newtop

# waterboxes from sixe 5 - 200 nm
print('creating waterbox sizes array')
box_edges = np.linspace(2.5, 20, num=10)

print('creating waterbox objects')
for edge in box_edges:
	print('creating waterbox with edge %d' % edge)
	waterbox = WaterBox(edge*unit.nanometers)
        
	print('writing pdb file')
	
	app.PDBFile.writeFile(waterbox.topology, waterbox.positions, open('../pdb_files/waterbox_%d.pdb' % edge, 'w'))
 
	integrator = openmm.LangevinIntegrator(
			           300*unit.kelvin,
				   1.0/unit.picosecond,
				   2.0*unit.femtosecond
	)	
	barostat = openmm.MonteCarloBarostat(
				 1.0*unit.bar,
				 300*unit.kelvin,
				 25
	)	

	platform = openmm.Platform.getPlatformByName('CUDA')
	prop = dict(CudaPrecision='mixed')

	# minimize each water box and see when openmm crashes


	print('testing waterbox of size %d' % edge)
	waterbox.system.addForce(barostat)
	simulation = app.Simulation(waterbox.topology, waterbox.system, integrator, platform, prop)
	simulation.context.setPositions(waterbox.positions)
	print('minimizing')
	simulation.minimizeEnergy()

