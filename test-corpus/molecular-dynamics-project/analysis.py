
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

# Load trajectory
u = mda.Universe("water.data", "traj.lammpstrj")

# Compute radial distribution function
rdf_analysis = rdf.InterRDF(u.select_atoms("type 1"), u.select_atoms("type 1"))
rdf_analysis.run()

print(f"RDF computed: {len(rdf_analysis.results.rdf)} bins")
