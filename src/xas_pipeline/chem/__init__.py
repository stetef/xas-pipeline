"""Chemistry: file formats and numerics, no batch or scheduler logic.

- :mod:`units`, :mod:`periodic` -- constants and element tables
- :mod:`xyz` -- XYZ geometries
- :mod:`hessian` -- ORCA ``.hess`` read/write
- :mod:`feff` -- FEFF/CORVUS output tables, chi(k) -> chi(R), chi health
- :mod:`springs` -- ligand spring models and their interpolation onto a cluster
- :mod:`spring_hessian` -- a ``.hess`` from a spring model and a geometry
- :mod:`spring_model` -- fitting spring models to ORCA Hessians (database building)
"""
