"""Pipeline stages, one module per step of the workflow.

ORCA input prep (:mod:`orca_prep`), the ORCA convergence check
(:mod:`orca_check`), the spring Hessian (:mod:`interp_hessian`), CORVUS prep
(:mod:`corvus_prep`), FEFF post-processing (:mod:`feff_process`), download
staging (:mod:`download`), disk clean-up (:mod:`cleanup`) and the
imaginary-frequency survey (:mod:`count_imag_freq`). Each is a library of
functions plus a ``main()`` behind an ``xas-*`` console script.
"""
