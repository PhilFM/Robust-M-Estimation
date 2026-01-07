## `image_trs` - Image translation, rotation and scale estimation

Robust estimation of translation, rotation and scale between points in two images line fitting to 2D points using the `gnc_smoothie` package.

Example code:

- [trs_solver.py](trs_solver.py) Example code using image translation, rotation and scale estimation with the `gnc_smoothie` package.
- [trs_welsch.py](trs_welsch.py] Python API for GNC Sup-GN image translation, rotation and scale estimation to be used by external programs.

Runnable code samples:

- [trs_derivative_check.py](trs_derivative_check.py) Run this to check that the derivative formulae implemented in the
  algorithm class `TRS` in `trs.py` is correct. This is done by comparing with numerically calculated derivatives.
- [trs_convergence_speed.py](trs_convergence_speed.py) Measures the convergence speed of various implementations of IRLS and Sup-GN.
- [trs_efficiency.py](tes_efficiency.py) Calculates the statistical efficiency of our estimator.

Support code:
- [trs.py](trs.py) Model class for calculation of 2D image transformation.
