## Robust line fitting to 2D points

Example code illustrating how to compute a line that fits to 2D points with outliers using the `gnc_smoothie` package.

Example code:

- [line_fit_solver.py](line_fit_solver.py) Example code using line fitting and the `gnc_smoothie` package.
- [line_fit_orthog_welsch.py](line_fit_orthog_welsch.py] Python API for GNC IRLS line fitting with orthogonal regression, to be used by external programs.

Runnable code samples:

- [line_fit_convergence_speed.py](line_fit_convergence_speed.py) Measures the convergence speed of IRLS and Sup-GN.
- [line_fit_breakdown.py](line_fit_breakdown.py) Experimental investigation of worst-case breakdown point for line fitting
  with equally spaced data samples in the x direction
- [line_fit_efficiency.py](line_fit_efficiency.py) Calculates the statistical efficiency of our line fitter compared to others.

Support code:
- [line_fit_orthog.py](line_fit_orthog.py) Model class for line fitting with orthogonal regression model.
