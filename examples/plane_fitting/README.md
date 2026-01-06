## Robust plane fitting to 2D points

Example code illustrating how to compute a plane that fits to 3D points with outliers using the `gnc_smoothie` package.

Example code:

- [plane_fit_solver.py](plane_fit_solver.py) Example code using plane fitting and the `gnc_smoothie` package.
- [plane_fit_orthog_welsch.py](plane_fit_orthog_welsch.py] Python API for GNC IRLS plane fitting with orthogonal regression, to be used by external programs.

Runnable code samples:

- [plane_fit_convergence_speed.py](plane_fit_convergence_speed.py) Measures the convergence speed of IRLS and Sup-GN.
- [plane_fit_efficiency.py](plane_fit_efficiency.py) Calculates the statistical efficiency of our plane fitter compared to others.

Support code:
- [plane_fit_orthog.py](plane_fit_orthog.py) Model class for plane fitting with orthogonal regression model.
