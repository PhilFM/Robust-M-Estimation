## Robust plane fitting to 2D points

Example code illustrating how to compute a plane that fits to 3D points with outliers using the `gnc_smoothie` package.

Top-level example code:

- [plane_fit_solver.py](plane_fit_solver.py) Example code using plane fitting and the `gnc_smoothie` package.

Other runnable example code samples:

- [plane_fit_deriv_check.py](plane_fit_deriv_check.py) Run this to check that the derivative formulae implemented in the plane fitting
  algorithm class `PlaneFit` in [plane_fit.py](plane_fit.py) is correct. This is done by comparing with numerically calculated derivatives.
- [plane_fit_convergence_speed.py](plane_fit_convergence_speed.py) Measures the convergence speed of IRLS and Sup-GN.

Support code:
- [plane_fit.py](plane_fit.py) Model class for plane fitting.
