## Robust line fitting to 2D points

Example code illustrating how to compute a line that fits to 2D points with outliers using the `gnc_smoothie` package.

- [line_fit.py](line_fit.py) Model class
- [line_fit_deriv_check.py](line_fit_deriv_check.py) Run this to check that the derivative formulae implemented in the line fitting
  algorithm class `LineFit` in `line_fit.py` is correct. This is done by comparing with numerically calculated derivatives.
- [line_fit_solver.py](line_fit_solver.py) Example code using line fitting and the `gnc_smoothie` package.
- [line_fit_convergence_speed.py](line_fit_convergence_speed.py) Measures the convergence speed of various implementations of IRLS and Sup-GN.
