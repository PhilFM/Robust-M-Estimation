## `registration_3d` - 3D point cloud registration

Robust estimation of 3D translation and rotation between two 3D point clouds using the `gnc_smoothie` package.

- `registration_deriv_check.py` Run this to check that the derivative formulae implemented in the
  algorithm class `PointRegistration` in `point_registration.py` is correct. This is done by comparing with numerically calculated derivatives.
- `registration_solver.py` Example code using 3D translation and rotation estimation with the `gnc_smoothie` package.
