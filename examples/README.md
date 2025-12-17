## Example code using the `gnc_smoothie` package

Four models are implemented in this folder:

- [mean](mean/README.md) Robust mean estimation.
- [line_fitting](line_fitting/README.md) Line fitting to 2D points.
- [plane_fitting](plane_fitting/README.md) Plane fitting to 3D points.
- [image_trs](image_trs/README.md) Fitting 2D translation, rotation and scale to points on the XY plane.
- [registration_3d](registration_3d/README.md) Calculate rotation and translation between two 3D point clouds.

There is also a [run_all.py](run_all.py) program that runs all the example code and prints any errors.
Output files are written to the top-level `test_output` folder.
