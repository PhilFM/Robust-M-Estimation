## Robust M-Estimation software

Python code and documentation supporting robust M-estimation using Iterated Reweighted Least Squares (IRLS)
and our custom Supervised Gauss-Newton (Sup-GN) algorithm.
Currently the code supports M-estimation for the problem of finding the robust mean given 1D data containing a mixture of values clustered in a population and outliers. The main specific algorithm is GNC-W, as documented in the submitted paper (to be added). This algorithm applies principles of Graduated Non-Convexity (Blake & Zisserman) to IRLS, using the Welsch influence function.

### Dependencies

The only requirements for the main `gnc_smoothie` package are Python 3.10+ with `numpy` and `matplotlib`.
Some of the example code also uses `scipy`.

### Structure

- [pypi_package](pypi_package/README.md) The source for the PyPi package `gnc_smoothie`.
- [examples](examples/README.md) Example code using the `gnc_smoothie` package.
