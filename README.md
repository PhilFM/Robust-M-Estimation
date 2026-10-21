# GNC Smoothie: Robust M-estimation using IRLS and variants

- [pypi_package](pypi_package/README.md) Python package
- [examples](examples/README.md) Example code.

## Build and install `gnc_smoothie` package

`gnc_smoothie` has been tested with Python version 3.14.

You will need to install the `build` Python package first:
```
python -m pip install build
```
Then you can proceed to build the `gnc_smoothie` package as a wheel file:
```
cd pypi_package
python -m build -s
python -m build -w
pip install .
```
## Local testing of GNC Smoothie package

To build and test GNC Smoothie PyPi package:
```
pip install pytest setuptools scipy
cd pypi_package
python setup.py build_ext --inplace
mv build/lib*/gnc_smoothie/cython_files/*.so src/gnc_smoothie/cython_files
cd tests
pytest
```
On Windows, replace the `mv *.so src/cython_files` command with
```
mv build/lib*/gnc_smoothie/cython_files/*.pyd src/gnc_smoothie/cython_files
```

## Local testing of GNC Smoothie example code

To build and run the GNC Smoothie tests:
```
pip install Cython
cd examples
python setup.py build_ext --inplace
mv *.so src/cython_files
cd tests
pytest
```
On Windows, replace the `mv *.so src/cython_files` command with
```
mv *.pyd src/cython_files/
```

The sample code requires additional packages.
Firstly open a new shell and install Timothee Mathieu's `RobustMeanEstimator`
package from source:
```
git clone https://github.com/TimotheeMathieu/RobustMeanEstimator.git
cd RobustMeanEstimator/
pip install .
```
Then you can return to the `Robust-M-Estimation` shell and perform the
following commands
```
pip install pandas statsmodels seaborn scikit-learn opencv-python open3d
cd ../src
python run_all.py
```
