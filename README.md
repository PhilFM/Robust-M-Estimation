# GNC Smoothie: Robust M-estimation using IRLS and variants

- [pypi_package](pypi_package/README.md) Python package
- [examples](examples/README.md) Example code.

## Build and install `gnc_smoothie` package
```
cd pypi_package
python -m build -w
pip install .
```
## Local testing of GNC Smoothie package

To build and test GNC Smoothie PyPi package:
```
pip install pytest
cd pypi_package
python setup.py build_ext --inplace
cd tests
pytest
```

## Local testing of GNC Smoothie example code

To build and run the GNC Smoothie example tests and samples:
```
pip install pytest
cd examples
python setup.py build_ext --inplace
mv *.so src/cython_files
cd tests
pytest
cd ../src
python run_all.py
```
