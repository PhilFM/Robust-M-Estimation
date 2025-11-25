## Robust M-Estimation

Python code and documentation supporting robust M-estimation using Iterated Reweighted Least Squares (IRLS). Currently the code supports M-estimation for the problem of finding the robust mean given 1D data containing a mixture of values clustered in a population and outliers. The main specific algorithm is GNC-W, as documented in the submitted paper (to be added). This algorithm applies principles of Graduated Non-Convexity (Blake & Zisserman) to IRLS, using the Welsch influence function.

### Dependencies

The main requirements are `numpy` and `matplotlib`. To install these use the command
```
pip install numpy matplotlib
```
There is also some code using Timothee Mathieu's [RobustMeanEstimator)[https://github.com/TimotheeMathieu/RobustMeanEstimator.git] algorithm, which has it's own installation method. If you want to use this, please check out the code and follow the instructions in the `README.md` file.

**NB** the `requirements.txt` file contains the python packages used on a Linux machine using Python 3.13.

### Structure of code

All the code is in the `Code` directory. The `Library` sub-directory contains some generic IRLS code used for different algorithms, and some specific IRLS parameter classes, which are also common to multiple algorithms. All the directly executable Python code is in the `Average` directory and its sub-directories. All programs may be executed using the command `python <Python file>` - there are no command line arguments. Output files where relevant are written into the `Outputs` directory. Here is a description of each program:

- `derivativeCheck.py` Run this to check that the first and second derivative formulae implemented in each mean estimation algorithm class (e.g. `Welsch/WelschMean.py`) are correct. This is done by comparing with numerically calculated derivatives.
- `majorizeExamples.py` Examples showing the technique of quadratic majorization for influence functions implemented in this package.
- `robustSolver.py` Compares a number of mean estimation algorithms and outputs the results to figures and JSON files. This code combines data with a Gaussian distribution with outliers having a uniform distribution.
- `studentTSolver.py` Compares a number of mean estimation algorithms and outputs the results to figures and JSON files. This code generates random data from the student-t distribution.

Now the programs within each sub-directory. To run them, `cd` into each sub-directory and execute `python <Python file>` as before.
- `Welsch` Code related to the Welsch influence function:
   - `welschSolver.py` Example code that generates simulated data and runs IRLS with the Welsch influence function.
   - `welschEfficiency.py` Code measuring the efficiency of the GNC-W estimator for different sample sizes, and comparing to the theoretical distribution.
- `PseudoHuber` Code related to the pseudo-Huber influence function.
   - `pseudoHuberSolver.py` xample code that generates simulated data and runs IRLS with the pseudo-Huber influence function.
- `GemanMcClure` Code related to the Geman-McClure influence function.
   - `gemanMcClureSolver.py` Example code that generates simulated data and runs IRLS with the Geman-McClure influence function.
- `Trimmed` Code related to various trimmed mean algorithms:
   - `trimmedMeanEfficiency.py` Calculation of the efficiency of the trimmed mean with simulated data.
   - `winsorisedMeanEfficiency.py` Calculation of the efficiency of the Winsorised mean with simulated data.

### Structure of code

We separate the basic IRLS algorithm (implemented in the code `Library/IRLS.py`) from the specific algorithm and parameter code. The style is similar to C++ templates. To run IRLS we need to build
1. An instance of a parameter class. There are two types of parameter class - simple and GNC. GNC parameter classes contain extra management functionality related to the GNC schedule. Examples:
   - `Library/WelschParams.py` implements a simple class, encapsulating a Welsch influence function with given `sigma` value.
   - `Library/GNC_WelschParams.py` implements a GNC class for the Welsch influence function, wherein `sigma` is successively reduced, after being initialised to a large value.
1. An instance of an estimator class using the parameter class instance. An instance of either a simple or a GNC parameter class may be used to build the estimator class instance. An example is `Average/Welsch/WelschMean.py`. The estimator class also requires that the input data and weights be passed in.
1. An instance of the `IRLS` class, using the estimator class instance and optional parameters. Once this is built, call the `run()` function to execute IRLS to completion.

Example code showing how to use the IRLS algorithm is in `Average/Welsch\welschSolver.py`.



