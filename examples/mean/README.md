## `mean` - Robust mean estimation

Example code illustrating how to compute a robust mean of scalar values using the `gnc_smoothie` package.
Some code uses Timothee Mathieu's `robust_mean` package [RobustMeanEstimator](https://github.com/TimotheeMathieu/RobustMeanEstimator.git).

Example code:
- [mean_welsch_solver.py](mean_welsch_solver.py) Example code that generates simulated data and runs IRLS and Sup-GN with the Welsch influence function.
- [mean_welsch.py](mean_welsch.py] Python API for GNC Sup-GN robust mean estimation to be used by external programs.

Runnable code samples:
- [mean_deriv_check.py](mean_deriv_check.py) Run this to check that the derivative formula implemented in the mean estimation
  algorithm class `RobustMean` in `gncs_robust_mean.py` is correct. This is done by comparing with numerically calculated derivatives.
- [mean_convergence_speed.py](mean_convergence_speed.py) Measures the convergence speed of various implementations of IRLS and Sup-GN.
- [majorize_examples.py](majorize_examples.py) Examples showing the technique of quadratic majorization for influence functions implemented in this package.
- [compare_influence.py](compare_influence.py) Compare results with a variety of influence functions.
- [welsch_efficiency.py](welsch_efficiency.py) Code measuring the efficiency of the GNC Welsch (GNC-W) estimator for different sample sizes,
  and comparing to the theoretical distribution.
- [mean_efficiency.py](mean_efficiency.py) Calculates the statistical efficiency of our mean estimator compared to others.
- [mean_check.py](mean_check.py) Examples of running various versions of mean estimation on random data.
- [mean_compare.py](mean_compare.py) Compares a number of mean estimation algorithms and outputs the results to figures and JSON files.
  This code combines data with a Gaussian distribution with outliers having a uniform distribution.
- [mean_compare_student_t.py](mean_compare_student_t.py) Compares a number of mean estimation algorithms and outputs the results to figures and JSON files.
  This code generates random data from the student-t distribution.
- [flat_welsch_solver.py](flat_welsch_solver.py) Example code for calculating the optimum mean estimate without using a GNC schedule.
- [trimmed_mean_efficiency.py](trimmed_mean_efficiency.py) Calculation of the efficiency of the trimmed mean with simulated data.
- [winsorised_mean_efficiency.py](winsorised_mean_efficiency.py) Calculation of the efficiency of the Winsorised mean with simulated data.

Support code:
- [gncs_robust_mean.py](gncs_robust_mean.py) Model class for robust mean estimation.
- [flat_welsch_mean](flat_welsch_mean.py) Calculation of global optimum mean estimate using 1D search, without using a GNC schedule.
- [trimmed_mean.py](trimmed_mean.py) Calculation of the trimmed mean with the level of trimming specified  the `trim_size` parameter.
- [winsorised_mean.py](winsorised_mean.py) Calculation of the Winsorised mean with the level of trimming specified  the `trim_size` parameter.
