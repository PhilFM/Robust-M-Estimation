## `mean` - Robust mean estimation

Example code illustrating how to compute a robust mean of scalar values using the `gnc_smoothie` package.
Some code uses Timothee Mathieu's `robust_mean` package [RobustMeanEstimator](https://github.com/TimotheeMathieu/RobustMeanEstimator.git).

Top-level example code:
- [welsch_solver.py](welsch_solver.py) Example code that generates simulated data and runs IRLS and Sup-GN with the Welsch influence function.

Other runnable examples:
- [mean_deriv_check.py](mean_deriv_check.py) Run this to check that the derivative formula implemented in the mean estimation
- [convergence_speed.py](convergence_speed.py) Measures the convergence speed of various implementations of IRLS and Sup-GN.
- [majorize_examples.py](majorize_examples.py) Examples showing the technique of quadratic majorization for influence functions implemented in this package.
- [compare_influence.py](compare_influence.py) Compare results with a variety of influence functions.
- [welsch_efficiency.py](welsch_efficiency.py) Code measuring the efficiency of the GNC Welsch (GNC-W) estimator for different sample sizes,
  and comparing to the theoretical distribution.
- [mean_check.py](mean_check.py) Examples of running various versions of mean estimation on random data.
- [pseudo_huber_solver.py](pseudo_huber_solver.py) Example code that generates simulated data and runs IRLS with the pseudo-Huber influence function.
- [geman_mcclure_solver.py](geman_mcclure_solver.py) Example code that generates simulated data and runs IRLS with the Geman-McClure influence function.
- [gnc_irls_p_solver.py](gnc_irls_p_solver.py) Example code that generates simulated data and runs IRLS with the GNC IRLS-p influence function.
- [robust_solver.py](robust_solver.py) Compares a number of mean estimation algorithms and outputs the results to figures and JSON files.
  This code combines data with a Gaussian distribution with outliers having a uniform distribution.
- [student_t_solver.py](student_t_solver.py) Compares a number of mean estimation algorithms and outputs the results to figures and JSON files.
  This code generates random data from the student-t distribution.
- [flat_welsch_solver.py](flat_welsch_solver.py) Example code for calculating the optimum mean estimate without using a GNC schedule.
- [trimmed_mean_efficiency.py](trimmed_mean_efficiency.py) Calculation of the efficiency of the trimmed mean with simulated data.
- [winsorised_mean_efficiency.py](winsorised_mean_efficiency.py) Calculation of the efficiency of the Winsorised mean with simulated data.

Support code:
- [gncs_robust_mean.py](gncs_robust_mean.py) Model class
  algorithm class `RobustMean` in `gncs_robust_mean.py` is correct. This is done by comparing with numerically calculated derivatives.
- [flat_welsch_mean](float_welsch_mean.py) Calculation of global optimum mean estimate using 1D search, without using a GNC schedule.
- [trimmed_mean.py](trimmed_mean.py) Calculation of the trimmed mean with the level of trimming specified  the `trim_size` parameter
- [winsorised_mean.py](winsorised_mean.py) Calculation of the Winsorised mean with the level of trimming specified  the `trim_size` parameter
