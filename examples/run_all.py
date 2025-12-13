import sys
sys.path.append("../pypi_package/src")

# robust mean estimation tests
sys.path.append("mean")
import mean.mean_deriv_check
import mean.mean_convergence_speed
import mean.majorize_examples
import mean.compare_influence
import mean.mean_check
import mean.student_t_solver
import mean.robust_solver
import mean.welsch_solver
import mean.welsch_efficiency
import mean.flat_welsch_solver
import mean.pseudo_huber_solver
import mean.geman_mcclure_solver
import mean.gnc_irls_p_solver
import mean.trimmed_mean_efficiency
import mean.winsorised_mean_efficiency

mean.mean_deriv_check.main(test_run=True, output_folder="../output")
mean.mean_convergence_speed.main(test_run=True, output_folder="../output")
mean.majorize_examples.main(test_run=True, output_folder="../output")
mean.compare_influence.main(test_run=True, output_folder="../output")
mean.mean_check.main(test_run=True, output_folder="../output")
mean.student_t_solver.main(test_run=True, output_folder="../test_output", quick_run=True)
mean.robust_solver.main(test_run=True, output_folder="../test_output", quick_run=True)
mean.welsch_solver.main(test_run=True, output_folder="../output")
mean.welsch_efficiency.main(test_run=True, output_folder="../output")
mean.flat_welsch_solver.main(test_run=True, output_folder="../output")
mean.pseudo_huber_solver.main(test_run=True, output_folder="../output")
mean.geman_mcclure_solver.main(test_run=True, output_folder="../output")
mean.gnc_irls_p_solver.main(test_run=True, output_folder="../output")
mean.trimmed_mean_efficiency.main(test_run=True, output_folder="../output")
mean.winsorised_mean_efficiency.main(test_run=True, output_folder="../output")

# line fitting tests
sys.path.append("line_fitting")
import line_fitting.line_fit_deriv_check
import line_fitting.line_fit_solver
import line_fitting.line_fit_convergence_speed
import line_fitting.line_fit_param_plot

line_fitting.line_fit_deriv_check.main(test_run=True, output_folder="../output")
line_fitting.line_fit_solver.main(test_run=True, output_folder="../output")
line_fitting.line_fit_convergence_speed.main(test_run=True, output_folder="../output")
line_fitting.line_fit_param_plot.main(test_run=True, output_folder="../output")

# image translation, rotation and scale tests
sys.path.append("image_trs")
import image_trs.trs_solver
import image_trs.trs_convergence_speed
import image_trs.trs_derivative_check

image_trs.trs_solver.main(test_run=True, output_folder="../output")
image_trs.trs_convergence_speed.main(test_run=True, output_folder="../output")
image_trs.trs_derivative_check.main(test_run=True, output_folder="../output")

# 3D point cloud registration tests
sys.path.append("registration_3d")
import registration_3d.registration_solver
import registration_3d.registration_deriv_check

registration_3d.registration_solver.main(test_run=True, output_folder="../output")
registration_3d.registration_deriv_check.main(test_run=True, output_folder="../output")
