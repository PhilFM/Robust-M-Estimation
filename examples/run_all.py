import sys
sys.path.append("../pypi_package/src")

# robust mean estimation tests
sys.path.append("mean")
import mean.mean_deriv_check
import mean.convergence_speed
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

mean.mean_deriv_check.main(testrun=True, output_folder="../Output")
mean.convergence_speed.main(testrun=True, output_folder="../Output")
mean.majorize_examples.main(testrun=True, output_folder="../Output")
mean.compare_influence.main(testrun=True, output_folder="../Output")
mean.mean_check.main(testrun=True, output_folder="../Output")
mean.student_t_solver.main(testrun=True, output_folder="../Output")
mean.robust_solver.main(testrun=True, output_folder="../Output")
mean.welsch_solver.main(testrun=True, output_folder="../Output")
mean.welsch_efficiency.main(testrun=True, output_folder="../Output")
mean.flat_welsch_solver.main(testrun=True, output_folder="../Output")
mean.pseudo_huber_solver.main(testrun=True, output_folder="../Output")
mean.geman_mcclure_solver.main(testrun=True, output_folder="../Output")
mean.gnc_irls_p_solver.main(testrun=True, output_folder="../Output")
mean.trimmed_mean_efficiency.main(testrun=True, output_folder="../Output")
mean.winsorised_mean_efficiency.main(testrun=True, output_folder="../Output")

# line fitting tests
sys.path.append("line_fitting")
import line_fitting.line_fit_deriv_check
import line_fitting.line_fit_solver
import line_fitting.line_fit_convergence_speed
import line_fitting.line_fit_param_plot

line_fitting.line_fit_deriv_check.main(testrun=True, output_folder="../Output")
line_fitting.line_fit_solver.main(testrun=True, output_folder="../Output")
line_fitting.line_fit_convergence_speed.main(testrun=True, output_folder="../Output")
line_fitting.line_fit_param_plot.main(testrun=True, output_folder="../Output")

# image translation, rotation and scale tests
sys.path.append("image_trs")
import image_trs.trs_solver
import image_trs.trs_convergence_speed
import image_trs.trs_derivative_check

image_trs.trs_solver.main(testrun=True, output_folder="../Output")
image_trs.trs_convergence_speed.main(testrun=True, output_folder="../Output")
image_trs.trs_derivative_check.main(testrun=True, output_folder="../Output")

# 3D point cloud registration tests
sys.path.append("registration_3d")
import registration_3d.registration_solver
import registration_3d.registration_deriv_check

registration_3d.registration_solver.main(testrun=True, output_folder="../Output")
registration_3d.registration_deriv_check.main(testrun=True, output_folder="../Output")
