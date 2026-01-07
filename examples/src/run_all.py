import sys
sys.path.append("../../pypi_package/src")

# supports import of Cython stuff
sys.path.append("cython_files")

# miscellaneous tests
sys.path.append("misc")
import misc.erf_check
import misc.welsch_majorize_check

misc.erf_check.main(test_run=True, output_folder="../../test_output")
misc.welsch_majorize_check.main(test_run=True, output_folder="../../test_output")

# robust mean estimation examples
sys.path.append("mean")
import mean.mean_convergence_speed
import mean.majorize_examples
import mean.compare_influence
import mean.mean_check
import mean.mean_solver
import mean.flat_welsch_solver
import mean.trimmed_mean_efficiency
import mean.winsorised_mean_efficiency
import mean.mean_solve_speed
import mean.welsch_efficiency
import mean.mean_efficiency
import mean.mean_compare_student_t
import mean.mean_compare

mean.mean_convergence_speed.main(test_run=True, output_folder="../../test_output")
mean.majorize_examples.main(test_run=True, output_folder="../../test_output")
mean.compare_influence.main(test_run=True, output_folder="../../test_output")
mean.mean_check.main(test_run=True, output_folder="../../test_output")
mean.mean_solver.main(test_run=True, output_folder="../../test_output")
mean.flat_welsch_solver.main(test_run=True, output_folder="../../test_output")
mean.trimmed_mean_efficiency.main(test_run=True, output_folder="../../test_output")
mean.winsorised_mean_efficiency.main(test_run=True, output_folder="../../test_output")
mean.mean_solve_speed.main(test_run=True, output_folder="../../test_output", quick_run=True)
mean.welsch_efficiency.main(test_run=True, output_folder="../../test_output", quick_run=True)
mean.mean_efficiency.main(test_run=True, output_folder="../../test_output", quick_run=True)
mean.mean_compare_student_t.main(test_run=True, output_folder="../../test_output", quick_run=True)
mean.mean_compare.main(test_run=True, output_folder="../../test_output", quick_run=True)

# line fitting examples
sys.path.append("line_fitting")
import line_fitting.line_fit_solver
import line_fitting.line_fit_convergence_speed
import line_fitting.line_fit_param_plot
import line_fitting.line_fit_breakdown
import line_fitting.line_fit_efficiency

line_fitting.line_fit_solver.main(test_run=True, output_folder="../../test_output")
line_fitting.line_fit_convergence_speed.main(test_run=True, output_folder="../../test_output")
line_fitting.line_fit_param_plot.main(test_run=True, output_folder="../../test_output")
line_fitting.line_fit_breakdown.main(test_run=True, output_folder="../../test_output", quick_run=True)
line_fitting.line_fit_efficiency.main(test_run=True, output_folder="../../test_output", quick_run=True)

# plane fitting examples
sys.path.append("plane_fitting")
import plane_fitting.plane_fit_solver
import plane_fitting.plane_fit_convergence_speed
import plane_fitting.plane_fit_breakdown
import plane_fitting.plane_fit_efficiency

plane_fitting.plane_fit_solver.main(test_run=True, output_folder="../../test_output")
plane_fitting.plane_fit_convergence_speed.main(test_run=True, output_folder="../../test_output")
plane_fitting.plane_fit_breakdown.main(test_run=True, output_folder="../../test_output", quick_run=True)
plane_fitting.plane_fit_efficiency.main(test_run=True, output_folder="../../test_output", quick_run=True)

# image translation, rotation and scale examples
sys.path.append("image_trs")
import image_trs.trs_solver
import image_trs.trs_convergence_speed
import image_trs.trs_derivative_check
import image_trs.trs_breakdown
import image_trs.trs_efficiency

image_trs.trs_solver.main(test_run=True, output_folder="../../test_output")
image_trs.trs_convergence_speed.main(test_run=True, output_folder="../../test_output")
image_trs.trs_derivative_check.main(test_run=True, output_folder="../../test_output")
image_trs.trs_breakdown.main(test_run=True, output_folder="../../test_output", quick_run=True)
image_trs.trs_efficiency.main(test_run=True, output_folder="../../test_output", quick_run=True)

# 3D point cloud registration examples
sys.path.append("registration_3d")
import registration_3d.registration_solver
import registration_3d.registration_deriv_check

registration_3d.registration_solver.main(test_run=True, output_folder="../../test_output")
registration_3d.registration_deriv_check.main(test_run=True, output_folder="../../test_output")
