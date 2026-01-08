import sys
import pytest
import numpy as np

sys.path.append("../src")
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.check_derivs import check_derivs
from gnc_smoothie_philfm.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie_philfm.linear_model.linear_regressor_pseudo_huber import LinearRegressorPseudoHuber
from gnc_smoothie_philfm.linear_model.linear_regressor_gnc_irls_p import LinearRegressorGNC_IRLSp

from evaluator_check import evaluator_check

def randomM11() -> float:
    return 2.0*(np.random.rand() - 0.5)

def test_derivs():
    np.random.seed(0) # We want the numbers to be the same on each run
    all_good = True
    for test_idx in range(10):
        n_models = np.random.randint(2, 5)
        dim = np.random.randint(3, 8)
        #print("dim=",dim,"n_models=",n_models)

        # ground-truth model
        model = np.zeros(n_models*dim)
        for i in range(n_models*dim):
            model[i] = randomM11()

        # build good data
        n_points = np.random.randint(5, 10)
        data = np.zeros((n_points,n_models,dim))
        weight = np.zeros(n_points)
        for i in range(n_points):
            for j in range(n_models):
                for k in range(dim):
                    data[i][j][k] = randomM11()

            weight[i] = 0.2+np.random.rand()

        assert(check_derivs(SupGaussNewton(GNC_NullParams(QuadraticInfluenceFunc()), data, model_instance=LinearRegressor(data[0]), weight=weight),
                            model, diff_threshold_AlB=1.e-4)) #, print_diffs=True, print_derivs=True):
        assert(check_derivs(SupGaussNewton(GNC_NullParams(WelschInfluenceFunc(sigma=0.5)), data, model_instance=LinearRegressor(data[0]), weight=weight),
                            model, diff_threshold_AlB=1.e-4)) #, print_diffs=True, print_derivs=True):
        assert(check_derivs(SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=0.5)), data, model_instance=LinearRegressor(data[0]), weight=weight),
                            model, diff_threshold_AlB=1.e-4)) #, print_diffs=True, print_derivs=True):
        assert(check_derivs(SupGaussNewton(GNC_NullParams(GemanMcClureInfluenceFunc(sigma=0.5)), data, model_instance=LinearRegressor(data[0]), weight=weight),
                            model, diff_threshold_AlB=1.e-4)) #, print_diffs=True, print_derivs=True):
        assert(check_derivs(SupGaussNewton(GNC_NullParams(GNC_IRLSpInfluenceFunc(p=0.9, rscale=0.8, epsilon=0.2)), data, model_instance=LinearRegressor(data[0]), weight=weight),
                            model, diff_threshold_AlB=1.e-4)) #, print_diffs=True, print_derivs=True):

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_models = np.random.randint(2, 5)
        dim = np.random.randint(5, 6)
        #print("dim=",dim,"n_models=",n_models)

        # ground-truth model
        model_gt = np.zeros(n_models*dim)
        for i in range(n_models*dim):
            model_gt[i] = randomM11()

        # build good data
        n_good_points = np.random.randint(5, 10)
        n_outliers = 0 #int(np.random.rand()*n_good_points)

        data = np.zeros((n_good_points+n_outliers,n_models,dim))
        data_x = np.zeros((n_good_points+n_outliers,n_models,dim-1))
        data_y = np.zeros((n_good_points+n_outliers,n_models))
        for i in range(n_good_points):
            for j in range(n_models):
                for k in range(dim-1):
                    data[i][j][k] = randomM11()
                    data_x[i][j][k] = data[i][j][k]

                data[i][j][dim-1] = model_gt[j*dim+dim-1]
                for k in range(dim-1):
                    data[i][j][dim-1] += model_gt[j*dim+k]*data[i][j][k]

                data_y[i][j] = data[i][j][dim-1]

        for i in range(n_outliers):
            for j in range(n_models):
                for k in range(dim-1):
                    data[n_good_points+i][j][k] = randomM11()
                    data_x[n_good_points+i][j][k] = data[n_good_points+i][j][k]

        sigma_base = 0.01
        sigma_limit = 1.0
        num_sigma_steps = 20

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]))
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        #print("model_gt=",model_gt)
        #print("model=",model)
        for j in range(n_models*dim):
            assert(model[j] == pytest.approx(model_gt[j]))

        #print("model_start=",model_start)
        linear_regressor = LinearRegressorWelsch(sigma_base, sigma_limit, num_sigma_steps)
        assert(linear_regressor.run(data))
        model = linear_regressor.final_model
        for j in range(n_models*dim):
            assert(model[j] == pytest.approx(model_gt[j]))

        # test with scipy convention for data argument
        assert(linear_regressor.run((data_x, data_y)))
        coeff = linear_regressor.final_coeff
        intercept = linear_regressor.final_intercept
        for j in range(n_models):
            for k in range(dim-1):
                assert(coeff[j][k] == pytest.approx(model_gt[j*dim+k]))

            assert(intercept[j] == pytest.approx(model_gt[j*dim+dim-1]))

def build_random_data():
    n_models = np.random.randint(2, 5)
    dim = np.random.randint(2, 10)
    #print("dim=",dim,"n_models=",n_models)
    n_points = np.random.randint(5, 10)*dim

    data = np.zeros((n_points,n_models,dim))
    weight = np.zeros(n_points)
    scale = np.zeros(n_points)
    for i in range(n_points):
        for j in range(n_models):
            for k in range(dim):
                data[i][j][k] = np.random.rand()

        weight[i] = 0.2+np.random.rand()
        scale[i] = 1.0+np.random.rand()

    return data,weight,scale

def build_random_model(data_item):
    n_models = data_item.shape[0]
    dim = data_item.shape[1]
    model = np.zeros(n_models*dim)
    for i in range(n_models*dim):
        model[i] = np.random.rand()

    return model

def check_final_models(data_item, model1, model2) -> bool:
    n_models = data_item.shape[0]
    dim = data_item.shape[1]
    assert(len(model1) == n_models*dim)
    assert(len(model2) == n_models*dim)
    for i in range(n_models*dim):
        assert(model1[i] == pytest.approx(model2[i]))

    return True

def test_evaluator_welsch():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        data,weight,scale = build_random_data()
        model = build_random_model(data[0])

        sigma_base = 0.2
        sigma_limit = 50.0
        num_sigma_steps = 20

        # check individual evaluator functions
        evaluator_instance = LinearRegressorWelschEvaluator(data[0])
        influence_func_instance = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps)
        param_instance.reset(False) # sets sigma to sigma_base

        assert(evaluator_check(evaluator_instance, param_instance, LinearRegressor(data[0]), model, data, weight, scale))

        max_niterations = 500

        # fast Sup-GN
        linear_regressor = LinearRegressorWelsch(sigma_base, sigma_limit, num_sigma_steps, max_niterations=max_niterations)
        assert(linear_regressor.run(data))
        fast_model = linear_regressor.final_model

        # check against slow reference version
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]), max_niterations=max_niterations)
        assert(optimiser_instance.run())
        slow_model = optimiser_instance.final_model

        assert(check_final_models(data[0], fast_model, slow_model))

def test_evaluator_pseudo_huber():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        data,weight,scale = build_random_data()
        model = build_random_model(data[0])

        sigma = 0.2

        # check individual evaluator functions
        evaluator_instance = LinearRegressorPseudoHuberEvaluator(data[0])
        influence_func_instance = PseudoHuberInfluenceFunc(sigma)
        param_instance = GNC_NullParams(influence_func_instance)

        assert(evaluator_check(evaluator_instance, param_instance, LinearRegressor(data[0]), model, data, weight, scale))

        max_niterations = 500

        # fast Sup-GN
        linear_regressor = LinearRegressorPseudoHuber(sigma, max_niterations=max_niterations)
        assert(linear_regressor.run(data))
        fast_model = linear_regressor.final_model

        # check against slow reference version
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]), max_niterations=max_niterations)
        assert(optimiser_instance.run())
        slow_model = optimiser_instance.final_model
        #print("model=",model)

        assert(check_final_models(data[0], fast_model, slow_model))

def test_evaluator_gnc_irls_p():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        data,weight,scale = build_random_data()
        model = build_random_model(data[0])

        p = np.random.rand()
        rscale = 0.5 + np.random.rand()
        epsilon_base = 0.1 + 0.2
        epsilon_limit = 1.0
        beta = 0.95

        # check individual evaluator functions
        evaluator_instance = LinearRegressorGNC_IRLSpEvaluator(data[0])
        influence_func_instance = GNC_IRLSpInfluenceFunc()
        param_instance = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, beta)
        param_instance.reset(False) # sets sigma to sigma_base

        assert(evaluator_check(evaluator_instance, param_instance, LinearRegressor(data[0]), model, data, weight, scale))

        max_niterations = 500

        # fast Sup-GN
        linear_regressor = LinearRegressorGNC_IRLSp(p, rscale, epsilon_base, epsilon_limit, beta, max_niterations=max_niterations)
        assert(linear_regressor.run(data))
        fast_model = linear_regressor.final_model

        # check against slow reference version
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]), max_niterations=max_niterations)
        assert(optimiser_instance.run())
        slow_model = optimiser_instance.final_model

        assert(check_final_models(data[0], fast_model, slow_model))

if __name__ == "__main__":
    test_derivs()
    test_answer()
    test_evaluator_welsch()
    test_evaluator_pseudo_huber()
    test_evaluator_gnc_irls_p()
