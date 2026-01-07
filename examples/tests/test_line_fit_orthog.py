import sys
import math
import pytest
import numpy as np

sys.path.append("../../pypi_package/src")
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../../pypi_package/tests")
from evaluator_test import evaluator_test

# supports import of Cython stuff
sys.path.append("../src/cython_files")
from line_fit_orthog_welsch_evaluator import LineFitOrthogWelschEvaluator

sys.path.append("../src/line_fitting")
from line_fit_orthog import LineFitOrthog
from line_fit_orthog_welsch import LineFitOrthogWelsch

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        # ground-truth line parameters
        a_gt = 2.0*(np.random.rand() - 0.5) # x-gradient
        while(True):
            b_gt = 2.0*(np.random.rand() - 0.5) # y-gradient
            if abs(b_gt) > 0.2: # make sure b_gt is not near zero
                break

        norm = math.sqrt(a_gt*a_gt + b_gt*b_gt)
        a_gt /= norm
        b_gt /= norm
        c_gt = 10.0*(np.random.rand()-0.5) # intercept

        # build good data
        n_good_points = np.random.randint(5, 20)
        n_outliers = 1 #int(np.random.rand()*n_good_points)

        x_min = 10.0*np.random.rand()
        x_max = x_min + 3.0 + 5.0*np.random.rand()

        data = np.zeros((n_good_points+n_outliers,2))
        for i in range(n_good_points):
            data[i][0] = x_min + i*(x_max-x_min)/(n_good_points-1)
            data[i][1] = -(a_gt*data[i][0] + c_gt)/b_gt
            #print(a_gt*data[i][0] + b_gt*data[i][1] + c_gt)

        for i in range(n_outliers):
            data[n_good_points+i][0] = x_min + np.random.rand()*(x_max-x_min)
            data[n_good_points+i][1] = -(a_gt*data[i][0] + c_gt)/b_gt + (np.random.rand() - 0.5)

        sigma_base = 0.01
        sigma_limit = 50.0
        num_sigma_steps = 20

        line_fitter = LineFitOrthogWelsch(sigma_base, sigma_limit, num_sigma_steps)
        assert(line_fitter.run(data))
        line = line_fitter.final_line
        lsgn = 1.0 if line[1]*b_gt > 0.0 else -1.0
        #print("line=",line,a_gt,b_gt,c_gt)
        assert(line[0] == pytest.approx(lsgn*a_gt))
        assert(line[1] == pytest.approx(lsgn*b_gt))
        assert(line[2] == pytest.approx(lsgn*c_gt))

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        optimiser_instance = IRLS(param_instance, data, evaluator_instance=LineFitOrthogWelschEvaluator())# print_warnings=True)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        #print("model=",model,a_gt,b_gt,c_gt)
        lsgn = 1.0 if model[1]*b_gt > 0.0 else -1.0
        assert(model[0] == pytest.approx(lsgn*a_gt))
        assert(model[1] == pytest.approx(lsgn*b_gt))
        assert(model[2] == pytest.approx(lsgn*c_gt))

        # compare with slow version
        optimiser_instance = IRLS(param_instance, data, model_instance=LineFitOrthog())
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        #print("model=",model,a_gt,b_gt,c_gt)
        lsgn = 1.0 if model[1]*b_gt > 0.0 else -1.0
        assert(model[0] == pytest.approx(lsgn*a_gt))
        assert(model[1] == pytest.approx(lsgn*b_gt))
        assert(model[2] == pytest.approx(lsgn*c_gt))

def test_evaluator():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_points = np.random.randint(5, 20)

        data = np.zeros((n_points,2))
        weight = np.zeros(n_points)
        scale = np.zeros(n_points)
        for i in range(n_points):
            data[i][0] = np.random.rand()
            data[i][1] = np.random.rand()
            weight[i] = 1.0 #0.2+np.random.rand()
            scale[i] = 1.0+np.random.rand()

        sigma_base = 0.2
        sigma_limit = 50.0
        num_sigma_steps = 20

        # check individual evaluator functions
        evaluator_instance = LineFitOrthogWelschEvaluator()
        influence_func_instance = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps)
        param_instance.reset(False) # sets sigma to sigma_base
        a,b = np.random.rand(), np.random.rand()
        norm = math.sqrt(a*a + b*b)
        a /= norm
        b /= norm
        model = [a, b, np.random.rand()] # a,b,c
        assert(evaluator_test(evaluator_instance, param_instance, LineFitOrthog(), model, data, weight, scale, irls_only=True))

        # fast IRLS
        max_niterations=200
        optimiser_instance = IRLS(param_instance, data, evaluator_instance=LineFitOrthogWelschEvaluator(), weight=weight, scale=scale,
                                  max_niterations=max_niterations)
        assert(optimiser_instance.run())
        model_fast = optimiser_instance.final_model

        # check against slow reference version
        optimiser_instance = IRLS(param_instance, data, model_instance=LineFitOrthog(), weight=weight, scale=scale,
                                  max_niterations=max_niterations)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        assert(model[0] == pytest.approx(model_fast[0]))
        assert(model[1] == pytest.approx(model_fast[1]))
        assert(model[2] == pytest.approx(model_fast[2]))

if __name__ == "__main__":
    test_answer()
    test_evaluator()
