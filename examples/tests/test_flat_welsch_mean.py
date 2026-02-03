import sys
import pytest
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../pypi_package/src")
from gnc_smoothie_philfm.base_irls import BaseIRLS
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points

sys.path.append("../src/mean")
from flat_welsch_mean import flat_welsch_mean

def objective_func(m, optimiser_instance) -> float:
    return optimiser_instance.objective_func([m])

def plotResult(optimiser_instance, data, weight, scale, m, x_max, sigma) -> None:
    dmin = dmax = data[0]
    for d in data:
        dmin = min(dmin, d)
        dmax = max(dmax, d)

        # allow border
        drange = dmax-dmin
        xMin = dmin - 0.05*drange
        xMax = dmax + 0.05*drange

    xMin = min(m-0.01,x_max-0.01)
    xMax = max(m+0.01,x_max+0.01)
    mlist = np.linspace(xMin, xMax, num=300)

    plt.figure(num=1, dpi=240)
    gncs_draw_data_points(plt, data, xMin, xMax, len(data), weight=weight, scale=0.05)
    plt.axvline(x = x_max, color = 'gray', label = 'Sampled', lw = 1.0, linestyle = 'solid')

    rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw = 1.0)
    plt.axvline(x = m,   color = 'green',   label = "Flat",   lw = 1.0, linestyle = 'solid')

    plt.legend()
    plt.show()

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        while True:
            n_values = np.random.randint(5, 20)
            data = np.zeros((n_values,1))
            weight = np.zeros(n_values)
            scale = np.zeros(n_values)

            x_range = 5.0+3.0*np.random.rand()
            for i in range(n_values):
                data[i][0] = x_range*np.random.rand()
                weight[i] = np.random.rand()
                scale[i] = 1.0 #+np.random.rand()

            sigma = 0.1+np.random.rand()

            # determine ground truth by sampling
            param_instance = GNC_NullParams(WelschInfluenceFunc(sigma))
            optimiser_instance = BaseIRLS(param_instance, data, model_instance=LinearRegressor(data[0]), weight=weight, scale=scale)

            n_samples = 200
            xlist = np.linspace(0.0, x_range, num=n_samples)
            rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
            ylist = rmfv(xlist, optimiser_instance=optimiser_instance)
            v_max = np.max(ylist)
            idx_max = np.argmax(ylist)
            x_max = xlist[idx_max]

            # reject this sample if the result is too ambiguous
            all_good = True
            for i in range(len(xlist)):
                if abs(i-idx_max) >= 2 and optimiser_instance.objective_func([xlist[i]]) > 0.99*v_max:
                    all_good = False
                    break

            if all_good:
                # second level sampling
                xlist = np.linspace(x_max - 2.0*x_range/n_samples, x_max + 2.0*x_range/n_samples, n_samples)
                rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
                ylist = rmfv(xlist, optimiser_instance=optimiser_instance)
                idx_max = np.argmax(ylist)
                x_max = xlist[idx_max]

                # third level sampling
                xlist = np.linspace(x_max - 2.0*x_range/(n_samples*n_samples), x_max + 2.0*x_range/(n_samples*n_samples), n_samples)
                rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
                ylist = rmfv(xlist, optimiser_instance=optimiser_instance)
                idx_max = np.argmax(ylist)
                x_max = xlist[idx_max]

                y_max = optimiser_instance.objective_func([x_max])
                break

        m = flat_welsch_mean(data, sigma, weight=weight, scale=scale, messages_file=None, output_folder="../../output")
        #plotResult(optimiser_instance, data, weight, scale, m, x_max, sigma)
        assert(m == pytest.approx(x_max, 1.e-4))

if __name__ == "__main__":
    test_answer()
