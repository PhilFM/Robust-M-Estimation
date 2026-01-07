import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

def objective_func(x, optimiser_instance):
    if optimiser_instance.objective_func_sign() < 0.0:
        return 1.0-optimiser_instance.objective_func([x])
    else:
        return optimiser_instance.objective_func([x])

def gradient(x, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([x], 1.0) # lambda_val
    return optimiser_instance.objective_func_sign()*a[0]

def centred_quadratic(x, a, c):
    return a*x*x + c

def draw_majorizer(plt, mlist, u, rhou, rhopu, colour):
    a = 0.5*rhopu/u
    c = rhou - a*u*u
    rmfv = np.vectorize(centred_quadratic, excluded={"a", "c"})
    plt.plot(mlist, rmfv(mlist, a=a, c=c), color = colour, lw = 1.0, label = "Majorizer u=" + str(u))

def plot_result(optimiser_instance, u_values:list, label:str, output_folder, file_name:str, test_run:bool):
    x_max = 3.0
    mlist = np.linspace(-x_max, x_max, num=300)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()

    colours = ["blue", "purple"]
    for u,col in zip(u_values,colours, strict=True):
        draw_majorizer(plt, mlist, u, objective_func(u,optimiser_instance), gradient(u,optimiser_instance), col)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw=1.0, label=label)

    ax.set_xlabel(r'r')

    plt.legend()
    plt.savefig(os.path.join(output_folder, file_name + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../../output"):
    model_instance = LinearRegressor(np.array([0.]))
    
    p = 0.0
    rscale = 1.0
    epsilon = 0.1
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0], numeric_derivs_influence=True),
                [1.0, 2.0], "GNC IRLSp0 influence function", output_folder, "gnc_irls_p0_majorizers", test_run)

    p = 0.5
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0], numeric_derivs_influence=True),
                [1.0, 2.0], "GNC IRLSp0.5 influence function", output_folder, "gnc_irls_ph_majorizers", test_run)

    p = 1.0
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0], numeric_derivs_influence=True),
                [1.0, 2.0], "GNC IRLSp1 influence function", output_folder, "gnc_irls_p1_majorizers", test_run)

    sigma = 1.0
    param_instance = GNC_NullParams(WelschInfluenceFunc(sigma))
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0]),
                [1.5, 2.0], "Welsch influence function", output_folder, "welsch_majorizers", test_run)

    param_instance = GNC_NullParams(PseudoHuberInfluenceFunc(sigma))
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0]),
                [1.5, 2.0], "Pseudo-Huber influence function", output_folder, "pseudo_huber_majorizers", test_run)
    
    param_instance = GNC_NullParams(GemanMcClureInfluenceFunc(sigma))
    plot_result(SupGaussNewton(param_instance, np.array([[0.0]]), model_instance=model_instance, weight=[1.0]),
                [1.0, 2.0], "Geman-McClure influence function", output_folder, "geman_mcclure_majorizers", test_run)

    if test_run:
        print("majorize_examples OK")

if __name__ == "__main__":
    main(False) # test_run
