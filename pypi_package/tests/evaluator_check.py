import sys
sys.path.append("../pypi_package/src")

import pytest
import numpy as np
import numpy.typing as npt

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS

def evaluator_check(evaluator_instance, param_instance, model_instance,
                    model: npt.ArrayLike, data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike,
                    irls_only: bool=False) -> bool:
    # reference slow instance
    if irls_only:
        optimiser_instance = IRLS(param_instance, data, model_instance=model_instance, weight=weight, scale=scale)
    else:
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, weight=weight, scale=scale)

    # check objective_func()
    objv = evaluator_instance.objective_func(model, None, param_instance.influence_func_instance, [data], [weight], [scale])
    objvp = optimiser_instance.objective_func(model)
    assert(objv == pytest.approx(objvp))

    # check weighted_derivs()
    model_size = len(model)
    if not irls_only:
        for lambda_val in (0.0, 0.5, 1.0):
            #print("lambda_val=",lambda_val)
            atot, AlBtot = evaluator_instance.weighted_derivs(model, None, param_instance.influence_func_instance,
                                                              lambda_val, [data], [weight], [scale])
            atotp, AlBtotp = optimiser_instance.weighted_derivs(model, lambda_val)
            #print("atot=",atot)
            #print("atotp=",atotp)
            #print("AlBtot=",AlBtot)
            #print("AlBtotp=",AlBtotp)
            for i in range(model_size):
                assert(atot[i] == pytest.approx(atotp[i]))
                for j in range(model_size):
                    assert(AlBtot[i][j] == pytest.approx(AlBtotp[i][j]))

    # check update_weights()
    data_size = len(data)
    new_weight = np.zeros(data_size)
    evaluator_instance.update_weights(model, None, param_instance.influence_func_instance,
                                      [data], [weight], [scale], [new_weight])
    new_weightp = np.zeros(data_size)
    optimiser_instance.update_weights(model, [new_weightp])
    for i in range(data_size):
        assert(new_weight[i] == pytest.approx(new_weightp[i]))

    # check weighted_fit()
    model_fit,model_ref = evaluator_instance.weighted_fit([data], [weight], [scale])
    if irls_only:
        model_fitp,model_refp = optimiser_instance.model_weighted_fit()
    else:
        model_fitp,model_refp = optimiser_instance.weighted_fit()

    #print("model_fit=",model_fit)
    #print("model_fitp=",model_fitp)
    for i in range(model_size):
        assert(model_fit[i] == pytest.approx(model_fitp[i]))

    return True

