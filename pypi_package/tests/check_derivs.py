import numpy as np

def check_model_derivs(
        model_instance,
        model,
        data,
        *,
        include_2nd_derivs: bool = False,
        small_diff: float = 1.e-4,
        diff_threshold_1st_deriv: float = 1.0e-7,
        diff_threshold_2nd_deriv: float = 1.0e-7,
        ) -> bool:
    update_model_ref = getattr(model_instance, "update_model_ref", None)
    if callable(update_model_ref):
        model_ref = update_model_ref(model)
    else:
        model_ref = None

    model_copy = np.copy(model)
    model_size = model.shape[0]

    # calculate residual size
    model_instance.cache_model(model_copy, model_ref)
    resid_size = len(model_instance.residual(data[0]))

    # calculate derivatives normally
    residual_1st_deriv_arr = np.zeros((len(data), resid_size, model_size))
    for j,d in enumerate(data):
        residual_1st_deriv_arr[j] = model_instance.residual_gradient(d)

    if include_2nd_derivs:
        residual_2nd_deriv_arr = np.zeros((len(data), resid_size, model_size, model_size))
        for j,d in enumerate(data):
            residual_2nd_deriv_arr[j] = model_instance.residual_2nd_deriv(d)

    for i in range(model_size):
        # calculate derivative numerically
        model_copy[i] -= small_diff
        model_instance.cache_model(model_copy, model_ref)
        residual_n_arr = np.zeros((len(data), resid_size))
        if include_2nd_derivs:
            residual_nn_arr = np.zeros((len(data), model_size, resid_size))
            residual_np_arr = np.zeros((len(data), model_size, resid_size))
            residual_pn_arr = np.zeros((len(data), model_size, resid_size))
            residual_pp_arr = np.zeros((len(data), model_size, resid_size))

        for j,d in enumerate(data):
            residual_n_arr[j] = model_instance.residual(d)

        if include_2nd_derivs:
            for k in range(model_size):
                if k != i:
                    model_copy[k] -= small_diff
                    model_instance.cache_model(model_copy, model_ref)
                    for j,d in enumerate(data):
                        residual_nn_arr[j][k] = model_instance.residual(d)

                    model_copy[k] += 2.0*small_diff
                    model_instance.cache_model(model_copy, model_ref)
                    for j,d in enumerate(data):
                        residual_np_arr[j][k] = model_instance.residual(d)

                    model_copy[k] = model[k]

        model_copy[i] += 2.0*small_diff
        model_instance.cache_model(model_copy, model_ref)
        residual_p_arr = np.zeros((len(data), resid_size))
        for j,d in enumerate(data):
            residual_p_arr[j] = model_instance.residual(d)

        if include_2nd_derivs:
            for k in range(model_size):
                if k != i:
                    model_copy[k] -= small_diff
                    model_instance.cache_model(model_copy, model_ref)
                    for j,d in enumerate(data):
                        residual_pn_arr[j][k] = model_instance.residual(d)

                    model_copy[k] += 2.0*small_diff
                    model_instance.cache_model(model_copy, model_ref)
                    for j,d in enumerate(data):
                        residual_pp_arr[j][k] = model_instance.residual(d)

                    model_copy[k] = model[k]

        for j,d in enumerate(data):
            for k in range(resid_size):
                deriv = 0.5*(residual_p_arr[j][k]-residual_n_arr[j][k])/small_diff
                if abs(deriv-residual_1st_deriv_arr[j][k][i]) > diff_threshold_1st_deriv:
                    print("Failure deriv i=",i,"j=",j,"k=",k,"deriv=",deriv,"check against",residual_1st_deriv_arr[j][k][i])
                    return False

        model_copy[i] = model[i]
        if include_2nd_derivs:
            model_instance.cache_model(model_copy, model_ref)
            for j,d in enumerate(data):
                residual_c = model_instance.residual(d)
                for k in range(resid_size):
                    diag_2nd_deriv = (residual_n_arr[j][k] - 2.0*residual_c[k] + residual_p_arr[j][k])/(small_diff*small_diff)
                    if abs(diag_2nd_deriv-residual_2nd_deriv_arr[j][k][i][i]) > diff_threshold_2nd_deriv:
                        print("Failure diag AlB i=",i,"j=",j,"k=",k,"2nd deriv=",diag_2nd_deriv,"check against",residual_2nd_deriv_arr[j][k][i][i])
                        return False

                    for l in range(model_size):
                        if l != i:
                            offdiag_2nd_deriv = 0.25*(residual_nn_arr[j][l][k] - residual_np_arr[j][l][k] - residual_pn_arr[j][l][k] + residual_pp_arr[j][l][k])/(small_diff*small_diff)
                            if abs(offdiag_2nd_deriv-residual_2nd_deriv_arr[j][k][i][l]) > diff_threshold_2nd_deriv:
                                print("Failure offdiag AlB i=",i,"j=",j,"k=",k,"l=",l,"2nd deriv=",offdiag_2nd_deriv,"check against",residual_2nd_deriv_arr[j][k][i][l])
                                return False

    return True

# Returns True if the derivatives are calculated accurately by the model
# instance residual_gradient() function
def check_derivs(
    optimiser_instance,
    model,
    data,
    *,
    weight = None,
    small_diff: float = 1.e-5,
    diff_threshold_a: float = 1.0e-6,
    diff_threshold_AlB: float = 1.0e-3,
    print_diffs: bool = False,
    print_derivs: bool = False,
) -> bool:
    model_ref = None
    include_2nd_derivs = False
    if optimiser_instance._model_instance is not None:
        update_model_ref = getattr(optimiser_instance._model_instance, "update_model_ref", None)
        if callable(update_model_ref):
            model_ref = update_model_ref(model)

        include_2nd_derivs = getattr(optimiser_instance._model_instance, "residual_2nd_deriv", None) is not None

    # ensure that residual_size is filled in
    optimiser_instance._set_data(data, weight=weight)
    optimiser_instance.objective_func(model, model_ref=model_ref)
    
    # test at different lambda values to make sure all derivatives work
    for lambda_val in (1.0, 0.5, 0.0):
        a, AlB = optimiser_instance.weighted_derivs(
            model, lambda_val, include_2nd_derivs=include_2nd_derivs, model_ref=model_ref
        )
        optimiser_instance.numeric_derivs_model = True
        optimiser_instance.numeric_derivs_influence = True
        anum, AlBnum = optimiser_instance.weighted_derivs(
            model, lambda_val, include_2nd_derivs=include_2nd_derivs, model_ref=model_ref
        )
        optimiser_instance.numeric_derivs_model = False
        optimiser_instance.numeric_derivs_influence = False

        if print_derivs:
            print("a:", a)
            print("anum:", anum)

        if print_diffs:
            print("Gradient check: adiff:", a - anum)

        for i in range(len(a)):
            if abs(a[i] - anum[i]) > diff_threshold_a:
                print("Failure a i=", i)
                return False

        if lambda_val == 1.0:
            AlBcheck = np.zeros((len(a),len(a)))
            for i in range(len(a)):
                for j in range(len(a)):
                    if i == j: # diagonal 2nd derivative
                        model_copy = np.copy(model)
                        totc = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
                        model_copy[i] = model[i] - small_diff
                        totn = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
                        model_copy[i] = model[i] + small_diff
                        totp = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
                        AlBcheck[i][i] = (totn - 2.0*totc + totp)/(small_diff*small_diff)
                    else:
                        model_copy[i] = model[i] - small_diff
                        model_copy[j] = model[j] - small_diff
                        totnn = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
                        model_copy[j] = model[j] + small_diff
                        totnp = optimiser_instance.objective_func(model_copy, model_ref=model_ref)

                        model_copy[i] = model[i] + small_diff
                        model_copy[j] = model[j] - small_diff
                        totpn = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
                        model_copy[j] = model[j] + small_diff
                        totpp = optimiser_instance.objective_func(model_copy, model_ref=model_ref)

                        AlBcheck[i][j] = 0.25*(totnn - totnp - totpn + totpp)/(small_diff*small_diff)

        if print_derivs:
            print("AlB:")
            print(AlB)
            print("AlBnum:")
            print(AlBnum)
            if lambda_val == 1.0:
                print("AlBcheck:")
                print(AlBcheck)

        if print_diffs:
            print("weighted derivative check lambda=", lambda_val, "AbBdiff:")
            print(AlB - AlBnum)

        for i in range(AlB.shape[0]):
            for j in range(AlB.shape[1]):
                if abs(AlB[i][j] - AlBnum[i][j]) > diff_threshold_AlB:
                    print("Failure AlB i,j=", i, j)
                    return False

        # check against a = dF/dx numerically
        acheck = np.zeros(len(a))
        for i in range(len(a)):
            model_copy = np.copy(model)
            model_copy[i] -= small_diff
            totn = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
            model_copy[i] = model[i] + small_diff
            totp = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
            acheck[i] = 0.5*(totp - totn)/small_diff

        if print_derivs:
            print("a=",a)
            print("anum=",anum)
            print("acheck=",anum)

        for i in range(len(a)):
            if abs(a[i] - acheck[i]) > diff_threshold_a:
                print("Failure a(2) i=", i, a[i] - acheck[i])
                return False

            if abs(anum[i] - acheck[i]) > diff_threshold_a:
                print("Failure anum i=", i)
                return False

        # check against A+B = d2F/dx2 numerically
        if lambda_val == 1.0:
            for i in range(len(a)):
                for j in range(len(a)):
                    if abs(AlB[i][j] - AlBcheck[i][j]) > diff_threshold_AlB:
                        print("Failure AlB(2) i,j=",i,j, "diff=",AlB[i][j] - AlBcheck[i][j],diff_threshold_AlB)
                        return False

    return True

# Returns True if the derivatives are calculated accurately for the model
# GNC step size calculation
def check_gnc_derivs(
    optimiser_instance,
    model,
    data,
    *,
    model_ref=None,
    weight = None,
    diff_threshold_aiv: float = 1.0e-7,
    diff_threshold_Aiv: float = 1.0e-8,
    print_diffs: bool = False,
    print_derivs: bool = False,
) -> bool:
    # ensure that residual_size is filled in
    optimiser_instance._set_data(data, weight=weight)
    optimiser_instance.objective_func(model, model_ref=model_ref)

    aiv, Aiv = optimiser_instance.weighted_gnc_derivs(
        model, model_ref=model_ref
    )
    optimiser_instance.numeric_derivs_model = True
    optimiser_instance.numeric_derivs_influence = True
    aivnum, Aivnum = optimiser_instance.weighted_gnc_derivs(
        model, model_ref=model_ref
    )
    optimiser_instance.numeric_derivs_model = False
    optimiser_instance.numeric_derivs_influence = False

    if print_derivs:
        print("aiv:", aiv)
        print("aivnum:", aivnum)

    if print_diffs:
        print("Gradient check: aivdiff:", aiv - aivnum)

    for i in range(len(aiv)):
        if abs(aiv[i] - aivnum[i]) > diff_threshold_aiv:
            print("Failure aiv i=", i)
            return False

    if print_derivs:
        print("Aiv:")
        print(Aiv)
        print("Aivnum:")
        print(Aivnum)

    if print_diffs:
        print("weighted GNC derivative check")
        print(Aiv - Aivnum)

    for i in range(Aiv.shape[0]):
        for j in range(Aiv.shape[1]):
            if abs(Aiv[i][j] - Aivnum[i][j]) > diff_threshold_Aiv:
                print("Failure Aiv i,j=", i, j)
                return False

    return True
