import numpy as np

# Returns True if the derivatives are calculated accurately by the model
# instance residual_gradient() function
def check_derivs(
    optimiser_instance,
    model,
    model_ref=None,
    diff_threshold_a: float = 1.0e-7,
    diff_threshold_AlB: float = 1.0e-8,
    print_diffs: bool = False,
    print_derivs: bool = False,
) -> bool:
    # ensure that residual_size is filled in
    optimiser_instance.objective_func(model, model_ref=model_ref)

    # test at different lambda values to make sure all derivatives work
    for lambda_val in (1.0, 0.5, 0.0):
        a, AlB = optimiser_instance.weighted_derivs(
            model, lambda_val, model_ref=model_ref
        )
        optimiser_instance.numeric_derivs_model = True
        optimiser_instance.numeric_derivs_influence = True
        anum, AlBnum = optimiser_instance.weighted_derivs(
            model, lambda_val, model_ref=model_ref
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

        if print_derivs:
            print("AlB:")
            print(AlB)
            print("AlBnum:")
            print(AlBnum)

        if print_diffs:
            print("weighted derivative check lambda=", lambda_val, "AbBdiff:")
            print(AlB - AlBnum)

        for i in range(AlB.shape[0]):
            for j in range(AlB.shape[1]):
                if abs(AlB[i][j] - AlBnum[i][j]) > diff_threshold_AlB:
                    print("Failure AlB i,j=", i, j)
                    return False

        # check against a = dF/dx numerically
        small_diff = 0.0001
        for i in range(len(a)):
            model_copy = np.copy(model)
            model_copy[i] -= small_diff
            totn = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
            model_copy[i] = model[i] + small_diff
            totp = optimiser_instance.objective_func(model_copy, model_ref=model_ref)
            anum[i] = 0.5*(totp - totn)/small_diff

        if print_derivs:
            print("a=",a)
            print("anum=",anum)

        for i in range(len(a)):
            if abs(a[i] - anum[i]) > diff_threshold_a:
                print("Failure a(2) i=", i)
                return False

    return True

# Returns True if the derivatives are calculated accurately for the model
# GNC step size calculation
def check_gnc_derivs(
    optimiser_instance,
    model,
    model_ref=None,
    diff_threshold_aiv: float = 1.0e-7,
    diff_threshold_Aiv: float = 1.0e-8,
    print_diffs: bool = False,
    print_derivs: bool = False,
) -> bool:
    # ensure that residual_size is filled in
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
