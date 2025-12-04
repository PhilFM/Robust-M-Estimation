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
    v = optimiser_instance.objective_func(model, model_ref=model_ref)
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

    return True
