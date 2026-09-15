import numpy as np

def minimiser_fast(func: callable,
                   initial_centre: np.array,
                   initial_half_range: np.array,
                   n_samples: list[int],
                   *,
                   args = None,
                   n_iterations: int = 10,
                   initial_n_samples: list[int] = None,
                   scale_factor: float = 1.0,
                   debug: bool = False) -> np.array:
    dim = len(initial_half_range)
    assert(dim == len(n_samples))
    assert(initial_n_samples is None or len(initial_n_samples) == dim)

    if initial_n_samples is None:
        initial_n_samples = n_samples

    mlist = []
    tot_n_samples = 1
    for i in range(dim):
        mlist.append(np.linspace(initial_centre[i] - initial_half_range[i], initial_centre[i] + initial_half_range[i], num=initial_n_samples[i]))
        tot_n_samples *= initial_n_samples[i]

    initial_sample = np.zeros((tot_n_samples, dim))
    vals = []
    for s in range(tot_n_samples):
        sp = s
        for i in range(dim-1,-1,-1):
            initial_sample[s][i] = mlist[i][sp % initial_n_samples[i]]
            sp //= initial_n_samples[i]

        if args is None:
            vals.append(func(initial_sample[s]))
        else:
            vals.append(func(initial_sample[s], *args))

    if False: #debug:
        print("initial_sample=",initial_sample)
        print("initial vals",vals)

    vidx = np.argmin(vals)
    best_sample = initial_sample[vidx]

    half_range = np.array(initial_half_range) * scale_factor / initial_n_samples
    if debug:
        print("init best_sample=",best_sample,func(best_sample, *args),"half_range=",half_range)

    for itn in range(n_iterations):
        mlist = []
        tot_n_samples = 1
        for i in range(dim):
            best_val = 1e10
            best_sample_coord = None
            for s in np.linspace(best_sample[i] - half_range[i], best_sample[i] + half_range[i], n_samples[i]):
                best_sample[i] = s
                val = func(best_sample, *args)
                if val < best_val:
                    best_sample_coord = s
                    best_val = val

            best_sample[i] = best_sample_coord

        half_range = half_range * scale_factor / n_samples
        if debug:
            print("itn=",itn,"best_sample,val=",best_sample,func(best_sample,*args),"half_range=",half_range)

    return best_sample,func(best_sample, *args)
