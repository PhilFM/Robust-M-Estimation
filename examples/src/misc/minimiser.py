import math
import numpy as np

def minimiser(func: callable,
              initial_centre: np.array,
              initial_half_range: np.array,
              n_samples: list[int],
              *,
              n_iterations: int = 10,
              initial_n_samples: list[int] = None,
              scale_factor: float = 1.0) -> np.array:
    dim = len(initial_half_range)
    assert(dim == len(n_samples))
    assert(initial_n_samples is None or len(initial_n_samples) == dim)
    vmax = 0.0
    xmax = np.zeros(dim)

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

        vals.append(func(initial_sample[s]))

    #print("initial_sample=",initial_sample)
    vidx = np.argmin(vals)
    best_sample = initial_sample[vidx]

    half_range = np.array(initial_half_range) * scale_factor / initial_n_samples
    #print("init best_sample=",best_sample,func(best_sample),"half_range=",half_range)
    for itn in range(n_iterations-1):
        mlist = []
        tot_n_samples = 1
        for i in range(dim):
            mlist.append(np.linspace(best_sample[i] - half_range[i], best_sample[i] + half_range[i], num=n_samples[i]))
            #if itn == 0 and i == 0:
            #    print("  mlist=",mlist)

            tot_n_samples *= n_samples[i]

        sample = np.zeros((tot_n_samples, dim))
        vals = []
        for s in range(tot_n_samples):
            sp = s
            for i in range(dim-1,-1,-1):
                sample[s][i] = mlist[i][sp % n_samples[i]]
                sp //= n_samples[i]

            vals.append(func(sample[s]))

        #print("sample=",sample)
        vidx = np.argmin(vals)
        best_sample = sample[vidx]
        half_range = half_range * scale_factor / n_samples
        #print("next best_sample=",best_sample,func(best_sample),"half_range=",half_range)

    return best_sample,func(best_sample)
