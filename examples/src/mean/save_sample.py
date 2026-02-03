import numpy as np
import matplotlib.pyplot as plt
import os

from weighted_mode import weighted_histogram

def save_sample(data: np.ndarray, sigma_pop: float, output_folder:str, output_file_name:str) -> None:
    bin_size = 0.6*sigma_pop
    x_min,counts = weighted_histogram(data, bin_size=bin_size)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    x_max = x_min + 0.5*(1+len(counts))*bin_size
    #print("x_min/x_max=",x_min,x_max)
    ticks = []
    labels = []
    step = 1
    while True:
        if 10*step < x_max-x_min:
            step *= 5
        else:
            break

        if 10*step < x_max-x_min:
            step *= 2
        else:
            break

    xi_min = step*(int(x_min)//step)
    xi_max = step*(1+int(x_max)//step)
    for x in range(xi_min,xi_max,step):
        ticks.append(2.0*(x-x_min)/bin_size)
        labels.append(x)

    ax.set_xticks(ticks, labels=labels)
    ax.set_ylim(0.0,1.05*max(counts))
    plt.plot(counts)
    #plt.legend()
    plt.savefig(os.path.join(output_folder, output_file_name), bbox_inches='tight')
