import numpy as np
import matplotlib.pyplot as plt

# draw graph of response to changing single parameter, checking for exceeding breakdown point
def check_for_breakdown(max_val, func_v, data, breakdown_thres, image_file, param_name, param_string,
                        test_run, all_good):
    vlist = np.linspace(-max_val, max_val, num=400)

    #print(data)
    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    ax.set_xlabel(r"$" + param_name + "$")
    ax.set_ylabel(r"$F(" + param_string + ")$")

    tot_list = func_v(vlist,data=data)
    max_idx = np.argmax(tot_list)
    max_v = vlist[max_idx]
    y_max = 1.05*tot_list[max_idx]
    ax.set_ylim((0.0, y_max))
    plt.plot(vlist, tot_list, lw = 1.0, color = 'green')

    plt.axvline(x = -breakdown_thres, color = 'b', ymax = y_max, lw = 1.0)
    plt.axvline(x =  breakdown_thres, color = 'b', ymax = y_max, lw = 1.0)
    plt.axvline(x =                0, color = 'r', ymax = y_max, lw = 1.0)
    plt.axvline(x =   max_v, color = 'cyan', ymax = y_max, lw = 1.0)

    #plt.legend()
    plt.savefig(image_file, bbox_inches='tight')
    if not test_run:
        print("Peak " + param_name + " vs. breakdown point threshold ratio:", abs(max_v)/breakdown_thres)

    if abs(max_v) > breakdown_thres:
        all_good = False
        if not test_run:
            plt.show()

    return all_good
