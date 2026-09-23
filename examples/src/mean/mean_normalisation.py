import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.projective_normalisation import ProjectiveNormalisation

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y,z] triplets
    mean_gt = 0.5

    sigma_pop = 0.1
    n_points = 100
    alpha = 0.6
    n_bad_points = int(alpha*n_points)
    n_good_points = n_points-n_bad_points
    data = np.zeros(n_points)
    for i in range(n_good_points):
        data[i] = mean_gt + np.random.normal(0.0, sigma_pop)
        #data[i] = mean_gt -sigma_pop + 2.0*sigma_pop*i/(n_points-1)

    for i in range(n_good_points,n_points):
        data[i] = 100.0 + 0.001*i

    # convert to 2D array
    data_2d = np.zeros((n_points,2))
    for i in range(n_points):
        data_2d[i][0] = data[i]
        data_2d[i][1] = -1.0

    pnorm = ProjectiveNormalisation()
    pnorm.run(data_2d)
    assert(pnorm.error_string is None)

    # check that normalisation worked
    tot_s = np.zeros((2,2))
    for v in pnorm.ndata:
        tot_s += np.outer(v,v)

    print("tot_s=",tot_s)

    # draw result on a circle
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    #ax.set_ylim(0.0,xy_range)
    #ax.set_xlim(0.0,xy_range)
    ax.set_aspect('equal')

    circle1 = plt.Circle((0.0,0.0), radius=1.0, color="gray", fill=False, linewidth=0.3)
    ax.add_patch(circle1)

    for d in pnorm.ndata:
        ax.plot(d[0], d[1], "o", markersize=1, color="g")

    plt.savefig(os.path.join(output_folder, "mean_normalisation.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("mean_normalisation OK")

if __name__ == "__main__":
    main(False) # test_run
