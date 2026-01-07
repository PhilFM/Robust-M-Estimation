import numpy as np
import matplotlib.pyplot as plt
import math
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from trs_welsch import TRSWelsch

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def apply_trs(trs, d, sigma=0.0):
    return (trs[1]*d[0] - trs[0]*d[1] + trs[2] + np.random.normal(0.0,sigma),
            trs[0]*d[0] + trs[1]*d[1] + trs[3] + np.random.normal(0.0,sigma))
            
def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    image_width = 1920
    image_height = 1080
    half_image_width = 0.5*image_width
    half_image_height = 0.5*image_height
    box_size = 900.0
    n_good_points = 100
    n_bad_points = 100

    for test_idx in range(0,1):
        angle_gt = 0.02*np.pi*np.random.rand()
        scale_gt = 0.95 #1.0 + 0.2*randomM11()
        s_gt = scale_gt*math.sin(angle_gt)
        c_gt = scale_gt*math.cos(angle_gt)
        sigma_pop = 2.0

        # model is s,c,tx,ty
        trs_gt = [s_gt, c_gt, -40.0, -10.0] #150.0*randomM11(), 70.0*randomM11()]
        data = np.zeros((n_good_points+n_bad_points,4))
        for i in range(n_good_points):
            while True:
                data[i][0] = randomM11()*half_image_width
                data[i][1] = randomM11()*half_image_height
                (data[i][2],data[i][3]) = apply_trs(trs_gt, data[i], sigma_pop)
                if data[i][2] > -half_image_width and data[i][2] < half_image_width and data[i][3] > -half_image_height and data[i][3] < half_image_height:
                    break

        # add outliers
        for i in range(n_good_points,n_good_points+n_bad_points):
            while True:
                data[i][0] = randomM11()*half_image_width
                data[i][1] = randomM11()*half_image_height
                data[i][2] = data[i][0] + 100.0*randomM11()
                data[i][3] = data[i][1] + 100.0*randomM11()
                if data[i][2] > -half_image_width and data[i][2] < half_image_width and data[i][3] > -half_image_height and data[i][3] < half_image_height:
                    break

        if not test_run:
            print("data=",data)

        p = 0.66667
        sigma_base = sigma_pop/p
        sigma_limit = image_width
        num_sigma_steps = 30
        trs_instance = TRSWelsch(sigma_base, sigma_limit, num_sigma_steps, max_niterations=100, debug=True)
        if trs_instance.run(data):
            trs = trs_instance.final_trs
            final_weight = trs_instance.final_weight

        if not test_run:
            print("trs_gt=",trs_gt,"trs=",trs)
            print("trsDiff=",trs-trs_gt)
            print("n_iterations:",trs_instance.debug_n_iterations)

        # draw first image points with box
        plt.close("all")
        plt.figure(num=1, dpi=120)
        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_axis_off()
        ax.set_xlim((0, image_width))
        ax.set_ylim((0, image_height))
        axis_lwidth = 3.0
        plt.plot((0,image_width),(0,0), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((0,image_width),(image_height,image_height), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((0,0),(0,image_height), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((image_width,image_width),(0,image_height), color=(0,0,0), linewidth=axis_lwidth)

        ref_box_width = 1.02*box_size
        ref_box_height = 1.02*box_size
        ref_box_coords = [[-0.5*ref_box_width, -0.5*ref_box_height],
                          [+0.5*ref_box_width, -0.5*ref_box_height],
                          [-0.5*ref_box_width, +0.5*ref_box_height],
                          [+0.5*ref_box_width, +0.5*ref_box_height]]
        plt.plot((half_image_width+ref_box_coords[0][0],half_image_width+ref_box_coords[1][0]),
                 (half_image_height+ref_box_coords[0][1],half_image_height+ref_box_coords[1][1]), color = "limegreen", label="Box reference")
        plt.plot((half_image_width+ref_box_coords[0][0],half_image_width+ref_box_coords[2][0]),
                 (half_image_height+ref_box_coords[0][1],half_image_height+ref_box_coords[2][1]), color = "limegreen")
        plt.plot((half_image_width+ref_box_coords[1][0],half_image_width+ref_box_coords[3][0]),
                 (half_image_height+ref_box_coords[1][1],half_image_height+ref_box_coords[3][1]), color = "limegreen")
        plt.plot((half_image_width+ref_box_coords[2][0],half_image_width+ref_box_coords[3][0]),
                 (half_image_height+ref_box_coords[2][1],half_image_height+ref_box_coords[3][1]), color = "limegreen")

        msize = 3.0
        plt.plot(half_image_width+data[0][0], half_image_height+data[0][1],
                 color = (1,0,0), marker='o', markersize=msize, label="Inlier data values") # will be overwritten with corrected colour
        plt.plot(half_image_width+data[0][0], half_image_height+data[0][1],
                 color = (0,0,1), marker='o', markersize=msize, label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight)
        for d,w in zip(data,final_weight, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.plot(half_image_width+d[0], half_image_height+d[1], color = color, marker = 'o', markersize=msize)

        plt.savefig(os.path.join(output_folder, "trs_solver.png"), bbox_inches='tight')
        if not test_run:
            plt.show()

        # draw second image points with transformed box
        plt.close("all")
        plt.figure(num=1, dpi=120)
        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_axis_off()
        plt.plot((0,image_width),(0,0), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((0,image_width),(image_height,image_height), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((0,0),(0,image_height), color=(0,0,0), linewidth=axis_lwidth)
        plt.plot((image_width,image_width),(0,image_height), color=(0,0,0), linewidth=axis_lwidth)
        ax.set_xlim((0, image_width))
        ax.set_ylim((0, image_height))

        box_coords = [apply_trs(trs, ref_box_coords[i]) for i in range(4)]
        plt.plot((half_image_width+box_coords[0][0],half_image_width+box_coords[1][0]),
                 (half_image_height+box_coords[0][1],half_image_height+box_coords[1][1]), color = "limegreen", label="Box reference")
        plt.plot((half_image_width+box_coords[0][0],half_image_width+box_coords[2][0]),
                 (half_image_height+box_coords[0][1],half_image_height+box_coords[2][1]), color = "limegreen")
        plt.plot((half_image_width+box_coords[1][0],half_image_width+box_coords[3][0]),
                 (half_image_height+box_coords[1][1],half_image_height+box_coords[3][1]), color = "limegreen")
        plt.plot((half_image_width+box_coords[2][0],half_image_width+box_coords[3][0]),
                 (half_image_height+box_coords[2][1],half_image_height+box_coords[3][1]), color = "limegreen")

        msize = 3.0
        plt.plot(half_image_width+data[0][2], half_image_height+data[0][3],
                 color = (1,0,0), marker='o', markersize=msize, label="Inlier data values") # will be overwritten with corrected colour
        plt.plot(half_image_width+data[0][2], half_image_height+data[0][3],
                 color = (0,0,1), marker='o', markersize=msize, label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight)
        for d,w in zip(data,final_weight, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.plot(half_image_width+d[2], half_image_height+d[3], color = color, marker = 'o', markersize=msize)

        plt.savefig(os.path.join(output_folder, "trs_solver2.png"), bbox_inches='tight')
        if not test_run:
            plt.show()

    if test_run:
        print("trs_solver OK")

if __name__ == "__main__":
    main(False) # test_run
