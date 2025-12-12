# assumes N0 good points are followed by N-N0 bad points where N=len(data)
def gncs_draw_data_points(plt, data, weight, x_min, x_max, N0, scale=0.1):
    first_good_point = first_bad_point = True
    ctr = 0

    # combine repeated data values into one
    count = {}
    for d, w in zip(data, weight, strict=True):
        s = str(d)
        if s in count:
            count[s] = count[s] + w
        else:
            count[s] = w

    # print("count=",count)
    for d in data:
        h = scale * count[str(d)]
        # print("d=",d," count=",count[str(d)])
        if d >= x_min and d <= x_max:
            if ctr < N0:
                if first_good_point:
                    plt.axvline(
                        x=d, color="b", ymax=h, label="Good data points", lw=1.0
                    )
                    first_good_point = False
                else:
                    plt.axvline(x=d, color="b", ymax=h, lw=1.0)
            else:
                if first_bad_point:
                    plt.axvline(
                        x=d,
                        color="blueviolet",
                        ymax=h,
                        label="Bad data points",
                        lw=1.0,
                        linestyle="dotted",
                    )
                    first_bad_point = False
                else:
                    plt.axvline(
                        x=d, color="blueviolet", ymax=h, lw=1.0, linestyle="dotted"
                    )

        ctr = ctr + 1
