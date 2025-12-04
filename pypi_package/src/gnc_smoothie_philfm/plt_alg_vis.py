model_values = {}

key = ("GroundTruth", "", "")
model_values[key] = {}
model_values[key]["label"] = "Ground truth"
model_values[key]["colour"] = "gray"
model_values[key]["linestyle"] = "solid"
model_values[key]["lw"] = 1.0

key = ("SupGN", "Welsch", "GNC_Welsch")
model_values[key] = {}
model_values[key]["label"] = "SUP-GN Welsch"
model_values[key]["colour"] = "green"
model_values[key]["linestyle"] = "solid"
model_values[key]["lw"] = 1.0

key = ("IRLS", "Welsch", "GNC_Welsch")
model_values[key] = {}
model_values[key]["label"] = "GNC-W"
model_values[key]["colour"] = "magenta"
model_values[key]["linestyle"] = (0, (1, 1))
model_values[key]["lw"] = 1.0

key = ("Flat", "Welsch", "GNC_Welsch")
model_values[key] = {}
model_values[key]["label"] = "Flat Welsch"
model_values[key]["colour"] = "green"
model_values[key]["linestyle"] = "dotted"
model_values[key]["lw"] = 1.5

key = ("SupGN", "PseudoHuber", "Welsch")
model_values[key] = {}
model_values[key]["label"] = "SUP-GN Pseudo-Huber"
model_values[key]["colour"] = "brown"
model_values[key]["linestyle"] = "dashed"
model_values[key]["lw"] = 1.0

key = ("IRLS", "PseudoHuber", "Welsch")
model_values[key] = {}
model_values[key]["label"] = "Pseudo-Huber IRLS"
model_values[key]["colour"] = "blue"
model_values[key]["linestyle"] = (0, (5, 1))
model_values[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
model_values[key] = {}
model_values[key]["label"] = "GNC IRLS-p0"
model_values[key]["colour"] = "purple"
model_values[key]["linestyle"] = "dashdot"
model_values[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp1")
model_values[key] = {}
model_values[key]["label"] = "GNC IRLS-p1"
model_values[key]["colour"] = "mediumpurple"
model_values[key]["linestyle"] = (0, (5, 1, 1, 1))
model_values[key]["lw"] = 1.0

key = ("Mean", "Basic", "")
model_values[key] = {}
model_values[key]["label"] = "Arithmetic mean"
model_values[key]["colour"] = "darkorange"
model_values[key]["linestyle"] = (0, (2, 2))
model_values[key]["lw"] = 1.0

key = ("Mean", "Trimmed", "")
model_values[key] = {}
model_values[key]["label"] = "Trimmed Mean"
model_values[key]["colour"] = "darkkhaki"
model_values[key]["linestyle"] = (0, (3, 1))
model_values[key]["lw"] = 1.0

key = ("Median", "Basic", "")
model_values[key] = {}
model_values[key]["label"] = "Median"
model_values[key]["colour"] = "deeppink"
model_values[key]["linestyle"] = (0, (3, 1, 1, 1))
model_values[key]["lw"] = 1.0

key = ("RME", "", "")
model_values[key] = {}
model_values[key]["label"] = "RME"
model_values[key]["colour"] = "deepskyblue"
model_values[key]["linestyle"] = (0, (4, 1))
model_values[key]["lw"] = 1.0


def gncs_draw_vline(plt, x: float, key, useLabel: bool = True, useLineStyle: bool = True):
    values = model_values[key]
    if useLabel:
        if useLineStyle:
            plt.axvline(
                x=x,
                color=values["colour"],
                label=values["label"],
                lw=values["lw"],
                linestyle=values["linestyle"],
            )  # , marker = 'o', markersize = 2.0)
        else:
            plt.axvline(
                x=x, color=values["colour"], label=values["label"], lw=values["lw"]
            )  # , marker = 'o', markersize = 2.0)
    else:
        if useLineStyle:
            plt.axvline(
                x=x,
                color=values["colour"],
                lw=values["lw"],
                linestyle=values["linestyle"],
            )  # , marker = 'o', markersize = 2.0)
        else:
            plt.axvline(
                x=x, color=values["colour"], lw=values["lw"]
            )  # , marker = 'o', markersize = 2.0)


def gncs_draw_curve(
    plt,
    vals,
    key,
    xvalues=None,
    drawMarkers: bool = True,
    hlightXValue: float = None,
    ax=None,
):
    values = model_values[key]
    if xvalues is None:
        if drawMarkers:
            plt.plot(
                list(range(len(vals))),
                vals,
                color=values["colour"],
                label=values["label"],
                lw=values["lw"],
                linestyle=values["linestyle"],
                marker="o",
                markersize=2.0,
            )
        else:
            plt.plot(
                list(range(len(vals))),
                vals,
                color=values["colour"],
                label=values["label"],
                lw=values["lw"],
                linestyle=values["linestyle"],
            )
    else:
        if drawMarkers:
            plt.plot(
                xvalues,
                vals,
                color=values["colour"],
                label=values["label"],
                lw=values["lw"],
                linestyle=values["linestyle"],
                marker="o",
                markersize=2.0,
            )
        else:
            plt.plot(
                xvalues,
                vals,
                color=values["colour"],
                label=values["label"],
                lw=values["lw"],
                linestyle=values["linestyle"],
            )

        if hlightXValue is not None:
            xprev = xvalues[0]
            yprev = vals[0]
            for x, y in zip(xvalues, vals, strict=True):
                if xprev < hlightXValue and x >= hlightXValue:
                    # interpolate y
                    alpha = (hlightXValue - xprev) / (x - xprev)
                    yint = yprev + alpha * (y - yprev)
                    # circle = plt.Circle((x,y), 1, color = values["colour"])
                    # add_patch(circle)
                    ax.scatter([hlightXValue], [yint], s=[10], color=values["colour"])

                xprev = x
                yprev = y
