modelValues = {}

key = ("GroundTruth", "", "")
modelValues[key] = {}
modelValues[key]["label"] = "Ground truth"
modelValues[key]["colour"] = "gray"
modelValues[key]["linestyle"] = "solid"
modelValues[key]["lw"] = 1.0

key = ("SupGN", "Welsch", "GNC_Welsch")
modelValues[key] = {}
modelValues[key]["label"] = "SUP-GN Welsch"
modelValues[key]["colour"] = "green"
modelValues[key]["linestyle"] = "solid"
modelValues[key]["lw"] = 1.0

key = ("IRLS", "Welsch", "GNC_Welsch")
modelValues[key] = {}
modelValues[key]["label"] = "GNC-W"
modelValues[key]["colour"] = "magenta"
modelValues[key]["linestyle"] = (0, (1, 1))
modelValues[key]["lw"] = 1.0

key = ("Flat", "Welsch", "GNC_Welsch")
modelValues[key] = {}
modelValues[key]["label"] = "Flat Welsch"
modelValues[key]["colour"] = "green"
modelValues[key]["linestyle"] = "dotted"
modelValues[key]["lw"] = 1.5

key = ("SupGN", "PseudoHuber", "Welsch")
modelValues[key] = {}
modelValues[key]["label"] = "SUP-GN Pseudo-Huber"
modelValues[key]["colour"] = "brown"
modelValues[key]["linestyle"] = "dashed"
modelValues[key]["lw"] = 1.0

key = ("IRLS", "PseudoHuber", "Welsch")
modelValues[key] = {}
modelValues[key]["label"] = "Pseudo-Huber IRLS"
modelValues[key]["colour"] = "blue"
modelValues[key]["linestyle"] = (0, (5, 1))
modelValues[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
modelValues[key] = {}
modelValues[key]["label"] = "GNC IRLS-p0"
modelValues[key]["colour"] = "purple"
modelValues[key]["linestyle"] = "dashdot"
modelValues[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp1")
modelValues[key] = {}
modelValues[key]["label"] = "GNC IRLS-p1"
modelValues[key]["colour"] = "mediumpurple"
modelValues[key]["linestyle"] = (0, (5, 1, 1, 1))
modelValues[key]["lw"] = 1.0

key = ("Mean", "Basic", "")
modelValues[key] = {}
modelValues[key]["label"] = "Arithmetic mean"
modelValues[key]["colour"] = "darkorange"
modelValues[key]["linestyle"] = (0, (2, 2))
modelValues[key]["lw"] = 1.0

key = ("Mean", "Trimmed", "")
modelValues[key] = {}
modelValues[key]["label"] = "Trimmed Mean"
modelValues[key]["colour"] = "darkkhaki"
modelValues[key]["linestyle"] = (0, (3, 1))
modelValues[key]["lw"] = 1.0

key = ("Median", "Basic", "")
modelValues[key] = {}
modelValues[key]["label"] = "Median"
modelValues[key]["colour"] = "deeppink"
modelValues[key]["linestyle"] = (0, (3, 1, 1, 1))
modelValues[key]["lw"] = 1.0

key = ("RME", "", "")
modelValues[key] = {}
modelValues[key]["label"] = "RME"
modelValues[key]["colour"] = "deepskyblue"
modelValues[key]["linestyle"] = (0, (4, 1))
modelValues[key]["lw"] = 1.0


def drawVLine(plt, x: float, key, useLabel: bool = True, useLineStyle: bool = True):
    values = modelValues[key]
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


def drawCurve(
    plt,
    vals,
    key,
    xvalues=None,
    drawMarkers: bool = True,
    hlightXValue: float = None,
    ax=None,
):
    values = modelValues[key]
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
