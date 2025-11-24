algValues = {}

key = ("GroundTruth", "", "")
algValues[key] = {}
algValues[key]["label"] = "Ground truth"
algValues[key]["colour"] = "gray"
algValues[key]["linestyle"] = "solid"
algValues[key]["lw"] = 1.0

key = ("IRLS", "Welsch", "GNC_Welsch")
algValues[key] = {}
algValues[key]["label"] = "GNC-W"
algValues[key]["colour"] = "magenta"
algValues[key]["linestyle"] = (0, (1,1))
algValues[key]["lw"] = 1.0

key = ("Flat", "Welsch", "GNC_Welsch")
algValues[key] = {}
algValues[key]["label"] = "Flat Welsch"
algValues[key]["colour"] = "green"
algValues[key]["linestyle"] = "dotted"
algValues[key]["lw"] = 1.5

key = ("IRLS", "PseudoHuber", "Welsch")
algValues[key] = {}
algValues[key]["label"] = "Pseudo-Huber IRLS"
algValues[key]["colour"] = "blue"
algValues[key]["linestyle"] = (0, (5, 1))
algValues[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
algValues[key] = {}
algValues[key]["label"] = "GNC IRLS-p0"
algValues[key]["colour"] = "purple"
algValues[key]["linestyle"] = "dashdot"
algValues[key]["lw"] = 1.0

key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp1")
algValues[key] = {}
algValues[key]["label"] = "GNC IRLS-p1"
algValues[key]["colour"] = "purple"
algValues[key]["linestyle"] = (0, (5,1,1,1))
algValues[key]["lw"] = 1.0

key = ("IRLS", "GNC_TLS", "GNC_TLS")
algValues[key] = {}
algValues[key]["label"] = "GNC TLS"
algValues[key]["colour"] = "cyan"
algValues[key]["linestyle"] = (0, (2,3,1,1))
algValues[key]["lw"] = 1.0

key = ("Mean", "Basic", "")
algValues[key] = {}
algValues[key]["label"] = "Arithmetic mean"
algValues[key]["colour"] = "darkorange"
algValues[key]["linestyle"] = (0, (2,2))
algValues[key]["lw"] = 1.0

key = ("Mean", "Trimmed", "")
algValues[key] = {}
algValues[key]["label"] = "Trimmed Mean"
algValues[key]["colour"] = "darkkhaki"
algValues[key]["linestyle"] = (0, (3,1))
algValues[key]["lw"] = 1.0

key = ("Median", "Basic", "")
algValues[key] = {}
algValues[key]["label"] = "Median"
algValues[key]["colour"] = "deeppink"
algValues[key]["linestyle"] = (0, (3, 1, 1, 1))
algValues[key]["lw"] = 1.0

key = ("RME", "", "")
algValues[key] = {}
algValues[key]["label"] = "RME"
algValues[key]["colour"] = "deepskyblue"
algValues[key]["linestyle"] = (0, (4, 1))
algValues[key]["lw"] = 1.0

def drawVLine(plt, x : float, key, useLabel: bool=True, useLineStyle: bool=True):
    values = algValues[key]
    if useLabel:
        if useLineStyle:
            plt.axvline(x=x, color = values["colour"], label = values["label"], lw = values["lw"], linestyle = values["linestyle"]) #, marker = 'o', markersize = 2.0)
        else:
            plt.axvline(x=x, color = values["colour"], label = values["label"], lw = values["lw"]) #, marker = 'o', markersize = 2.0)
    else:
        if useLineStyle:
            plt.axvline(x=x, color = values["colour"], lw = values["lw"], linestyle = values["linestyle"]) #, marker = 'o', markersize = 2.0)
        else:
            plt.axvline(x=x, color = values["colour"], lw = values["lw"]) #, marker = 'o', markersize = 2.0)

def drawCurve(plt, vals, key, xvalues = None, drawMarkers:bool=True, hlightXValue:float=None, ax=None):
    values = algValues[key]
    if xvalues is None:
        if drawMarkers:
            plt.plot(list(range(len(vals))), vals, color = values["colour"], label = values["label"], lw = values["lw"], linestyle = values["linestyle"], marker = 'o', markersize = 2.0)
        else:
            plt.plot(list(range(len(vals))), vals, color = values["colour"], label = values["label"], lw = values["lw"], linestyle = values["linestyle"])
    else:
        if drawMarkers:
            plt.plot(xvalues, vals, color = values["colour"], label = values["label"], lw = values["lw"], linestyle = values["linestyle"], marker = 'o', markersize = 2.0)
        else:
            plt.plot(xvalues, vals, color = values["colour"], label = values["label"], lw = values["lw"], linestyle = values["linestyle"])

        if hlightXValue is not None:
            xprev = xvalues[0]
            yprev = vals[0]
            for x,y in zip(xvalues,vals):
                if xprev < hlightXValue and x >= hlightXValue:
                    # interpolate y
                    alpha = (hlightXValue - xprev)/(x - xprev)
                    yint = yprev + alpha*(y-yprev)
                    #circle = plt.Circle((x,y), 1, color = values["colour"])
                    #add_patch(circle)
                    ax.scatter([hlightXValue],[yint],s=[10],color = values["colour"])

                xprev = x
                yprev = y
                
        
