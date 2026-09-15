from line_fit_data_model import LineFitDataModel

class LineFitMethod:
    def __init__(self,
                 data_model:LineFitDataModel):
        self._data_model = data_model
        self._F_deriv_thres = 0.002
        self._error_scale = 10.0
        self._error_scale_sqr = 10.0

    def data_model(self):
        return self._data_model
