import numpy as np

def linear_regressor_convert_data(data: tuple) -> np.array:
    assert(len(data) == 2)
    data_x = data[0]
    data_y = data[1]
    assert(len(data_x) == len(data_y))
    if data_y.ndim == 1:
        data_y = np.reshape(data_y, (len(data_y),1,1))
        rsize = 1
    else:
        assert(data_y.ndim == 2)
        rsize = len(data_y[0])
        data_y = np.reshape(data_y, (len(data_y),rsize,1))

    if data_x.ndim == 1:
        assert(rsize == 1)
        data_x = np.reshape(data_x, (len(data_x),1,1))
    elif data_x.ndim == 2:
        if rsize == 1:
            data_x = np.reshape(data_x, (len(data_x),1,len(data_x[0])))
        else:
            assert(rsize == len(data_x[0]))
            data_x = np.reshape(data_x, (len(data_x),rsize,1,))
    else:
        assert(data_x.ndim == 3)

    #print("data_x.shape:",data_x.shape)
    #print("data_y.shape:",data_y.shape)
    return np.concatenate((data_x, data_y), axis=2)

def linear_regressor_convert_model(model: np.array, data_item) -> (np.array,np.array):
    if data_item.ndim == 2:
        rsize = len(data_item)
        msize = len(data_item[0])
    else:
        assert(data_item.ndim == 1)
        rsize = 1
        msize = len(data_item)

    modelp = model.reshape((rsize,msize))
    return (modelp[:,:msize-1], modelp[:,msize-1:].reshape(rsize))

