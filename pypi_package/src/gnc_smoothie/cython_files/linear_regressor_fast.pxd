cimport cython

cdef inline cython.double linear_regressor_calc_residual(cython.double[:] model,
                                                         cython.double[:,:] data_item, cython.double[:] residual):
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    dim: cython.Py_ssize_t = data_item.shape[1]-1
    for j in range(data_item.shape[0]):
        residual[j] = 0.0
        for k in range(dim):
            residual[j] += model[j*data_item.shape[1]+k]*data_item[j][k]

        residual[j] += model[j*data_item.shape[1]+dim] - data_item[j][dim]

    rsqr: cython.double = 0.0
    for j in range(data_item.shape[0]):
        rsqr += residual[j]*residual[j]

    return rsqr

cdef inline linear_regressor_calc_grad(cython.double[:,:] data_item, cython.double[:] residual, cython.double[:,:] grad):
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    dim: cython.Py_ssize_t = data_item.shape[1]-1
    for j in range(data_item.shape[0]):
        for k in range(dim):
            grad[j][k] = data_item[j][k]*residual[j]

        grad[j][dim] = residual[j]

cdef inline increment_weighted_deriv_sums(cython.double lambda_b, cython.double[:,:] data_item, cython.double[:,:] grad,
                                          cython.double rhop, cython.double Bterm, cython.double w,
                                          cython.double[:] atot, cython.double[:,:] AlBtot):
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    m: cython.Py_ssize_t
    dim: cython.Py_ssize_t = data_item.shape[1]-1
    for j in range(data_item.shape[0]):
        offset: cython.Py_ssize_t = j*data_item.shape[1]
        for k in range(dim):
            atot[offset+k] += w * rhop * grad[j][k]
            for l in range(k,dim):
                AlBtot[offset+k][offset+l] += w * (rhop*data_item[j][k]*data_item[j][l] + lambda_b * Bterm * grad[j][k]*grad[j][l])

            AlBtot[offset+k][offset+dim] += w * (rhop*data_item[j][k] + lambda_b * Bterm * grad[j][k]*grad[j][dim])

        atot[offset+dim] += w * rhop * grad[j][dim]
        AlBtot[offset+dim][offset+dim] += w * (rhop + lambda_b * Bterm * grad[j][dim]*grad[j][dim])
        for m in range(j+1,data_item.shape[0]):
            offsetp: cython.Py_ssize_t = m*data_item.shape[1]
            for k in range(dim):
                for l in range(dim):
                    AlBtot[offset+k][offsetp+l] += w * lambda_b * Bterm * grad[j][k]*grad[m][l]

                AlBtot[offset+k][offsetp+dim] += w * lambda_b * Bterm * grad[j][k]*grad[m][dim]
                        
            for l in range(dim):
                AlBtot[offset+dim][offsetp+l] += w * lambda_b * Bterm * grad[j][dim]*grad[m][l]

            AlBtot[offset+dim][offsetp+dim] += w * lambda_b * Bterm * grad[j][dim]*grad[m][dim]

cdef inline increment_weighted_gnc_deriv_sums(cython.double[:,:] data_item, cython.double[:,:] grad,
                                              cython.double rhop, cython.double rhopiv, cython.double Bterm, cython.double w,
                                              cython.double[:] aivtot, cython.double[:,:] Aivtot):
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    m: cython.Py_ssize_t
    dim: cython.Py_ssize_t = data_item.shape[1]-1
    for j in range(data_item.shape[0]):
        offset: cython.Py_ssize_t = j*data_item.shape[1]
        for k in range(dim):
            aivtot[offset+k] += w * rhopiv * grad[j][k]
            for l in range(k,dim):
                Aivtot[offset+k][offset+l] += w * (rhop*data_item[j][k]*data_item[j][l] + Bterm * grad[j][k]*grad[j][l])

            Aivtot[offset+k][offset+dim] += w * (rhop*data_item[j][k] + Bterm * grad[j][k]*grad[j][dim])

        aivtot[offset+dim] += w * rhopiv * grad[j][dim]
        Aivtot[offset+dim][offset+dim] += w * (rhop + Bterm * grad[j][dim]*grad[j][dim])
        for m in range(j+1,data_item.shape[0]):
            offsetp: cython.Py_ssize_t = m*data_item.shape[1]
            for k in range(dim):
                for l in range(dim):
                    Aivtot[offset+k][offsetp+l] += w * Bterm * grad[j][k]*grad[m][l]

                Aivtot[offset+k][offsetp+dim] += w * Bterm * grad[j][k]*grad[m][dim]
                        
            for l in range(dim):
                Aivtot[offset+dim][offsetp+l] += w * Bterm * grad[j][dim]*grad[m][l]

            Aivtot[offset+dim][offsetp+dim] += w * Bterm * grad[j][dim]*grad[m][dim]

