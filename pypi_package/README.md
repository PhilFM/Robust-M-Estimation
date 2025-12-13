# GNC Smoothie package

Python library supporting M-estimation using two algorithms: the well-known Iteratively Reweighted Least Squares (IRLS)
and our custom Supervised Gauss-Newton algorithm. Author: Philip McLauchlan `philipmclauchlan6@gmail.com`.

First some introductory theory.

## M-estimation
M-estimation is a generalisation of maximum-likelihood estimation. 
We assume a known population probability density function (PDF) $f(.)$, parametrised by a vector of parameters ${\bf x}$,
and a set of independent and identically distributed data ${\bf z}_i$, $i=1,...,n$ sampled from the population.
The general model for the observations is

$${\bf z}_i = {\bf h}_i({\bf x}) + \text{noise}
$$
