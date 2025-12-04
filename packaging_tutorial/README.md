# GNC Smoothie package

Python library supporting M-estimation using two algorithms: the well-known Iteratively Reweighted Least Squares (IRLS)
and our custom Supervised Gauss-Newton algorithm. First some introductory theory.

## M-estimation
M-estimation is a generalisation of maximum-likelihood estimation. 
We assume a known population probability density function (PDF) \\( f(.) \\), parametrised by a vector of parameters \\( {\bf x} \\),
and a set of independent and identically distributed data \\( {\bf z}_i \\), \\( i=1,...,n \\) sampled from the population.
The general model for the observations is
\\[
  {\bf z}_i = {\bf h}_i({\bf x}) + \text{noise}
\\]
for some observation model function \\( {\bf h}_i({\bf x}) \\).
The distribution of the noise determined by the population PDF, which is defined as some function
\\[
  f({\bf z}_i - {\bf h}_i({\bf x})) = f({\bf r}_i)
\\]
defining the \\( i \\)'th data error or "residual" vector \\( {\bf r}_i \\) as
\\[
  {\bf r}_i({\bf x}) = {\bf z}_i - {\bf h}_i({\bf x})
\\]

For instance for Normal distributed observation errors with standard deviation \\( \sigma \\) we would have
\\[
  f({\bf r}_i) = \\,\mathrm{e}^{-\frac{|| {\bf r}_i ||^2}{2\sigma^2}}
\\]

The maximum likelihood estimator of \\( {\bf x} \\) can be computed as 

\\[
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\\,\text{max}} \left( \prod_{i=1}^n f({\bf r}_i({\bf x})) \right)
\\]

or equivalently,
\\[
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\\,\text{min}} \left( \sum_{i=1}^n - \log f({\bf r}_i({\bf x})) \right)
\\]

M-estimation generalises this method by substituting a different function into the above sum, so we instead compute
\\[
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\\,\text{min}} \left( \sum_{i=1}^n \rho(|| {\bf r}_i({\bf x}) ||) \right)
\\]
for some function \\( \rho(r_i) \\) where
\\[
  r_i = || {\bf r}_i({\bf x}) ||
\\]

We write the objective function above as

\\[
  F({\bf x}) = \sum_{i=1}^n \rho(|| {\bf r}_i({\bf x}) ||)
\\]

or
\\[
  F({\bf x}) = \sum_{i=1}^n \rho(r_i({\bf x}))
\\]

In the special case of Normally distributed observation errors, giving rise to standard least-squares,
\\( \rho(r) \sim r^2 \\), the squared error in the observations.
The development and popularisation of M-estimation was driven by the need to fit models to data with outliers, i.e.
data not sampled from the population pdf but from a distinct distribution or distributions.
When outliers are present the least-squares method breaks down because single outliers
can have a huge influence, leading to a wildly incorrect value for \\( \widehat{\bf x} \\).
To allow for outliers \\( \rho(r) \\) is shaped by reducing the value of \\( \rho(r) \\) for large \\( r \\) error values.
The choice of influence function \\( \psi(r) = d\rho(r)/dr \\) is driven by a trade-off between the desire to
provide a good accuracy in the resulting
estimate for \\( \widehat{\bf x} \\) while providing robustness to noise in the data.
For instance, instead of the quadratic function required for least-squares, the pseudo-Huber influence
function [1] is asymptotically linear in order to provide some level of robustness. 

## The Welsch influence function

Redescending influence functions have the property that their gradient tends to zero at either ends of the range.
This allows them to be robust to outliers with large errors. On the negative side, redescending influence functions
have the problem that the objective function above minimised by M-estimation may have multiple local minima.
It is difficult to ensure that the global minimum is reached. When the standard method "iteratively reweighted least-squares"
(IRLS) is used, the result will depend on the quality of the initial value of \\( {\bf x} \\)
used for the iteration.
    
We start with the Welsch influence function [2]. This uses a negative Gaussian:
\\[
  \rho(r) = \frac{\sigma^2}{2} \left( 1 - \\,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \right)
\\]
\\[
  \psi(r) = \frac{d\rho(r)}{dr} = \frac{r}{2} \\,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
\\]
where the width \\( \sigma \\) of the Gaussian is known as the "wavelength" of the Welsch influence function.
Using a Gaussian influence function, whose gradient tends to zero for large errors, ensures robustness to large errors,
because their influence on the solution will be very small.
However in general it is presumed in the literature that solving M-estimation using redescending influence functions
requires a good initial estimate of the solution and comes with no guarantees of convergence. Recent work [3] in IRLS using
Graduated-Non-Convextity [4] has clarified that this is not always the case, and for many practical problems we can
achieve the *global* optimum solution without any initial model estimate being provided.

## Iteratively reweighted least squares (IRLS)

IRLS is the standard technique used to solve the non-linear optimisation problems that arise in M-estimation using robust
influence functions. IRLS assumes that the non-robust least-squares solution
for \\( {\bf x} \\) is soluble in closed form, given some "weights" assigned
to the data points. In other words there is a (simple) algorithm that can be used to solve the optimisation problem

\\[
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\\,\text{min}} \left( \sum_{i=1}^n w_i || {\bf z}_i - {\bf h}_i({\bf x}) ||^2)\right)
\\]

or
\\[
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\\,\text{min}} \left( F_{\text{LS}}({\bf x}) \right)
\\]

where
\\[
  F_{\text{LS}}({\bf x}) = \sum_{i=1}^n w_i r_i({\bf x})^2 = \sum_{i=1}^n w_i ||{\bf z}_i - {\bf h}_i({\bf x})||^2
\\]

for weights \\( w_i \\).
IRLS is based on the observation that the solution \\( \widehat{\bf x} \\) must be a stationary point of the
objective function \\( F_{\text{LS}} \\) in the above equation, so we must have

\\[
  \frac{dF_{\text{LS}}}{d{\bf x}} = \sum_{i=1}^n w_i r_i \frac{dr_i}{d{\bf x}} = {\bf 0}
\\]

The stationary point condition for solving the original optimisation problem for \\( \widehat{\bf x} \\) is the similar equation
\\[
  \frac{dF}{d{\bf x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{d{\bf x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{dr_i}\frac{dr_i}{d{\bf x}} = \sum_{i=1}^n \psi(r_i)\frac{dr_i}{d{\bf x}} = {\bf 0}
\\]

Comparing the equations involving \\( \frac{dF_{\text{LS}}}{d{\bf x}} \\) and \\( \frac{dF}{d{\bf x}} \\) above,
we see that the appropriate weight to use is
\\[
  w_i = \frac{1}{r_i} \psi(r_i)
\\]

This choice will ensure that solving for \\( \frac{dF_{\text{LS}}}{d{\bf x}} \\) will also solve for \\( \frac{dF}{d{\bf x}} \\)
to first order, and hopefully improve the solution.
IRLS repeats the following two steps to convergence, given an initial state estimate \\( \widehat{\bf x} \\):
1. Estimate weights using the above equation for \\( w_i \\) for each data item, to be used when calculating the next value
   for \\( \widehat{\bf x} \\).
   \\( r_i \\) and its derivative are evaluated at the current solution \\( \widehat{\bf x} \\).
1. Calculate the next estimate for \\( \widehat{\bf x} \\), using the updated weights \\( w_i \\) from the previous step.

IRLS normally requires a good initial estimate \\( \widehat{\bf x} \\) of \\( {\bf x} \\) to avoid local
minimal in the objective function.

## Supervised Gauss-Newton algorithm (Sup-GN)

Beginning with the objective function \\( F({\bf x}) \\), let us assume that we have an existing estimate \\( \widehat{\bf x}^{\*} \\)
of \\( {\bf x} \\). We can then try to improve this estimate by solving
\\[
  \frac{dF({\bf x})}{d{\bf x}} = {\bf 0}
\\]
We then build the first-order approximation to the weighting function \\( \psi(r) \\) that solves for an improved \\( \widehat{\bf x} \\):
\\[
  \frac{dF({\bf x})}{d{\bf x}} + \frac{d^2 F}{d{\bf x}^2} (\widehat{\bf x} - \widehat{\bf x}^{\*}) = {\bf 0}
\\]

where the derivatives are evaluated at \\( {\bf x}=\widehat{\bf x}^{\*} \\), or
\\[
  \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d{\bf x}} + \sum_{i=1}^n \left( \frac{d^2\rho(r_i)}{dr_i^2} \frac{d r_i}{d{\bf x}}^\intercal \frac{d r_i}{d{\bf x}} + \psi(r_i) \frac{d^2 r_i}{d{\bf x}^2} \right) (\widehat{\bf x} - \widehat{\bf x}^{\*}) = {\bf 0}
\\]
where \\( r_i \\) and the derivatives are again evaluated at \\( {\bf x}=\widehat{\bf x}^{\*} \\).
Noting the equation for \\( r_i \\) in the section above, we can write

\\[
  \frac{dr_i}{d{\bf x}} = \frac{dr_i}{d{\bf r}_i} \frac{d{\bf r}_i}{d{\bf x}} = \frac{1}{r_i} {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
\\]

and from this derive

\\[
  \frac{d^2 r_i}{d{\bf x}^2} = \frac{1}{r_i^3} \left( \frac{d{\bf r}_i}{d{\bf x}} \right)^{\intercal} \left( r_i^2 I - {\bf r}_i {\bf r}_i^\intercal \right) \frac{d{\bf r}_i}{d{\bf x}} + \frac{1}{r_i} {\bf r}_i^{\intercal} \frac{d^2{\bf r}_i}{d{\bf x}^2}
\\]

We assume that the data error \\( {\bf r}_{i} \\) is a smooth function of \\( {\bf x} \\) and so ignore the second
derivative term involving \\( d^2{\bf r}_i/d{\bf x}^2 \\).
Substituting the above (without the second term) into the equation for \\( \widehat{\bf x} - \widehat{\bf x}^{\*} \\) above
and combining with the equation for \\( \frac{dr_i}{d{\bf x}} \\) provides the result

\\[
  \sum_{i=1}^n {\bf a} + (A + B) (\widehat{\bf x} - \widehat{\bf x}^{\*}) = {\bf 0}
\\]

where

\\[
  {\bf a} = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
\\]

\\[
  A = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) \left(\frac{d{\bf r}_i}{d{\bf x}}\right)^\intercal \frac{d{\bf r}_i}{d{\bf x}}
\\]

\\[
  B = \sum_{i=1}^n \frac{1}{r_i^3} \left(r_i\frac{d^2\rho}{dr_i^2} - \psi(r_i) \right) \left(\frac{d{\bf r}_i}{d{\bf x}}\right)^\intercal {\bf r}_i {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
\\]

For the Welsch influence function we obtain

\\[
  \frac{1}{r_i} \psi(r_i) = \frac{1}{2} \\,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
\\]

\\[
  \frac{1}{r_i^3} \left(r_i\frac{d^2\rho}{dr_i^2} - \psi(r_i) \right) = -\frac{1}{2\sigma^2} \\,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
\\]

We can solve the Gauss-Newton update equations to provide
updated parameters \\( \widehat{\bf x} \\) given residuals, derivatives and
hence matrices \\( A \\), \\( B \\) evaluated at the previous parameters \\( \widehat{\bf x}^{\*} \\).
However a direct Gauss-Newton iteration gives no guarantee of
convergence. We propose the following "damped" Gauss-Newton updates, in the manner of Levenberg-Marquardt [5] damping:
\\[
  \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d{\bf x}} + (A + \lambda B) (\widehat{\bf x} - \widehat{\bf x}^{\*}) = {\bf 0}
\\]

where \\( \lambda \\) in the range \\( [0,1] \\) is a damping factor. When \\( \lambda=1 \\) (no damping)
we have a pure Gauss-Newton update. When \\( \lambda=0 \\) (maximum damping), we apply an update that is
exactly equivalent to IRLS for linear data models (proof omitted).
As a result we can treat the extreme value \\( \lambda=0 \\) as a "safe" iteration
that will guarantee, at least for linear models, a convergent update.
The Sup-GN algorithm then proceeeds as follows:

First initialize \\( \widehat{\bf x}^{\*} \\) in the same way as IRLS (least-squares solution with weights \\( w_i \\) set to one), and
set \\( \lambda=1 \\). Then given a damping adjustment factor \\( k<1 \\):
1. Solve the damped Gauss-Newton update equation above to produce an updated estimate \\( \widehat{\bf x} \\).
1. Check the objective function \\( F() \\) evaluated at \\( \widehat{\bf x}^{\*} \\) and \\( \widehat{\bf x} \\).
If we managed to improve the objective function, we can reduce the damping, otherwise we need to reject the new
estimate and increase the damping:
   - If \\( F(\widehat{\bf x}) < F(\widehat{\bf x}^{\*}) \\), set \\( \lambda \leftarrow k\lambda \\) and \\( \widehat{\bf x}^{\*} \leftarrow \widehat{\bf x} \\).
   - Else set \\( \lambda \leftarrow \min(1,\frac{\lambda}{k}) \\).
1. Iterate to convergence.

The advantage of this algorithm over IRLS is that it provides much faster convergence when we are near the solution.
It is well known that Gauss-Newton iterations can provide quadratic convergence [6], and we are taking advantage of this, whilst
still maintaining the option of pure IRLS iterations to guarantee convergence.
  
## References

[1] Charbonnier et al. in "Deterministic edge-preserving regularization in computed imaging", PAMI 6(2), 1997.

[2] Holland & Welsch "Robust regression using iteratively reweighted least-squares", Communications in Statistics-theory and Methods, 6(9), 1977.

[3] Peng et al. "On the Convergence of IRLS and Its Variants in Outlier-Robust Estimation", CVPR 2023.

[4] Blake & Zisserman "Visual reconstruction", MIT Press, 1987.

[5] K. Levenberg "A method for the solution of certain non – linear problems in least squares", Quarterly of Applied Mathematics, 2, 1944.

[6] A. Björck, "Numerical Methods for Least Squares Problems", (1996)

## GNC Smoothie software

The Python library is based on `numpy` and contains the following components:

- `IRLS.py` Top-level IRLS class. Build it minimally from a parameter class instance `param_instance`, a model class instance
`model_instance` and a `data` array. You need to supply `model_instance` and the `data` array yourself; the `param_instance` can be build from a class provided by GNC Smoothie. `model_instance` is an instance of a class you need to design, that implements a couple
of methods. The precise methods you need to implement are documented in `IRLS.py`.
- `SupGaussNewton.py` Top-level Supervised Gauss-Newton (Sup-GN) class. The parameters are very similar to the `IRLS` class
  and are documented in `SupGaussNewton.py`

The simplest non-trivial example of an IRLS/Sup-GN model class
is to support fitting a straight line through 2D data, with the model \\( y=ax+b \\), where \\( a \\) is the gradient of the line and
\\( b \\) is the intercept. The model class for line fitting might look like this:
```
import numpy as np

# Line model is y = a*x + b
class LineFit:
    def __init__(self):
        pass

    def cache_model(self, model, model_ref=None):
        self.__a = model[0]
        self.__b = model[1]

    # r = a*xi + b - yi
    def residual(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array([self.__a*x + self.__b - y])

    # dr/d(a b) = (x 1)
    def residual_gradient(self, data_item) -> np.array:
        x = data_item[0]
        return np.array([[x, 1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 2 # a,b
```
In this case the data items will have two values each, for \\( x,y \\). So an example data array could be
```
data = np.array([[0.0, 0.90], [0.1, 0.95], [0.2, 1.0], [0.3, 1.05], [0.4, 1.1]])
```
Then the code to build and run IRLS could look like ths.
```
from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.null_params import NullParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

sigma = 0.2
param_instance = NullParams(WelschInfluenceFunc(sigma))
model_instance = LineFit()
optimiser_instance = SupGaussNewton(param_instance, model_instance, data)
model = optimiser_instance.run()
print("line a b:",model)
```
The correct line parameters should be printed:
```
line a b: [0.5 0.9]
```
To use Supervised Gauss-Newton instead simply substitute SupGaussNewton for IRLS in the above code.
