# GNC Smoothie package

Python library supporting M-estimation using two algorithms: the well-known Iteratively Reweighted Least Squares (IRLS)
and our custom Supervised Gauss-Newton algorithm. First some introductory theory.

## M-estimation
M-estimation is a generalisation of maximum-likelihood estimation. 
We assume a known population probability density function (PDF) \\(f(.)\\), parametrised by a vector of parameters \\(\mathbf{x}\\),
and a set of independent and identically distributed data \\(\mathbf{z}_i\\), \\(i=1,...,n\\) sampled from the population.
The general model for the observations is
$$ \mathbf{z}_i = \mathbf{h}_i(\mathbf{x}) + \text{noise}
$$
for some observation model function \\(\mathbf{h}_i(\mathbf{x})\\).
The distribution of the noise determined by the population PDF, which is defined as some function
$$ f(\mathbf{z}_i - \mathbf{h}_i(\mathbf{x})) = f(\mathbf{r}_i)
$$
defining the \\(i\\)'th data error or ``residual'' vector \\(\mathbf{r}_i\\) as
$$ \mathbf{r}_i(\mathbf{x}) = \mathbf{z}_i - \mathbf{h}_i(\mathbf{x})
$$
For instance for Normal distributed observation errors with standard deviation \\(\sigma\\) we would have
$$ f(\mathbf{r}_i) = \,\mathrm{e}^{-\frac{||\mathbf{r}_i||^2}{2\sigma^2}}
$$
The maximum likelihood estimator of \\(\mathbf{x}\\) can be computed as 
$$ \widehat{\mathbf{x}} = \argmax_{\mathbf{x}} \left( \prod_{i=1}^n f(\mathbf{r}_i(\mathbf{x})) \right)
$$
or equivalently,
$$ \widehat{\mathbf{x}} = \argmin_{\mathbf{x}} \left( \sum_{i=1}^n - \log f(\mathbf{r}_i(\mathbf{x})) \right)
$$
M-estimation generalises this method by substituting a different function into the above sum, so we instead compute
$$
\widehat{\mathbf{x}} = \argmin_{\mathbf{x}} \left( \sum_{i=1}^n \rho(||\mathbf{r}_i(\mathbf{x})||) \right)
$$
for some function \\(\rho(r_i)\\) where
$$ r_i = ||\mathbf{r}_i(\mathbf{x})||
$$
We write the objective function above as
$$ F(\mathbf{x}) = \sum_{i=1}^n \rho(|| \mathbf{r}_i(\mathbf{x})||) = \sum_{i=1}^n \rho(r_i(\mathbf{x}))
$$
In the special case of Normally distributed observation errors, giving rise to standard least-squares,
\\(\rho(r) \sim r^2\\), the squared error in the observations.
The development and popularisation of M-estimation was driven by the need to fit models to data with outliers, i.e.
data not sampled from the population pdf but from a distinct distribution or distributions.
When outliers are present the least-squares method breaks down because single outliers
can have a huge influence, leading to a wildly incorrect value for \\(\widehat{\mathbf{x}}\\).
To allow for outliers \\(\rho(r)\\) is shaped by reducing the value of \\(\rho(r)\\) for large \\(r\\) error values.
The choice of influence function \\(\psi(r) = d\rho(r)/dr\\) is driven by a trade-off between the desire to
provide a good accuracy in the resulting
estimate for \\(\widehat{\mathbf{x}}\\) while providing robustness to noise in the data.
For instance, instead of the quadratic function required for least-squares, the pseudo-Huber influence
function~[1] is asymptotically linear in order to provide some level of robustness. 

## The Welsch influence function

Redescending influence functions have the property that their gradient tends to zero at either ends of the range.
This allows them to be robust to outliers with large errors. On the negative side, redescending influence functions
have the problem that the objective function above minimised by M-estimation may have multiple local minima.
It is difficult to ensure that the global minimum is reached. When the standard method ``iteratively reweighted least-squares''
(IRLS) is used, the result will depend on the quality of the initial value of \\(\mathbf{x}\\)
used for the iteration.
    
We start with the Welsch influence function [2]. This uses a negative Gaussian:
$$ \rho(r) = \frac{\sigma^2}{2} \left( 1 - \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \right)
$$
$$ \psi(r) = \frac{d\rho(r)}{dr} = \frac{r}{2} \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
$$
where the width \\(\sigma\\) of the Gaussian is known as the ``wavelength'' of the Welsch influence function.
Using a Gaussian influence function, whose gradient tends to zero for large errors, ensures robustness to large errors,
because their influence on the solution of equation~\ref{M_estimator} will be very small.
However in general it is presumed in the literature that solving M-estimation using redescending influence functions
requires a good initial estimate of the solution and comes with no guarantees of convergence. Recent work [3] in IRLS using
Graduated-Non-Convextity [4] has clarified that this is not always the case, and for many practical problems we can
achieve the *global* optimum solution without any initial model estimate being provided.

## Iteratively reweighted least squares (IRLS)

IRLS is the standard technique used to solve the non-linear optimisation problems that arise in M-estimation using robust
influence functions. IRLS assumes that the non-robust least-squares solution
for \\(\mathbf{x}\\) is soluble in closed form, given some ``weights'' assigned
to the data points. In other words there is a (simple) algorithm that can be used to solve the optimisation problem
$$ \widehat{\mathbf{x}} = \argmin_{\mathbf{x}} \left( \sum_{i=1}^n w_i ||\mathbf{z}_i - \mathbf{h}_i(\mathbf{x})||^2)\right)
   = \argmin_{\mathbf{x}} \left( F_{\text{LS}}(\mathbf{x}) \right)
$$
where
$$ F_{\text{LS}}(\mathbf{x}) = \sum_{i=1}^n w_i r_i(\mathbf{x})^2 = \sum_{i=1}^n w_i ||\mathbf{z}_i - \mathbf{h}_i(\mathbf{x})||^2
$$
for weights \\(w_i\\).
IRLS is based on the observation that the solution \\(\widehat{\mathbf{x}}\\) must be a stationary point of the
objective function \\(F_{\text{LS}}\\) in the above equation, so we must have
$$ \frac{dF_{\text{LS}}}{d\mathbf{x}} = \sum_{i=1}^n w_i r_i \frac{dr_i}{d\mathbf{x}} = \mathbf{0}
$$
The stationary point condition for solving the original optimisation problem for \\(\widehat{\mathbf{x}}\\) is the similar equation
$$ \frac{dF}{d\mathbf{x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{d\mathbf{x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{dr_i}\frac{dr_i}{d\mathbf{x}} = \sum_{i=1}^n \psi(r_i)\frac{dr_i}{d\mathbf{x}} = \mathbf{0}
$$
Comparing the equations involving \\(\frac{dF_{\text{LS}}}{d\mathbf{x}}\\) and \\(\frac{dF}{d\mathbf{x}}\\) above,
we see that the appropriate weight to use is
$$  w_i = \frac{1}{r_i} \psi(r_i)
$$
This choice will ensure that solving for \\(\frac{dF_{\text{LS}}}{d\mathbf{x}}\\) will also solve for \\(\frac{dF}{d\mathbf{x}}\\)
to first order, and hopefully improve the solution.
IRLS repeats the following two steps to convergence, given an initial state estimate \\(\widehat{\mathbf{x}}\\):
1. Estimate weights using the above equation for \\(w_i\\) for each data item, to be used when calculating the next value
   for \\(\widehat{\mathbf{x}}\\).
   \\(r_i\\) and its derivative are evaluated at the current solution \\(\widehat{\mathbf{x}}\\).
1. Calculate the next estimate for \\(\widehat{\mathbf{x}}\\), using the updated weights \\(w_i\\) from the previous step.

IRLS normally requires a good initial estimate \\(\widehat{\mathbf{x}}\\) of \\(\mathbf{x}\\) to avoid local
minimal in the objective function.

## Supervised Gauss-Newton algorithm (Sup-GN)

Beginning with the objective function~\ref{M_estimator_F}, let us assume that we have an existing estimate~\\(\widehat{\mathbf{x}}^*\\)
of~\\(\mathbf{x}\\). We can then try to improve this estimate by solving
$$ \frac{dF(\mathbf{x})}{d\mathbf{x}} = \mathbf{0}
$$
We then build the first-order approximation to the weighting function \\(\psi(r)\\) that solves for an improved \\(\widehat{\mathbf{x}}\\):
$$ \left. \frac{dF(\mathbf{x})}{d\mathbf{x}} \right|_{\mathbf{x}=\widehat{\mathbf{x}}^*} + \left. \frac{d^2 F}{d\mathbf{x}^2}\right|_{\mathbf{x}=\widehat{\mathbf{x}}^*} (\widehat{\mathbf{x}} - \widehat{\mathbf{x}}^*) = \mathbf{0}
$$
or
$$ \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d\mathbf{x}} + \sum_{i=1}^n \left( \frac{d^2\rho(r_i)}{dr_i^2} \frac{d r_i}{d\mathbf{x}}^\intercal \frac{d r_i}{d\mathbf{x}} + \psi(r_i) \frac{d^2 r_i}{d\mathbf{x}^2} \right) (\widehat{\mathbf{x}} - \widehat{\mathbf{x}}^*) = \mathbf{0}
$$
where \\(r_i\\) and the derivatives are evaluated at \\(\mathbf{x}=\widehat{\mathbf{x}}^*\\).
Noting the equation for \\(r_i\\) in the section above, we can write
$$ \frac{dr_i}{d\mathbf{x}} = \frac{dr_i}{d\mathbf{r}_i} \frac{d\mathbf{r}_i}{d\mathbf{x}} = \frac{1}{r_i} \mathbf{r}_i^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
and from this derive
$$ \frac{d^2r_i}{d\mathbf{x}^2} = \frac{1}{r_i^3} \left(\frac{d\mathbf{r}_i}{d\mathbf{x}}\right)^\intercal \left(r_i^2 I - \mathbf{r}_i \mathbf{r}_i^\intercal \right) \frac{d\mathbf{r}_i}{d\mathbf{x}} + \frac{1}{r_i} \mathbf{r}_i^\intercal \frac{d^2\mathbf{r}_i}{d\mathbf{x}^2}
$$
We assume that the data error \\(\mathbf{r}_i\\) is a smooth function of \\(\mathbf{x}\\) and so ignore the second
derivative term involving \\(d^2\mathbf{r}_i/d\mathbf{x}^2\\).
Substituting the above (without the second term) into the equation for \\(\widehat{\mathbf{x}} - \widehat{\mathbf{x}}^*\\) above
and combining with the equation for \\(\frac{dr_i}{d\mathbf{x}}\\) provides the result
$$ \sum_{i=1}^n \mathbf{a} + (A + B) (\widehat{\mathbf{x}} - \widehat{\mathbf{x}}^*) = \mathbf{0}
$$
where
$$ \mathbf{a} = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) \mathbf{r}_i^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
$$ A = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) \left(\frac{d\mathbf{r}_i}{d\mathbf{x}}\right)^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
$$ B = \sum_{i=1}^n \frac{1}{r_i^3} \left(r_i\frac{d^2\rho}{dr_i^2} - \psi(r_i) \right) \left(\frac{d\mathbf{r}_i}{d\mathbf{x}}\right)^\intercal \mathbf{r}_i \mathbf{r}_i^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
For the Welsch influence function we obtain
$$ \mathbf{a}_{\textrm{Welsch}} = \sum_{i=1}^n \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \mathbf{r}_i^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
$$ A_{\textrm{Welsch}} = \sum_{i=1}^n \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \left(\frac{d\mathbf{r}_i}{d\mathbf{x}}\right)^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
$$ B_{\textrm{Welsch}} = \sum_{i=1}^n -\frac{1}{\sigma^2} \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \left(\frac{d\mathbf{r}_i}{d\mathbf{x}}\right)^\intercal \mathbf{r}_i \mathbf{r}_i^\intercal \frac{d\mathbf{r}_i}{d\mathbf{x}}
$$
We can solve the Gauss-Newton update equations~\ref{NewtonUpdate2} to provide
updated parameters~\\(\widehat{\mathbf{x}}\\) given residuals, derivatives and
hence matrices \\(A\\), \\(B\\) evaluated at the previous parameters~\\(\widehat{\mathbf{x}}^*\\).
However a direct Gauss-Newton iteration gives no guarantee of
convergence. We propose the following ``damped'' Gauss-Newton updates, in the manner of Levenberg-Marquardt [5] damping:
$$ \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d\mathbf{x}} + (A + \lambda B) (\widehat{\mathbf{x}} - \widehat{\mathbf{x}}^*) = \mathbf{0}
$$
where \\(\lambda\\) in the range \\([0,1]\\) is a damping factor. When \\(\lambda=1\\) (no damping)
we have a pure Gauss-Newton update. When \\(\lambda=0\\) (maximum damping), we apply an update that is
exactly equivalent to IRLS for linear data models (proof omitted).
As a result we can treat the extreme value \\(\lambda=0\\) as a ``safe'' iteration
that will guarantee, at least for linear models, a convergent update.
The Sup-GN algorithm then proceeeds as follows:

First initialize \\(\widehat{\mathbf{x}}^*\\) in the same way as IRLS (least-squares solution with weights \\(w_i\\) set to one), and
set \\(\lambda=1\\). Then given a damping adjustment factor \\(k<1\\):
1. Solve the damped Gauss-Newton update equation above to produce an updated estimate \\(\widehat{\mathbf{x}}\\).
1. Check the objective function \\(F()\\) evaluated at \\(\widehat{\mathbf{x}}^*\\) and \\(\widehat{\mathbf{x}}\\).
If we managed to improve the objective function, we can reduce the damping, otherwise we need to reject the new
estimate and increase the damping:
   - If \\(F(\widehat{\mathbf{x}}) < F(\widehat{\mathbf{x}}^*)\\), set \\(\lambda \leftarrow k\lambda\\) and \\(\widehat{\mathbf{x}}^* \leftarrow \widehat{\mathbf{x}}\\).
   - Else set \\(\lambda \leftarrow \min(1,\frac{\lambda}{k})\\).
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
is to support fitting a straight line through 2D data, with the model \\(y=ax+b\\), where \\(a\\) is the gradient of the line and
\\(b\\) is the intercept. The model class for line fitting might look like this:
```
import math
import numpy as np

# Line model is y = a*x + b
class LineFit:
    def __init__(self):
        pass

    # r = a*xi + b - yi
    def residual(self, model, data_item, model_ref=None) -> np.array:
        a = model[0]
        b = model[1]
        x = data_item[0]
        y = data_item[1]
        return np.array([a*x + b - y])

    # dr/d(a b) = (x 1)
    def residual_gradient(self, model, data_item, model_ref=None) -> np.array:
        x = data_item[0]
        return np.array([[x, 1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 2 # a,b
```
In this case the data items will have two values each, for \\(x,y\\). So an example data array could be
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
