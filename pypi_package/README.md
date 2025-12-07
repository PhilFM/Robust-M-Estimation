# GNC Smoothie package

Python library supporting M-estimation using two algorithms: the well-known Iteratively Reweighted Least Squares (IRLS)
and our custom Supervised Gauss-Newton algorithm. Author: Philip McLauchlan `philipmclauchlan6@gmail.com`.

First some introductory theory.

## M-estimation
M-estimation is a generalisation of maximum-likelihood estimation. 
We assume a known population probability density function (PDF) $ f(.) $, parametrised by a vector of parameters $ {\bf x} $,
and a set of independent and identically distributed data $ {\bf z}_i $, $ i=1,...,n $ sampled from the population.
The general model for the observations is
$$
  {\bf z}_i = {\bf h}_i({\bf x}) + \text{noise}
$$
for some observation model function $ {\bf h}_i({\bf x}) $.
The distribution of the noise determined by the population PDF, which is defined as some function
$$
  f({\bf z}_i - {\bf h}_i({\bf x})) = f({\bf r}_i)
$$
defining the $ i $'th data error or "residual" vector $ {\bf r}_i $ as
$$
  {\bf r}_i({\bf x}) = {\bf z}_i - {\bf h}_i({\bf x})
$$

For instance for Normal distributed observation errors with standard deviation $ \sigma $ we would have
$$
  f({\bf r}_i) = \,\mathrm{e}^{-\frac{|| {\bf r}_i ||^2}{2\sigma^2}}
$$

The maximum likelihood estimator of $ {\bf x} $ can be computed as 

$$
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\,\text{max}} \left( \prod_{i=1}^n f({\bf r}_i({\bf x})) \right)
$$

or equivalently,
$$
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\,\text{min}} \left( \sum_{i=1}^n - \log f({\bf r}_i({\bf x})) \right)
$$

M-estimation generalises this method by substituting a different function into the above sum, so we instead compute
$$
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\,\text{min}} \left( \sum_{i=1}^n \rho(|| {\bf r}_i({\bf x}) ||) \right)
$$
for some function $ \rho(r_i) $ where
$$
  r_i = || {\bf r}_i({\bf x}) ||
$$

We write the objective function above as

$$
  F({\bf x}) = \sum_{i=1}^n \rho(|| {\bf r}_i({\bf x}) ||)
$$

or
$$
  F({\bf x}) = \sum_{i=1}^n \rho(r_i({\bf x}))
$$

In the special case of normally distributed observation errors, this give rise to standard least-squares,
$ \rho(r) \sim r^2 $, the squared error in the observations.
The development and popularisation of M-estimation was driven by the need to fit models to data with outliers, i.e.
data not sampled from the population pdf but from a distinct distribution or distributions.
When outliers are present the least-squares method breaks down because single outliers
can have a huge influence, leading to a wildly incorrect value for $ \widehat{\bf x} $.
To allow for outliers $ \rho(r) $ is shaped by reducing the value of $ \rho(r) $ for large $ r $ error values.
The choice of influence function $ \psi(r) = d\rho(r)/dr $ is driven by a trade-off between the desire to
provide a good accuracy in the resulting
estimate for $ \widehat{\bf x} $ while providing robustness to noise in the data.
For instance, instead of the quadratic function required for least-squares, the pseudo-Huber influence
function [1] is asymptotically linear in order to provide some level of robustness. 

## The Welsch influence function

Redescending influence functions have the property that their gradient tends to zero at either ends of the range.
This allows them to be robust to outliers with large errors. On the negative side, redescending influence functions
have the problem that the objective function above minimised by M-estimation may have multiple local minima.
It is difficult to ensure that the global minimum is reached. When the standard method "iteratively reweighted least-squares"
(IRLS) is used, the result will depend on the quality of the initial value of $ {\bf x} $
used for the iteration.
    
We start with the Welsch influence function [2]. This uses a negative Gaussian:
$$
  \rho(r) = \frac{\sigma^2}{2} \left( 1 - \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \right)
$$
$$
  \psi(r) = \frac{d\rho(r)}{dr} = \frac{r}{2} \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
$$
where the width $ \sigma $ of the Gaussian is known as the "wavelength" of the Welsch influence function.
Using a Gaussian influence function, whose gradient tends to zero for large errors, ensures robustness to large errors,
because their influence on the solution will be very small.
However in general it is presumed in the literature that solving M-estimation using redescending influence functions
requires a good initial estimate of the solution and comes with no guarantees of convergence. Recent work [3] in IRLS using
Graduated-Non-Convextity [4] has clarified that this is not always the case, and for many practical problems we can
achieve the *global* optimum solution without any initial model estimate being provided.

## Iteratively reweighted least squares (IRLS)

IRLS is the standard technique used to solve the non-linear optimisation problems that arise in M-estimation using robust
influence functions. IRLS assumes that the non-robust least-squares solution
for $ {\bf x} $ is soluble in closed form, given some "weights" assigned
to the data points. In other words there is a (simple) algorithm that can be used to solve the optimisation problem

$$
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\,\text{min}} \left( \sum_{i=1}^n w_i || {\bf z}_i - {\bf h}_i({\bf x}) ||^2)\right)
$$

or
$$
  \widehat{\bf x} = \underset{{\bf x}}{\text{arg}\,\text{min}} \left( F_{\text{LS}}({\bf x}) \right)
$$

where
$$
  F_{\text{LS}}({\bf x}) = \sum_{i=1}^n w_i r_i({\bf x})^2 = \sum_{i=1}^n w_i ||{\bf z}_i - {\bf h}_i({\bf x})||^2
$$

for weights $ w_i $.
IRLS is based on the observation that the solution $ \widehat{\bf x} $ must be a stationary point of the
objective function $ F_{\text{LS}} $ in the above equation, so we must have

$$
  \frac{dF_{\text{LS}}}{d{\bf x}} = \sum_{i=1}^n w_i r_i \frac{dr_i}{d{\bf x}} = {\bf 0}
$$

The stationary point condition for solving the original optimisation problem for $ \widehat{\bf x} $ is the similar equation
$$
  \frac{dF}{d{\bf x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{d{\bf x}} = \sum_{i=1}^n \frac{d\rho(r_i)}{dr_i}\frac{dr_i}{d{\bf x}} = \sum_{i=1}^n \psi(r_i)\frac{dr_i}{d{\bf x}} = {\bf 0}
$$

Comparing the equations involving $ \frac{dF_{\text{LS}}}{d{\bf x}} $ and $ \frac{dF}{d{\bf x}} $ above,
we see that the appropriate weight to use is
$$
  w_i = \frac{1}{r_i} \psi(r_i)
$$

This choice will ensure that solving for $ \frac{dF_{\text{LS}}}{d{\bf x}} $ will also solve for $ \frac{dF}{d{\bf x}} $
to first order, and hopefully improve the solution.
IRLS repeats the following two steps to convergence, given an initial state estimate $ \widehat{\bf x} $:
1. Estimate weights using the above equation for $ w_i $ for each data item, to be used when calculating the next value
   for $ \widehat{\bf x} $.
   $ r_i $ and its derivative are evaluated at the current solution $ \widehat{\bf x} $.
1. Calculate the next estimate for $ \widehat{\bf x} $, using the updated weights $ w_i $ from the previous step.

IRLS normally requires a good initial estimate $ \widehat{\bf x} $ of $ {\bf x} $ to avoid local
minimal in the objective function.

## Supervised Gauss-Newton algorithm (Sup-GN)

Beginning with the objective function $ F({\bf x}) $, let us assume that we have an existing estimate $ \widehat{\bf x}^{*} $
of $ {\bf x} $. We can then try to improve this estimate by solving
$$
  \frac{dF({\bf x})}{d{\bf x}} = {\bf 0}
$$
We then build the first-order approximation to the weighting function $ \psi(r) $ that solves for an improved $ \widehat{\bf x} $:
$$
  \frac{dF({\bf x})}{d{\bf x}} + \frac{d^2 F}{d{\bf x}^2} (\widehat{\bf x} - \widehat{\bf x}^{*}) = {\bf 0}
$$

where the derivatives are evaluated at $ {\bf x}=\widehat{\bf x}^{*} $, or
$$
  \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d{\bf x}} + \sum_{i=1}^n \left( \frac{d^2\rho(r_i)}{dr_i^2} \frac{d r_i}{d{\bf x}}^\intercal \frac{d r_i}{d{\bf x}} + \psi(r_i) \frac{d^2 r_i}{d{\bf x}^2} \right) (\widehat{\bf x} - \widehat{\bf x}^{*}) = {\bf 0}
$$
where $ r_i $ and the derivatives are again evaluated at $ {\bf x}=\widehat{\bf x}^{*} $.
Noting the equation for $ r_i $ in the section above, we can write

$$
  \frac{dr_i}{d{\bf x}} = \frac{dr_i}{d{\bf r}_i} \frac{d{\bf r}_i}{d{\bf x}} = \frac{1}{r_i} {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
$$

and from this derive

$$
  \frac{d^2 r_i}{d{\bf x}^2} = \frac{1}{r_i^3} \left( \frac{d{\bf r}_i}{d{\bf x}} \right)^{\intercal} \left( r_i^2 I - {\bf r}_i {\bf r}_i^\intercal \right) \frac{d{\bf r}_i}{d{\bf x}} + \frac{1}{r_i} {\bf r}_i^{\intercal} \frac{d^2{\bf r}_i}{d{\bf x}^2}
$$

We assume that the data error $ {\bf r}_{i} $ is a smooth function of $ {\bf x} $ and so ignore the second
derivative term involving $ d^2{\bf r}_i/d{\bf x}^2 $.
Substituting the above (without the second term) into the equation for $ \widehat{\bf x} - \widehat{\bf x}^{*} $ above
and combining with the equation for $ \frac{dr_i}{d{\bf x}} $ provides the result

$$
  \sum_{i=1}^n {\bf a} + (A + B) (\widehat{\bf x} - \widehat{\bf x}^{*}) = {\bf 0}
$$

where

$$
  {\bf a} = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
$$

$$
  A = \sum_{i=1}^n \frac{1}{r_i} \psi(r_i) \left(\frac{d{\bf r}_i}{d{\bf x}}\right)^\intercal \frac{d{\bf r}_i}{d{\bf x}}
$$

$$
  B = \sum_{i=1}^n \frac{1}{r_i^3} \left(r_i\frac{d^2\rho}{dr_i^2} - \psi(r_i) \right) \left(\frac{d{\bf r}_i}{d{\bf x}}\right)^\intercal {\bf r}_i {\bf r}_i^\intercal \frac{d{\bf r}_i}{d{\bf x}}
$$

For the Welsch influence function we obtain

$$
  \frac{1}{r_i} \psi(r_i) = \frac{1}{2} \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
$$

$$
  \frac{1}{r_i^3} \left(r_i\frac{d^2\rho}{dr_i^2} - \psi(r_i) \right) = -\frac{1}{2\sigma^2} \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}}
$$

We can solve the Gauss-Newton update equations to provide
updated parameters $ \widehat{\bf x} $ given residuals, derivatives and
hence matrices $ A $, $ B $ evaluated at the previous parameters $ \widehat{\bf x}^{*} $.
However a direct Gauss-Newton iteration gives no guarantee of
convergence. We propose the following "damped" Gauss-Newton updates, in the manner of Levenberg-Marquardt [5] damping:
$$
  \sum_{i=1}^n \psi(r_i) \frac{dr_i}{d{\bf x}} + (A + \lambda B) (\widehat{\bf x} - \widehat{\bf x}^{*}) = {\bf 0}
$$

where $ \lambda $ in the range $ [0,1] $ is a damping factor. When $ \lambda=1 $ (no damping)
we have a pure Gauss-Newton update. When $ \lambda=0 $ (maximum damping), we apply an update that is
exactly equivalent to IRLS for linear data models (proof omitted).
As a result we can treat the extreme value $ \lambda=0 $ as a "safe" iteration
that will guarantee, at least for linear models, a convergent update.
The Sup-GN algorithm then proceeeds as follows:

First initialize $ \widehat{\bf x}^{*} $ in the same way as IRLS (least-squares solution with weights $ w_i $ set to one), and
set $ \lambda=1 $. Then given a damping adjustment factor $ k<1 $:
1. Solve the damped Gauss-Newton update equation above to produce an updated estimate $ \widehat{\bf x} $.
1. Check the objective function $ F() $ evaluated at $ \widehat{\bf x}^{*} $ and $ \widehat{\bf x} $.
If we managed to improve the objective function, we can reduce the damping, otherwise we need to reject the new
estimate and increase the damping:
   - If $ F(\widehat{\bf x}) < F(\widehat{\bf x}^{*}) $, set $ \lambda \leftarrow k\lambda $ and $ \widehat{\bf x}^{*} \leftarrow \widehat{\bf x} $.
   - Else set $ \lambda \leftarrow \min(1,\frac{\lambda}{k}) $.
1. Iterate to convergence.

The advantage of this algorithm over IRLS is that it provides much faster convergence when we are near the solution.
It is well known that Gauss-Newton iterations can provide quadratic convergence [6], and we are taking advantage of this, whilst
still maintaining the option of pure IRLS iterations to guarantee convergence.

## GNC Smoothie software

The Python library is based on `numpy` and contains the following top-level modules:

### IRLS class: `irls.py`

Top-level 'IRLS' class. Once you have constructed an instance of this class, call the `run()`
method to run it. This returns `True` on successful convergence, `False` on failure.
The final model and model reference (see below) are stored in `final_model` and `final_model_ref`,
whether the `run()` method succeeds or not.

Here are the parameters that need to be passed to the `IRLS` class constructor. Optional parameters follow.
- `param_instance` Defines the GNC schedule to be followed by IRLS. If GNC is not being used then
   this can be a `NullParams` instance imported from `null_params.py.
   Should have an internal influence_func_instance
   that specifies the IRLS influence function to be used. The influence_func_instance
   should provide the following method:
   - `summary(self) -> str` Returns a string containing the values of the internal parameters.
                    `param_instance` itself should provide the following methods:
   - `reset(self, init: bool = True)`
           Resets the internal influence_func_instance according to the stage of the
           GNC schedule indicated by the init parameter. If init is `True`, reset to the
           starting value to prepare for the GNC process to start. If init is `False`,
           reset to the final stage of GNC.
   - `at_final_state(self) -> bool`
           Returns `True` if the GNC schedule has reached the final stage, `False` otherwise
   - `update(self)` Update the influence_func_instance to the next step in the GNC schedule.
- `model_instance` The model being fitted to the data, an instance of a class you design
     that provides the following methods:
   - `cache_model(self, model, model_ref=None)`
     Use this method to cache the model, prior to the `residual()` method being called on each data item.
   - `residual(self, data_item, data_id:int=None) -> np.array`
     Calculates the residual (error) of the `data_item` given the model. If the model
     contains reference parameters e.g. for estimating rotation, these are passed
     as `model_ref`. If there are different types of data to be handled, the `data_id`
     is the id of this data item (see the data_ids array below)
   - `linear_model_size(self) -> int`
     Returns the number of parameters in the model if the model is linear.
     The `BaseIRLS` class uses an internal weighted_fit() method to fit a linear model
     to the data with specified weights, so that the programmer does not have to
     implement it. If the model is non-linear, don't define this method.
   - `weighted_fit(self, data, data_ids, weight, scale) -> (np.array, np.array)`
     If `linear_model_size()` is not provided, the model is not linear. If a closed-form
     solution for the best model given the data with weights nevertheless exists,
     implement it in yout class. The `scale` array indicates that certain data items are less accurate and so have a
     scale value > 1, indicating that the influence function for that data item
     should be stretched by the given scale factor.
     For non-linear problems with no closed-form solution, pass a suitable starting
     point as the model_start (and optionally model_ref_start) parameters, see below.
     In that case the `weighted_fit()` method is not used.
- `data` An array of data items. Each data item should itself be an array.

Now the optional parameters for the `IRLS` class constructor:
- `data_ids` An array of index values, indicating different types of data. The id is to be handled
     inside your `residual()` methods. You define its meaning.
     If this array is not provided, all data is assumed to be the same type and zero is passed
     as the id for all data items.
- `weight` An optional array of float weight values for each data item.
     If not provided, weights are initialised to one.
- `scale` An optional array of scale values, indicating that one or more data items are known to
     have reduced accuracy, i.e. a wider influence function. The scale indicates the stretching
     to apply to the influence function for that data item.
-  `numeric_derivs_influence: bool` Whether to calculate derivatives of the influence function numerically
     from a provided `rho()` method or directly using a provided `rhop()` method.
-  `max_niterations: int` Maximum number of IRLS iterations to apply before aborting.
-  `diff_thres: float` Terminate when successful update changes the model model parameters by less than this value.
-  `print_warnings: bool` Whether to print debugging information.
-  `model_start` Optional starting value for model model parameters
-  `model_ref_start` Optional starting reference parameters for model, e.g. if optimising rotation
-  `debug: bool` Whether to return extra debugging data on exit:
   - The number of iterations actually applied
   - The norm of the model parameters change at each iteration, as a list of difference values
   - A list of the model parameters at each iteration

### Supervised Gauss-Newton class: `sup_gauss_newton.py`

Top-level `SupGaussNewton` class, an implementation of Supervised Gauss-Newton (Sup-GN).
Sup-GN is an alternative to IRLS most suitable for the two cases:
- Linear model where the data relates to the model via a linear function.
- Non-linear model where there is no closed-form solution to calculating the
      model parameters from the weighted data.

Use the basic IRLS in the remaining case where a non-trivial closed-form solution for the model
is available, such as 3D point cloud registration (SVD solution). IRLS is not suitable for
non-linear problems where a closed-form solution is not available, but Sup-GN can be used in
such problems so long as a reasonable starting point for the model can be supplied (see the
`model_start` and `model_ref_start` parameters below). For linear models Sup-GN provides a simpler
model implementation than IRLS, since the closed-form solution for the model is calculated
internally. Also Sup-GN converges quadratically for linear models when close to the solution.

Once you have constructed an instance of the `SupGaussNewton` class, call the `run()`
method to run it. This returns `True` on successful convergence, `False` on failure.
The final model and model reference (see below) are stored in `final_model` and `final_model_ref`,
whether the `run()` method succeeds or not.

The parameters to the `SupGaussNewton` constructor are very similar to the `IRLS` class,
but there are some twists due to Sup-GN requiring differentiation of the model residual.
Here are the parameters you need to pass to the `SupGaussNewton` class:
- `param_instance` Defines the GNC schedule to be followed by IRLS. If GNC is not being used then
    this can be a `NullParams` instance imported from `null_params.py.
    Should have an internal `influence_func_instance`
    that specifies the IRLS influence function to be used. This `influence_func_instance`
    should provide these methods:
   - `objective_func_sign(self) -> float`
        Returns either one or minus one depending on whether the objective function
        increases for large residuals (one) or decreases (minus one). Typical IRLS
        objective functions such as Huber and Geman-McClure increase for large residuals,
        so most functions will return one. The version of Welsch we have implemented in
        `gnc_welsch_params.py` uses a nagative sense, which slightly simplifies the
        implementation, because otherwise we would have to add one to the objective
        function in order to keep it positive.
   - `rho(self, rsqr: float, s: float) -> float`
        The objective function given
      - `rsqr` The square of the L2 norm of the residual vector
      - `s` The scale of the data item indicating its known inaccuracy, so a value >= 1.
     Returns the value of the objective function.
   - `rhop(self, rsqr: float, s: float) -> float`
        The influence function, which is equal to the derivative with respect to $ r $
        of `rho(rsqr,s)` divided by $ r $, where $ r $ is the L2 norm of the residual vector.
        If `numeric_derivs_influence` is set to `True` (see below) then the derivatives
        are calculated numerically from `rho()` and `rhop()` is not required.
   - `Bterm(self, rsqr: float, s: float) -> float`
        Implements $ (r*\rho''(r) - \rho'(r))/(r^3) $ where ' indicates derivative.
        If `numeric_derivs_influence` is set to `True` (see below) then the derivatives
        are calculated numerically from rho() and Bterm() is not required.
   - `summary(self) -> str`
        A string containing the values of the internal parameters.

   param_instance itself shoule provide the following methods:
   - `reset(self, init: bool = True)`
     Resets the internal `influence_func_instance` according to the stage of the
     GNC schedule indicated by the `init` parameter. If `init` is `True`, reset to the
     starting value to prepare for the GNC process to start. If `init` is `False`,
     reset to the final stage of GNC.
   - `at_final_state(self) -> bool`
     Returns `True` if the GNC schedule has reached the final stage, `False` otherwise.
   - `update(self)` Update the `influence_func_instance` to the next step in the GNC schedule.

- `model_instance` The model being fitted to the data, an instance of a class you design
  that provides the following methods:
   - `cache_model(self, model, model_ref=None)`
     Use this method to cache the model, prior to `residual()` and `residual_gradient()`
     methods being called on each data item.
   - `residual(self, data_item, data_id:int=None) -> np.array`
     Calculates the residual (error) of the data_item given the model. If the model
     contains reference parameters e.g. for estimating rotation, these are passed
     as `model_ref`. If there are different types of data to be handled, the `data_id`
     is the id of this data item (see the `data_ids` array below)
   - `residual_gradient(self, data_item, data_id:int=None) -> np.array`
     The Jacobian or derivative matrix of the residual vector with respect
     to the model parameters. If the `numeric_derivs_model` parameter is set to `True`
     (see below) then the derivatives are calculated numerically using the `residual()`
     method. If there are different types of data to be handled, the `data_id`
     is the id of this data item (see the `data_ids` array below)
   - `linear_model_size(self) -> int`
     Returns the number of parameters in the model if the model is linear.
     The `BaseIRLS` class uses an internal `weighted_fit()` method to fit a linear model
     to the data with specified weights, so that the programmer does not have to
     implement it. If the model is non-linear, don't define this method.
   - `weighted_fit(self, data, data_ids, weight, scale) -> (np.array, np.array)`
     If `linear_model_size()` is not provided, the model is not linear. If a closed-form
     solution for the best model given the data with weights nevertheless exists,
     implement it in yout class. The `scale`
     array indicates that certain data items are less accurate and so have a
     scale value > 1, indicating that the influence function for that data item
     should be stretched by the given scale factor.
     For non-linear problems with no closed-form solution, pass a suitable starting
     point as the `model_start` (and optionally `model_ref_start`) parameters, see below.
     In that case the `weighted_fit()` method is not used.
- `data` An array of data items. Each data item should itself be an array.

Now the optional parameters for the `SupGaussNewton` class constructor:
- `data_ids` An array of index values, indicating different types of data. The id is to be handled
     inside your `residual()` and `residual_gradient()` methods. You define its meaning.
     If this array is not provided, all data is assumed to be the same type and zero is passed
     as the id for all data items.
- `weight` An optional array of float weight values for each data item.
     If not provided, weights are initialised to one
- `scale` An optional array of scale values, indicating that one or more data items are known to
     have reduced accuracy, i.e. a wider influence function. The scale indicates the stretching
     to apply to the influence function for that data item.
- `numeric_derivs_model: bool` Whether to calculate derivatives of the data residual vector with respect to the
     model parameters numerically using a provided `residual()` method or directly
     using a provided `residual_gradient()` method.
- `numeric_derivs_influence: bool` Whether to calculate derivatives of the influence function numerically
     from a provided `rho()` method or directly using a provided `rhop()` method.
- `max_niterations: int` Maximum number of Sup-GN iterations to apply before aborting
- `residual_tolerance: float` An optional parameter that is used to terminate Sup-GN when the improvement to the
     objective function value is smaller than the provided threshold
- `lambda_start: float` Starting value for the Sup-GN damping, similar to Levenberg-Marquart damping.
     In Sup-GN the level of damping is high when lambda is small, so normally it is
     best to start with an optimistic small value.
- `lambda_max: float` Maximum value for lambda in Sup-GN damping. This should be in the range [0,1].
- `lambda_scale: float` Scale factor to multiply lambda by when an iteration successfully reduces/increases
     the objective function (depending on the +/- sign specified by
     `param_instance.influence_func_instance.influence_func_sign()`, see above).
     When the iteration is not successful, the model change is reverted and $ \lambda $ is divided
     by this factor to increase the damping at the next iteration.
- `diff_thres: float` Terminate when successful update changes the model parameters by less than this value.
- `print_warnings: bool` Whether to print debugging information.
- `model_start` Optional starting value for model parameters
- `model_ref_start` Optional starting reference parameters for model, e.g. if optimising rotation
- `debug: bool` Whether to return extra debugging data on exit:
   - The number of iterations actually applied
   - The norm of the model parameters change at each iteration, as a list of difference values
   - A list of the model parameters at each iteration

### Example code for the `IRLS` and `SupGaussNewton` classes

The simplest non-trivial example of an IRLS/Sup-GN model class
is to support fitting a straight line through 2D data, with the model $ y=ax+b $, where $ a $ is the gradient of the line and
$ b $ is the intercept. The model class for line fitting might look like this:
```
import numpy as np

# Line model is y = a*x + b
class LineFit:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__a = model[0]
        self.__b = model[1]

    # r = a*xi + b - yi
    def residual(self, data_item, data_id:int=None) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array([self.__a*x + self.__b - y])

    # dr/d(a b) = (x 1)
    def residual_gradient(self, data_item, data_id:int=None) -> np.array:
        x = data_item[0]
        return np.array([[x, 1.0]])

    # return number of parameters in model if the model is linear,
    # otherwise don't define this method
    def linear_model_size(self) -> int:
        return 2 # a,b
```
In this case the data items will have two values each, for $ x,y $. So an example data array could be
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
if optimiser_instance.run():
    model = optimiser_instance.final_model
    print("line a b:",model)
```
The correct line parameters should be printed:
```
line a b: [0.5 0.9]
```
To use Supervised Gauss-Newton instead simply substitute `SupGaussNewton` for `IRLS` in the above code.

### Base class for IRLS and Sup-GN algorithms `base_irls.py`

Implements the many features in common between IRLS and Sup-GN. Should not be used directly in
your code.

### Checking derivatives: `check_derivs.py`

When you design a model class to be used for Sup-GN optimisation, and have written your `residual()`
method defining how to calculate the model/data errors, you have a design choice:
1. Implement the `residual_gradient()` method yourself, first working out the Jacobian matrix of
   the residual vector.
1. Calculating the derivatives numerically. This is handled internally by the `SupGaussNewton` class.
   You just need to pass `numeric_derivs_model=True` in the arguments to the `SupGaussNewton` constructor.

In the former case, you will want to check that your derivative calculation is correct.
Use the `check_derivs()` method to do this. It calculates derivatives both ways, first calling
your `residual_gradient()` method and then calculating them numerically, and compares
them using thresholds. If the error in any derivative is greater than the provided threshold
the method returns `False`. The required arguments to the method are:
- `optimiser_instance` An instance of `SupGaussNewton` built with an instance of your model class.
   The `SupGaussNewton` instance also needs a `param_instance` and some data to construct it - see
   below for an example of how to do this.
- `model` An example vector of model parameters.

Optional argmuments that may be required:
- `model_ref` Reference model parameters, for instance if the model contains rotation parameters.
- `diff_threshold_a: float` Threshold used for the terms of the $ {\bf a} $ vector in
  the Sup-GN iteration.
- `diff_threshold_AlB: float` Threshold used for the terms of the $ A $ and $ B $ matrices in
  the Sup-GN iteration.
- `print_diffs: bool` Whether to print the differences between the analytic and numerical
  derivative estimates.
- `print_derivs: bool` Whether to print all the derivatives being compared.

For example, to test the derivatives of the `LineFit` class above you could use this code:

```
from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.null_params import NullParams
from gnc_smoothie.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie.check_derivs import check_derivs

data = [[2.0, -1.0]] # single data point
derivs_good = check_derivs(SupGaussNewton(NullParams(QuadraticInfluenceFunc()), LineFit(), data), [1.0, 2.0], # model a,b
                           diff_threshold_AlB=1.e-4)
```
Note that we use the simplest parameter class `NullParams` and influence function class `QuadraticInfluenceFunc` to build
the `SupGaussNewton` instance, although the derivative testing will work for other classes.

### Welsch influence function `welsch_influence_func.py`

This provides the class `WelschInfluenceFunc`, that implements the Welsch influence function, defined as
$$
  \rho(r) = \frac{\sigma^2}{2} \left( 1 - \,\mathrm{e}^{-\frac{r^2}{2\sigma^2}} \right)
$$

with a single parameter $ sigma $. This is the simplest robust influence function.
We recommend using the Welsch influence function over others because of its simplicity and superior convergence properties
(details to come). The Welsch influence function is redescending, meaning that the gradient tends to zero at either ends of the range.
This provides it with remarkable robustness, even in the presence of very bad outliers.
Wrap it with a `GNC_WelschParams` class to provide it with a GNC schedule and outstanding convergence.

### GNC Welsch schedule class `gnc_welsch_params.py`

Build a parameter class instance by building a `GNC_WelschInfluenceFunc` instance and then
wrapping it into this `GNC_WelschParams` class to provide the GNC schedule. The `sigma` value
starts at a high value `sigma_limit` and descends geometrically through `num_sigma_steps`
values to a low value `sigma_base` that approximates the population error standard deviation.

### Pseudo-Huber influence function `pseudo_huber_influence_func.py`

The class `PseudoHuberInfluenceFunc` implements the fully differentiable
version [1] of the original Huber influence function [7].
$$
   \rho(r) = \sigma^2 \left( \sqrt{1 + (r/\sigma)^2} - 1 \right)
$$

with a single parameter $ \sigma $. You can use this influence function if you know that your outliers
are relatively small. The Pseudo-Huber objective function $ \rho(r) $ is convex, so you don't
need to use a GNC schedule. Instead use it with `NullParams`. On the other hand, if the outliers are
very bad then it will fail to converge to the correct solution.

### Geman-McClure influence function `geman_mcclure_influence_func.py`

This `GemanMcClureInfluenceFunc` class implements the Geman-McClure influence function [8], defined by
$$
   \rho(r) = \frac{r^2}{\sigma^2 + r^2}
$$

with a single parameter $ \sigma $. Geman-McClure is another redescending influence function,
so it is suitable for being embedded in a GNC schedule and handling large outliers. Wrap it in the
same `GNC_WelschParams` GNC schedule class used for the Welsch influence function if you want
to try this, but we still recommend using the Welsch influence function over Geman-McClure.

### GNC IRLS-p influence function `gnc_irls_p_influence_func.py`

This `GNC_IRLSpInfluenceFunc` class implements the GNC schedule recommended by Peng et al. in their
excellent paper [3]. Peng et al. prove that for many IRLS problems, GNC IRLS-p provides guaranteed
fast convergence. The influence function is more complex than the above alternatives - see the paper
and the code for details. Combining this class with the GNC schedule class `GNC_IRLSpParams` (see below)
provides an alternative to our recommended `GNC_WelschParams` + `GNC_WelschInfluenceFunc` combination.

### GNC IRLS-p schedule class `gnc_irls_p_params.py`

Build a parameter class instance by building a `GNC_IRLSpInfluenceFunc` instance and then
wrapping it into this `GNC_IRLSpParams` class to provide the GNC schedule.

### Quadratic influence function `quadratic_influence_func.py`

This file provides the `QuadraticInfluenceFunc` class that implements the quadratic objective function that
is used in non-robust least squares. The main use of this class is to provide a simple mechanism for
building an instance of `SupGaussNewton` for the purpose of checking derivative calculations in the model
`residual_gradient()` method.

### Null parameter class `null_params.py`

When you want to apply standard IRLS, or use Sup-GN optimisation without GNC, wrap your influence function
in a `NullParams` instance. 

### Using a model reference `model_ref`

It is usually desirable to model your system with the minimum number of parameters, to avoid redundancy.
This is sometimes in conflict with symmetry and gauge invariance issues. The canonical example of this
is estimating a model containing a rotation. Consider the *3D point registration* problem of calculating the rotation and
translation between two point clouds, where the correspondence is known in advance. There is a closed-form
solution for this problem [9], but in the case of outliers we should look to implement this in the IRLS
and Sup-GN frameworks. To represent rotation minimally, we need to use three parameters. These might be
one of the various three-angle representations, or Rodrigues parameters. However any minimal representation
will have singularities. Worse than this, the choice of initial coordinate frame will affect the result
because of the non-linearities present in any minimal representation of rotation [10].

The solution to this problem is to use a non-minimal rotation representation as a "reference" for estimating
small changes. For instance you can use a rotation matrix, and combine it with a small rotation representation
such as Rodrigues parameters as implemented in `scipy.spatial.transform.Rotation.from_mrp`.
The idea is that before each IRLS/Sup-GN iteration, the reference rotation matrix is updated to the latest
rotation, including the change made at the previous iteration. The residual is calculated by combining
the reference rotation with the small rotation change at the current iteration. The residual derivatives
are calculated assuming that the small rotation parameters are zero.

The implementation of model reference in `gnc_smoothie` allows you to control it completely within
your model class. Here is how an implementation of a model class for 3D point cloud registration might look.
```
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ls_registration import LS_PointCloudRegistration

class PointRegistration:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        rotd = Rot.from_mrp(-0.25*model[0:3])
        self.__R = np.matmul(Rot.as_matrix(rotd), model_ref)
        self.__t = model[3:6]

    # r = y - R*x - t
    #   = Rs*R0*x + t, Rs = ( 1  -az  ay), R0*x = (R0_xx*x_x + R0_xy*x_y + R0_xz*x_z) = (R0x_x)
    #                       ( az  1  -ax)         (R0_yx*x_x + R0_yy*x_y + R0_yz*x_z)   (R0x_y)
    #                       (-ay  ax  1 )         (R0_zx*x_x + R0_zy*x_y + R0_zz*x_z)   (R0x_z)
    # where R0x = R0*x
    def residual(self, data_item, data_id:int=None) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array(y - np.matmul(self.__R,x) - self.__t)

    # dr   (  0     R0x_z -R0x_y)          dr
    # -- = (-R0x_z   0     R0x_x) = Rx_x,  -- = -I_3x3
    # da   ( R0x_y -R0x_x   0   )          dt
    def residual_gradient(self, data_item, data_id:int=None) -> np.array:
        x = data_item[0]
        Rx = np.matmul(self.__R,x)
        return np.array([[   0.0,  Rx[2], -Rx[1], -1.0,  0.0,  0.0],
                         [-Rx[2],    0.0,  Rx[0],  0.0, -1.0,  0.0],
                         [ Rx[1], -Rx[0],    0.0,  0.0,  0.0, -1.0]])

    def update_model_ref(self, model, prev_model_ref=None):
        rotd = Rot.from_mrp(-0.25*model[0:3])
        if prev_model_ref is None:
            R = Rot.as_matrix(rotd)
        else:
            R = np.matmul(Rot.as_matrix(rotd), prev_model_ref)

        # reset model parameters because they are subsumed by reference
        model[0:3] = 0.0

        # convert to quaternion and back to matrix to ensure orthogonality
        q = Rot.as_quat(Rot.from_matrix(R))
        return Rot.as_matrix(Rot.from_quat(q))

    # fits the model to the data
    def weighted_fit(self, data, data_ids, weight, scale) -> (np.array, np.array):
        R,t = LS_PointCloudRegistration(data, weight)
        model = np.zeros(6)
        model[3:6] = t
        return model,R
```
The reference rotation is first created in the `weighted_fit` method that implements the algorithm of [9]
to initiate the model. It returns the model and the rotation matrix `R` that is return as the initial `model_ref`.
In the `cache_model()` method, the reference rotation matrix $ R_0 $ and the small rotation $ R_s $
are combined as a cached rotation $ R = R_s R_0 $. This combined $ R $ is then used to calculate
the residual. In the `residual_gradient` method, the derivatives are calculated with respect to the small
rotation parameters, and a small angle approximation is used. The `update_model_ref` method updates the
reference rotation matrix $ R_0 $ based on the model change (small rotation) and the previous state
of the rotation reference `prev_model_ref`. The small rotation parameters in the model are reset to zero,
ready for the next iteration. A necessary refinement is that we should ensure that the rotation reference
maintains its orthogonality. Because it is computed incrementally, floating point errors could push it
away from being an actual rotation matrix. We solve this problem by converting into a quaternion and
back to a matrix.

## References

[1] P. Charbonnier et al. in "Deterministic edge-preserving regularization in computed imaging", PAMI 6(2), 1997.

[2] P.W. Holland & R.E. Welsch "Robust regression using iteratively reweighted least-squares", Communications in Statistics-theory and Methods, 6(9), 1977.

[3] L. Peng et al. "On the Convergence of IRLS and Its Variants in Outlier-Robust Estimation", CVPR 2023.

[4] A. Blake & A. Zisserman "Visual reconstruction", MIT Press, 1987.

[5] K. Levenberg "A method for the solution of certain non – linear problems in least squares", Quarterly of Applied Mathematics, 2, 1944.

[6] A. Björck, "Numerical Methods for Least Squares Problems", (1996)

[7] P.J. Huber, "Robust Estimation of a Location Parameter", The Annals of Mathematical Statistics 35(1), 1964.

[8] D. Geman and S. Geman, "Bayesian image analysis", in "Disordered systems and biological organization", Springer, 1986.

[9] B.K.P. Horn, H.M. Hilden and S. Negahdaripour, "Closed-form solution of absolute orientation using orthonormal matrices", Journal of the Optical Society of America, 5(7), 1988.

[10] P.F. McLauchlan, "Gauge invariance in projective 3D reconstruction", Proceedings IEEE Workshop on Multi-View Modeling and Analysis of Visual Scenes (MVIEW'99), 1999.
