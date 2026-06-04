# Implements GNC schedule for the Welsch influence function, as a wrapper for
# a WelschInfluenceFunc instance. The sigma parameter in the Welsch influence
# function is initialised at a high value sigma_limit, and gradually reduced
# over num_sigma_steps steps to a minimum value sigma_base approximating the
# size of the population distribution standard deviation (actually just above it).
# The internal parameter beta is the factor to multiply sigma by at each step,
# so beta <= 1.
import math


class GNC_WelschParams:
    def __init__(
        self,
        influence_func_instance,
        sigma_base: float,
        *,
        sigma_limit: float = None,
        num_sigma_steps: int = 1,
        min_beta: float = 0.2,
        beta_deriv_scale: float = 0.5,
    ):
        self.influence_func_instance = influence_func_instance
        self.__sigma_base = sigma_base
        self.__sigma_limit = sigma_base if sigma_limit is None else sigma_limit
        self.__num_sigma_steps = num_sigma_steps
        self.__beta = math.exp(
            (math.log(sigma_base) - math.log(self.__sigma_limit))
            / self.__num_sigma_steps
        )
        self.__min_beta = min_beta
        self.__beta_deriv_scale = beta_deriv_scale

        # set parameters to final values
        self.reset(init=False)

    def reset(self, *, init: bool = True) -> None:
        self.influence_func_instance.sigma = (
            self.__sigma_limit if init else self.__sigma_base
        )
        self.__step = 0

    def n_steps(self) -> int:
        return self.__num_sigma_steps

    def alpha(self) -> float:
        return (
            self.__step / self.__num_sigma_steps
            if self.influence_func_instance.sigma > self.__sigma_base
            else 1.0
        )

    def filter_size(self) -> float:
        return self.influence_func_instance.sigma

    def supports_factor_argument(self) -> bool:
        return True

    # update sigma to a lower value
    def increment(self, max_abs_normalised_deriv: float = None) -> None:
        self.__step = min(1 + self.__step, self.__num_sigma_steps)
        self.influence_func_instance.sigma = (
            self.__sigma_base
            if self.__step >= self.__num_sigma_steps
            else max(self.__sigma_base, (self.__beta if max_abs_normalised_deriv is None else min(self.__beta, max(self.__min_beta, 1.0 - self.__beta_deriv_scale/(0.001+max_abs_normalised_deriv)))) * self.influence_func_instance.sigma)
        )

    # used to check results when debugging
    #def set_alpha(self, alpha: float) -> None:
    #    if alpha <= 0.0:
    #        self.influence_func_instance.sigma = self.__sigma_limit
    #    elif alpha >= 1.0:
    #        self.influence_func_instance.sigma = self.__sigma_base
    #    else:
