# Implements GNC schedule for the Welsch influence function, as a wrapper for
# a WelschInfluenceFunc instance. The sigma parameter in the Welsch influence
# function is initialised at a high value sigma_limit, and gradually reduced
# over num_sigma_steps steps to a minimum value sigma_base approximating the
# size of the population distribution standard deviation (actually just above it).
# The internal parameter beta is the factor to multiply sigma by at each step,
# so beta <= 1.
# Normally max_niterations should be set to a higher value than num_sigma_steps
# so that a good number of iterations will be applied at the final sigma value.
import math


class GNC_WelschParams:
    def __init__(
        self,
        influence_func_instance,
        sigma_base: float,
        sigma_limit: float = None,
        num_sigma_steps: int = None,
        max_niterations: int = 1000000,
    ):
        self.influence_func_instance = influence_func_instance
        self.__sigma_base = sigma_base
        self.__sigma_limit = sigma_base if sigma_limit is None else sigma_limit
        if sigma_limit != sigma_base:
            # GNC schedule for sigma
            self.__num_sigma_steps = (
                max_niterations if num_sigma_steps is None else num_sigma_steps
            )
            if self.__num_sigma_steps > max_niterations:
                raise ValueError("Too many sigma steps")

            self.__beta = math.exp(
                (math.log(sigma_base) - math.log(self.__sigma_limit))
                / (self.__num_sigma_steps - 1.0)
            )
        else:
            # fixed sigma
            self.__num_sigma_steps = 0
            self.__beta = 1.0

        # set parameters to final values
        self.reset(False)

    def reset(self, init: bool = True):
        if init:
            self.influence_func_instance.sigma = self.__sigma_limit
        else:
            self.influence_func_instance.sigma = self.__sigma_base

    # whether we have reached the final sigma value
    def at_final_stage(self) -> bool:
        return True if self.influence_func_instance.sigma <= self.__sigma_base else False

    # update sigma to a lower value
    def update(self):
        self.influence_func_instance.sigma = max(
            self.__sigma_base, self.__beta * self.influence_func_instance.sigma
        )
