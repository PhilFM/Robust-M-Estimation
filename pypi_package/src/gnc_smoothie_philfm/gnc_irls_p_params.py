import math


class GNC_IRLSpParams:
    def __init__(
        self,
        influence_func_instance,
        p,
        rscale,
        epsilon_base,
        *,
        epsilon_limit=None,
        beta=None,
    ):
        self.influence_func_instance = influence_func_instance
        self.influence_func_instance.p = p
        self.influence_func_instance.rscale = rscale
        self.__epsilon_base = epsilon_base
        self.__epsilon_limit = epsilon_base if epsilon_limit is None else epsilon_limit
        self.__beta = 0.8 if beta is None else beta

        # work out how many epsilon steps there are
        self.reset(init=True)
        self.__n_steps = 100000  # something big
        n_steps = 0
        while self.influence_func_instance.epsilon > self.__epsilon_base:
            self.increment()
            n_steps += 1

        self.__n_steps = n_steps

        # set parameters to final values
        self.reset(init=False)

    def reset(self, *, init=True) -> None:
        self.influence_func_instance.epsilon = (
            self.__epsilon_limit if init else self.__epsilon_base
        )
        self.__step = 0

    def n_steps(self) -> int:
        return self.__n_steps

    def alpha(self) -> float:
        return (
            self.__step / self.__n_steps
            if self.influence_func_instance.epsilon > self.__epsilon_base
            else 1.0
        )

    def increment(self) -> None:
        self.__step = min(1 + self.__step, self.__n_steps)
        self.influence_func_instance.epsilon = (
            self.__epsilon_base
            if self.__step >= self.__n_steps
            else self.__beta
            * math.pow(
                self.influence_func_instance.epsilon,
                2.0 - self.influence_func_instance.p,
            )
        )
