import math


class GNC_IRLSpParams:
    def __init__(
        self,
        influence_func_instance,
        p,
        rscale,
        epsilon_base,
        epsilon_limit=None,
        beta=None,
    ):
        self.influence_func_instance = influence_func_instance
        self.influence_func_instance.p = p
        self.influence_func_instance.rscale = rscale
        self.epsilon_base = epsilon_base
        self.epsilon_limit = epsilon_base if epsilon_limit is None else epsilon_limit
        self.beta = 1.0 if beta is None else beta

        # set parameters to final values
        self.reset(False)

    def reset(self, init=True):
        if init:
            self.influence_func_instance.epsilon = self.epsilon_limit
        else:
            self.influence_func_instance.epsilon = self.epsilon_base

    def at_final_stage(self) -> bool:
        return (
            True if self.influence_func_instance.epsilon <= self.epsilon_base else False
        )

    def update(self):
        self.influence_func_instance.epsilon = max(
            self.epsilon_base,
            self.beta
            * math.pow(
                self.influence_func_instance.epsilon,
                2.0 - self.influence_func_instance.p,
            ),
        )  # epsilon = max(epsilon_base, beta*math.pow(epsilon,2.0-p))
