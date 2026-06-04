# When GNC is not required, you can switch it off by packaging your influence
# function instance into a NullParams instance. For instance, the Pseudo-Huber
# and the trivial quadratic influence functions are convex so do not require GNC.
class GNC_NullParams:
    def __init__(self, influence_func_instance):
        self.influence_func_instance = influence_func_instance

    def reset(self, *, init: bool = True) -> None:
        pass

    def n_steps(self) -> int:
        return 0

    def alpha(self) -> float:
        return 1.0

    def filter_size(self) -> float:
        return 0.0

    def supports_factor_argument(self) -> bool:
        return False

    def increment(self) -> None:
        pass
