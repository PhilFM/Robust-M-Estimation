# When GNC is not required, you can switch it off by packaging your influence
# function instance into a NullParams instance. For instance, the Pseudo-Huber
# and the trivial quadratic influence functions are convex so do not require GNC. 
class NullParams:
    def __init__(self, influence_func_instance):
        self.influence_func_instance = influence_func_instance

    def reset(self, init: bool = True):
        pass

    def at_final_stage(self) -> bool:
        return True

    def update(self):
        pass
