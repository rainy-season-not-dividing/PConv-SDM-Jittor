import jittor as jt
from jittor.optim import Optimizer


class Adagrad(Optimizer):
    """Small Jittor Adagrad optimizer compatible with PyTorch baseline settings."""

    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0):
        super().__init__(list(params), lr)
        self.eps = eps
        self.weight_decay = weight_decay
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad():
                    continue
                grad = g + p * weight_decay if weight_decay else g
                v.update(v + grad * grad)
                p.update(p - lr * grad / (jt.sqrt(v) + eps))
        self.post_step()

