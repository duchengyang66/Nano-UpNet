"""Loss smoothing utility for stable GAN training logs."""


class LossSmoother:
    def __init__(self, window_size):
        if window_size <= 0:
            window_size = 1
        self.window_size = window_size
        self.losses = []

    def reset(self):
        self.losses = []

    def update(self, loss_dict):
        if not loss_dict:
            return
        self.losses.append(dict(loss_dict))
        if len(self.losses) > self.window_size:
            self.losses.pop(0)

    def get_smoothed(self):
        if not self.losses:
            return {}
        keys = self.losses[0].keys()
        out = {}
        for k in keys:
            vs = [d[k] for d in self.losses if k in d]
            if vs:
                out[k] = sum(vs) / len(vs)
        return out

    def __len__(self):
        return len(self.losses)
