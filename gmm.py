import torch.nn as nn

class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(128, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return res, score

