import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            device=torch.device('cpu')
    ):
        super(discriminator, self).__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_net()
        self.to(device)
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x, clip=True):
        out = self.trunk(x)
        return out
