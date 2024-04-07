import torch
import torch.nn as nn

class discriminator(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            activation: nn.Module = nn.ReLU,
            layer_type: str = "MLP",
            device: str = "cpu",
            fewshot_columns=None,
    ):
        super(discriminator, self).__init__()

        self.device = torch.device(device)
        self.activation = activation
        self.fewshot_columns=fewshot_columns
        
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Sequential(*[nn.Linear(256, 256) for _ in range(hidden_dims)])
        self.linear3 = nn.Linear(256, 1)
        
        self._init_net()
        self.to(device)
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # nn.init.orthogonal_(layer.weight)
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x, clip=True):
        x = torch.tanh(self.linear1(x).clip(-10, 10) if clip else self.linear1(x))
        for fc in self.linear2:
            x = torch.tanh(fc(x).clip(-10, 10) if clip else fc(x))
        out = self.linear3(x)
        return out
