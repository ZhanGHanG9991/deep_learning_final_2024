import torch
import torch.nn as nn

class VanDerPol(nn.Module):
    def __init__(self, alpha1, alpha2, W):
        super(VanDerPol, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.W = W

    def forward(self, t, hidden):
        h1, h2 = hidden[0], hidden[1]
        #TODO: residual or ...?
        dh1dt = self.alpha1 * h1 * (1 - h2**2) + self.alpha2 * h2 + self.W*h1
        dh2dt = -h1
        return torch.stack([dh1dt, dh2dt], dim = 0)

class RNNwithODE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, alpha1, alpha2, W):
        super(RNNwithODE, self).__init__()
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.ode_func = VanDerPol(alpha1, alpha2, W)
        self.rnn = nn.RNNCell(input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def input_to_hidden(self, x):
        h1_initial = torch.randn(x.size(0), self.hidden_dim)
        h2_initial = torch.randn(x.size(0), self.hidden_dim)
        hidden = torch.stack([h1_initial, h2_initial]).to(x.device)
        return hidden
    
    def forward(self, x, hidden = None, t_span = None):
        if hidden is None:
            hidden = self.input_to_hidden(x)

        if t_span is None:
            t_span = torch.linspace(0, 1, 5, device=hidden.device)

        # hidden = odeint(self.ode_func, hidden, t_span)
        for _ in range(len(t_span)):
            hidden += 0.2 * self.ode_func(t_span, hidden)
        output = self.fc(hidden[0])
        return output, hidden
