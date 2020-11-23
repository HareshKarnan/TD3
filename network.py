import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Net(torch.nn.Module):
    """network that defines the forward dynamics model"""
    def __init__(self, n_feature, n_hidden, n_output, activations=nn.ReLU, action_activation=False):
        super(Net, self).__init__()

        self.fc_in = nn.Linear(n_feature, n_hidden)
        self.fc_h1 = nn.Linear(n_hidden, n_hidden)
        self.fc_h2 = nn.Linear(n_hidden, n_hidden)

        self.fc_out = nn.Linear(n_hidden, n_output)
        self.last_activation = action_activation
        self.activations = activations

    # pylint: disable=arguments-differ
    def forward(self, x):
        out = self.activations()(self.fc_in(x))
        out = self.activations()(self.fc_h1(out))
        out = self.activations()(self.fc_h2(out))
        out = self.fc_out(out)

        if self.last_activation: return nn.Tanh()(out) # using a tanh activation for actions ?

        return out

class StochasticNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, activations=nn.Tanh, action_activation=False):
        super().__init__()
        self.network = GaussianNet(n_feature=n_feature,
                                   n_hidden=n_hidden,
                                   n_output=n_output,
                                   activations=activations,
                                   action_activation=action_activation)

    def forward(self, x):
        return self.network(x)

    def loss(self, x, y):
        normal = self.forward(x)
        loss = torch.mean(-normal.log_prob(y))
        return loss

    def sample(self, x):
        normal = self.forward(x)
        samples = normal.sample()
        return samples

class GaussianNet(torch.nn.Module):
    """network that defines the forward dynamics model"""
    def __init__(self, n_feature, n_hidden, n_output, activations=nn.Tanh, action_activation=False):
        super(GaussianNet, self).__init__()

        self.fc_in = nn.Linear(n_feature, n_hidden)
        self.fc_h1 = nn.Linear(n_hidden, n_hidden)
        self.fc_h2 = nn.Linear(n_hidden, n_hidden)

        self.fc_out = nn.Linear(n_hidden, 2*n_output)
        self.single_fc = nn.Linear(n_feature, 2*n_output)
        self.last_activation = action_activation
        self.activations = activations

        torch.nn.init.xavier_uniform_(self.fc_in.weight)
        torch.nn.init.xavier_uniform_(self.fc_h1.weight)
        torch.nn.init.xavier_uniform_(self.fc_h2.weight)

        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        torch.nn.init.xavier_uniform_(self.single_fc.weight)

    # pylint: disable=arguments-differ
    def forward(self, x):
        out = self.activations()(self.fc_in(x))
        out = self.activations()(self.fc_h1(out))
        out = self.activations()(self.fc_h2(out))
        out = self.fc_out(out)
        out = out + self.single_fc(x)

        out = out.view(-1, list(out.shape)[-1])
        mean, std = torch.split(out, out.shape[1]//2, dim=1)

        std = torch.exp(std)

        m = MultivariateNormal(mean, covariance_matrix=torch.diag_embed(std))
        return m