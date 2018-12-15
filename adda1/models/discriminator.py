"""Discriminator model for ADDA."""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            # 500 500
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            # 500 2
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
