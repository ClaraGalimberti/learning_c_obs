import torch
import torch.nn.functional as F


class Rossler(torch.nn.Module):
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7, h: float = 0.01):
        """
        Rössler attractor in CT
        Args:
            h (float):      Time constant for discretization.
        """
        super().__init__()
        self.state_dim = 3
        self.out_dim = 1

        self.a = a
        self.b = b
        self.c = c

        self.h = h

    def dynamics(self, x):
        assert x.shape[-1] == self.state_dim
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=-1)
        x1_ = - x2 - x3
        x2_ = x1 + self.a * x2
        x3_ = self.b * torch.ones_like(x3) + x3 * (x1 - self.c * torch.ones_like(x3))
        f = torch.cat([x1_, x2_, x3_], dim=-1)
        return f

    def output(self, x):
        assert x.shape[-1] == self.state_dim
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=-1)
        y = x2
        return y

    def noiseless_forward(self, t, x: torch.Tensor):
        """
        forward of the plant without the process noise.
        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
        """
        x = x.view(-1, 1, self.state_dim)

        # Calculate x_dot
        f = self.dynamics(x)
        # Calculate x^+ using forward Euler integration scheme
        x_ = x + self.h * f
        return x_  # shape = (batch_size, 1, state_dim)

    def forward(self, t, x, w):
        """
        forward of the plant with the process noise.
        Args:
            - t (int):          current time step
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)
        Returns:
            next state.
        """
        return self.noiseless_forward(t, x) + w.view(-1, 1, self.state_dim)

    # simulation
    def rollout(self, x_init, w_log, v_log, T, train=False):
        """
        rollout for rollouts of the process noise
        Args:
            - x_init of shape (batch_size, 1, state_dim)
            - w_log of shape (batch_size, T, state_dim)
            - v_log of shape (batch_size, T, out_dim)
            - T (int):  number of steps
            # - ts of shape (T)
        Return:
            - x_log of shape (batch_size, T, state_dim)
            - y_log of shape (batch_size, T, out_dim)
        """
        # initial state
        assert x_init.shape == w_log[:, 0:1, :].shape  # shape = (batch_size, 1, state_dim)
        if train:
            x_log, y_log = self._sim(x_init, w_log, v_log, T)
        else:
            with torch.no_grad():
                x_log, y_log = self._sim(x_init, w_log, v_log, T)
        return x_log, y_log

    def _sim(self, x_init, w_log, v_log, T):
        x = x_init
        x_log = x + w_log[:, 0:1, :]
        y_log = self.output(x) + v_log[:, 0:1, :]
        for t in range(T):
            x = self.forward(t=t, x=x, w=w_log[:, t:t+1, :])  # shape = (batch_size, 1, state_dim)
            if t > 0:
                x_log = torch.cat((x_log, x), 1)
                y_log = torch.cat((y_log, self.output(x) + v_log[:, t:t+1, :]), 1)
        return x_log, y_log
