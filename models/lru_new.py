import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRU_new(nn.Module):
    """
    Implements a Linear Recurrent Unit (LRU) following the parametrization of "Resurrecting Linear Recurrences" paper.
    The LRU is simulated using Parallel Scan (fast!) when scan=True (default), otherwise recursively (slow)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 state_features: int,
                 h: float,
                 lambda_re_min: float = -5.,
                 lambda_re_max: float = -0.5,
                 max_phase: float = 6.283,
                 ):
        super().__init__()

        # set dimensions
        self.dim_internal = state_features
        self.nx = state_features
        self.dim_in = in_features
        self.dim_out = out_features
        self.h = h
        self.epsilon = 1e-3

        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)*2-1
        self.alpha = nn.Parameter(torch.log(-(u1 * (lambda_re_max - lambda_re_min) + lambda_re_min)))
        self.theta = nn.Parameter(max_phase * u2)
        lambda_re = -torch.exp(self.alpha) - self.epsilon
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(lambda_re) - torch.square(lambda_re))))
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        # self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

        H1 = torch.randn([self.dim_internal, self.dim_internal])
        H2 = torch.randn([self.dim_internal, self.dim_internal])
        rho = torch.cat([torch.cat([H1, -H2], dim=1),
                       torch.cat([H2, H1], dim=1)], dim=0)
        self.rho = rho / torch.norm(rho)
        self.rhoinv = torch.inverse(self.rho)

        self.T_inv = ((1 / torch.sqrt(torch.tensor(2.0))) *
                      torch.cat([torch.cat([torch.eye(state_features), torch.eye(state_features)], dim=1),
                                 torch.cat([-1j * torch.eye(state_features), 1j * torch.eye(state_features)], dim=1)
                                ], dim=0))  # shape (2n, 2n), complex

        self.P = torch.matmul(torch.complex(self.rho, torch.zeros_like(self.rho)), self.T_inv)
        self.Pinv = torch.inverse(self.P)

        # define trainable params
        self.training_param_names = ['D', 'nu_log', 'theta_log', 'gamma_log', 'B', 'C']

    def updateParameters(self):
        pass

    def forward(self, t, x, u, validation=False):
        """
        Forward pass of SSM.
        Args:
            x (torch.Tensor): Current real state with the size of (batch_size, 1, 2*self.dim_internal).
            u (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        batch_size = u.shape[0]

        x_complex = torch.complex(x, torch.zeros_like(x))
        x_tilde = F.linear(x_complex, self.Pinv)
        x_bar, x_bar_conj = torch.split(x_tilde, [self.dim_internal, self.dim_internal], dim=-1)

        lambda_re = -torch.exp(self.alpha) - self.epsilon
        lambda_im = self.theta
        lambda_c = torch.complex(lambda_re, lambda_im)  # A matrix

        x_bar_dot = lambda_c * x_bar + F.linear(torch.complex(u, torch.zeros(1)), self.B)
        x_tilde_dot = torch.concat([x_bar_dot, torch.conj(x_bar_dot)], dim =-1)
        x_dot = F.linear(x_tilde_dot, self.P)

        if validation:
            print("Validation, sum of abs imaginary part: ", torch.abs(x_dot.imag).sum())

        return x_dot.real

    def output(self, t, xi, u):
        y_out = 2 * F.linear(xi, self.C).real + F.linear(u, self.D)
        return y_out

    # simulation
    def rollout(self, xi_init, u_log, T, train=False):
        """
        rollout REN for rollouts of the input
        Args:
            - xi_init of shape (batch_size, 1, nx)
            - u_log of shape (batch_size, T, nu)
            - T (int):  number of steps
            # - ts of shape (T)
        Return:
            - xi_log of shape (batch_size, T, nx)
        """
        # initial state
        if train:
            xi_log = self._sim(xi_init, u_log, T)
        else:
            with torch.no_grad():
                xi_log = self._sim(xi_init, u_log, T)
        return xi_log

    def _sim(self, xi_init, u_log, T):
        xi = xi_init
        x_log = xi_init
        for t in range(T):
            x_dot = self.forward(t=t, x=xi, u=u_log[:, t:t + 1, :])  # shape = (batch_size, 1, state_dim)
            xi = xi + self.h * x_dot
            if t > 0:
                x_log = torch.cat((x_log, xi), 1)
        return x_log