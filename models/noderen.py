import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

# import os

# from torchdiffeq import odeint_adjoint as odeint


class ContractiveNodeREN(nn.Module):
    # TODO: What about the h ?
    def __init__(self, nx, ny, nu, nq, h, sigma="tanh", epsilon=0.01, bias=False):
        """Model of a contractive NodeREN.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -bias (bool, optional): choose if the model has non-null biases. Default to False.
        """
        super().__init__()
        # Dimensions of Inputs, Outputs, States

        self.nx = nx  # no. internal-states
        self.ny = ny  # no. output
        self.nu = nu  # no. inputs
        self.nq = nq  # no. non-linear states
        self.epsilon = epsilon
        std = 0.0005  # standard deviation used to draw randomly the initial weights of the model.

        self.h = h

        self.eye_mask_w = torch.eye(self.nq)

        # Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx, nx) * std)
        # self.Ptilde = nn.Parameter(torch.randn(nx,nx)*std)
        self.Chi = nn.Parameter(torch.randn(nx, nq) * std)
        # Initialization of the Weights:
        self.Y1 = nn.Parameter(torch.randn(nx, nx) * std)
        self.B2 = nn.Parameter(torch.randn(nx, nu) * std)
        self.D12 = nn.Parameter(torch.randn(nq, nu) * std)
        self.C2 = nn.Parameter(torch.randn(ny, nx) * std)
        self.D21 = nn.Parameter(torch.randn(ny, nq) * std)
        self.D22 = nn.Parameter(torch.randn(ny, nu) * std)
        BIAS = bias
        if BIAS:
            self.bx = nn.Parameter(torch.randn(nx, 1) * std)
            self.bv = nn.Parameter(torch.randn(nq, 1) * std)
            self.by = nn.Parameter(torch.randn(ny, 1) * std)
        else:
            self.bx = torch.zeros(nx, 1)
            self.bv = torch.zeros(nq, 1)
            self.by = torch.zeros(ny, 1)
        self.X = nn.Parameter(
            torch.randn(nx + nq, nx + nq) * std)
        # Initialization of the last Parameters:
        self.A = torch.zeros(nx, nx)
        # self.Y= torch.zeros(nx,nx)
        self.D11 = torch.zeros(nq, nq)
        self.C1 = torch.zeros(nq, nx)
        self.B1 = torch.zeros(nx, nq)
        self.P = torch.zeros(nx, nx)
        self.updateParameters()  # Update of: A, B1, C1, D11
        # Choosing the activation function:
        if sigma == "tanh":
            self.act = nn.Tanh()
        elif sigma == "sigmoid":
            self.act = nn.Sigmoid()
        elif sigma == "relu":
            self.act = nn.ReLU()
        elif sigma == "identity":
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx)
        self.P = P
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq)
        # Partition of H in --> [H1 H2;H3 H4]
        h1, h2 = torch.split(H, [self.nx, self.nq], dim=0)  # split the matrices in two big rows
        H1, H2 = torch.split(h1, [self.nx, self.nq], dim=1)  # split each big row in two chunks
        H3, H4 = torch.split(h2, [self.nx, self.nq], dim=1)

        Y = -0.5 * (H1 + P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P), Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        self.C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2 - self.Chi
        self.B1 = F.linear(torch.inverse(P), Z.T)

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        w = self.calculate_w(t, xi, u)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(
            torch.ones(n_initial_states, 1, 1), self.bx) + F.linear(u, self.B2)
        return xi_

    def output(self, t, xi, u):
        """
        Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        w = self.calculate_w(t, xi, u)
        By = F.linear(torch.ones(n_initial_states, 1), self.by)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt

    def calculate_w(self, t, xi, u):
        # TODO: agregar los biases!
        batch_size = xi.shape[0]
        w = torch.zeros(batch_size, 1, self.nq)
        # update each row of w
        for i in range(self.nq):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(xi, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u, self.D12[i, :])
            w = w + (self.eye_mask_w[i, :] * self.act(v)).reshape(batch_size, 1, self.nq)
        return w

    # simulation
    def rollout(self, xi_init, u_log, T, train=False):
        # TODO: Remove T as input and obtain it from the u_log length
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
        xi_log = xi
        for t in range(T):
            xi_dot = self.forward(t=t, xi=xi, u=u_log[:, t:t+1, :])  # shape = (batch_size, 1, state_dim)
            xi = xi + self.h * xi_dot
            if t > 0:
                xi_log = torch.cat((xi_log, xi), 1)
        return xi_log


class StableSystem(nn.Module):
    # TODO: What about the h ?
    def __init__(self, nx, ny, nu, nq, h, sigma=None, epsilon=0.01, bias=False):
        """Model of a LTI system.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): useless
            -sigma (string): useless
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -bias (bool, optional): useless
        """
        super().__init__()
        # Dimensions of Inputs, Outputs, States

        self.nx = nx  # no. internal-states
        self.ny = ny  # no. output
        self.nu = nu  # no. inputs
        self.epsilon = epsilon
        std = 0.5  # standard deviation used to draw randomly the initial weights of the model.

        self.h = h

        self.lambdas = - torch.rand(nx) * 5 - epsilon
        self.V = torch.randn(nx,nx)
        assert torch.linalg.matrix_rank(self.V) == nx
        self.A = torch.matmul(self.V, torch.matmul(torch.diag(self.lambdas), self.V.transpose(0,1)))
        while not max(torch.real(torch.linalg.eig(self.A)[0])) < -100*epsilon:
            self.lambdas = - torch.rand(nx) * 5 - epsilon
            self.V = torch.randn(nx,nx)
            self.A = torch.matmul(self.V, torch.matmul(torch.diag(self.lambdas), self.V.transpose(0, 1)))

        # Initialization of the Free Matrices:
        self.B = nn.Parameter(torch.randn(nx, nu) * std)
        self.C = nn.Parameter(torch.randn(ny, nx) * std)
        self.D = nn.Parameter(torch.randn(ny, nu) * std)

    def updateParameters(self):
        pass

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        xi_ = F.linear(xi, self.A) + F.linear(u, self.B)
        return xi_

    def output(self, t, xi, u):
        """
        Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        yt = F.linear(xi, self.C) + F.linear(u, self.D22)
        return yt

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
        xi_log = xi
        for t in range(T):
            xi_dot = self.forward(t=t, xi=xi, u=u_log[:, t:t+1, :])  # shape = (batch_size, 1, state_dim)
            xi = xi + self.h * xi_dot
            if t > 0:
                xi_log = torch.cat((xi_log, xi), 1)
        return xi_log
