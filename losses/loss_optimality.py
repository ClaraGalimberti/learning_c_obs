import torch
from torch.func import vmap, jacrev, hessian


class LocalOptLoss(torch.nn.Module):
    def __init__(self, sys, sys_z, coup, model_Px, r):
        super().__init__()
        # self.sys = sys
        # self.coup = coup
        self.f = sys.dynamics
        self.h = sys.output
        self.tau = coup.tau
        self.T = coup.sigma
        self.psi = sys_z
        self.model_Pinv = model_Px
        self.Rinv = torch.eye(sys.out_dim) * (1/r)
        # self.R = torch.eye(sys.out_dim) * r
        # self.loss_MSE = torch.nn.MSELoss()

    def forward(self, x_batch, e_batch):
        # squeeze dims equal to 1 (the one in the middle)
        if x_batch.ndim == 3 and x_batch.shape[1] == 1:
            x_batch = x_batch.squeeze(1)  # -> (B, n)
        if e_batch.ndim == 3 and e_batch.shape[1] == 1:
            e_batch = e_batch.squeeze(1)  # -> (B, m)

        # Mark requires_grad True for x_batch (we need jacobians wrt x_hat which depends on x through tau(T(x)+e) ...)
        # Important: do this BEFORE vmap. functorch disallows requires_grad_ inside vmapped function.
        x_batch = x_batch.detach().requires_grad_(True)
        e_batch = e_batch.detach().requires_grad_(True)
        # e_batch = e_batch.detach()  # e.g. if e is small noise, and you don't need grads wrt e; if yes, set requires_grad_(True)

        # x_batch has dimension (B, n)
        # e_batch has dimension (B, m)
        # We want to apply vmap over the dimension of B. This is the dim 0 for both x_batch and e_batch
        loss_values = vmap(self.compute_loss_single, in_dims=(0, 0))(x_batch, e_batch)  # -> (B,)

        return loss_values.mean()

    def compute_loss_single(self, x, e):
        # x: shape (n,), e: shape (m,)
        # No llamar a x.requires_grad_() aquí!

        z = self.T(x) + e  # (m,)
        x_hat = self.tau(z)  # (n,)
        # phi evaluated at (T(x_hat), h(x_hat))
        phi_eval = self.psi(0, self.T(x_hat).unsqueeze(0), self.h(x_hat).unsqueeze(0)).squeeze()  # (m,)
        y_hat = self.h(x_hat)  # (p,)

        # Jacobians:
        # jacrev for vector->vector returns (out_dim, in_dim)
        Jh = jacrev(self.h)(x_hat)  # shape (p, n)
        Jf = jacrev(self.f)(x_hat)  # shape (n, n)
        Jtau = jacrev(self.tau)(self.T(x_hat))  # shape (n, m) if tau: R^m -> R^n
        Jphi = jacrev(lambda zz: self.psi(0, zz.unsqueeze(0), y_hat.unsqueeze(0)).squeeze())(self.T(x_hat))

        # Hessian of tau wrt z at z = T(x_hat)
        Htau = hessian(self.tau)(self.T(x_hat))  # shape (n, m, m)

        # P^{-1}(x_hat)
        P_inv = self.model_Pinv(x_hat)  # expected shape (n, n)

        # First term: P_inv * Jh^T * R_inv * Jh * (x - x_hat)
        diff = x - x_hat  # (n,)
        # Jh: (p,n) -> Jh.T: (n,p)
        # compute inside: Jh @ diff -> (p,), R_inv @ (p,) -> (p,), Jh.T @ (p,) -> (n,)
        term1 = P_inv @ (Jh.T @ (self.Rinv @ (Jh @ diff)))  # (n,)

        # Second big bracket: -Jf @ Jtau + Jtau @ Jphi + red_term
        # # first addend:
        # Jf: (n,n)
        # Jtau: (n,m)
        # Jf @ Jtau -> (n,m)
        term2_1 = - (Jf @ Jtau)
        # # second addend:
        # Jtau: (n,m)
        # Jphi: (m,m)
        # Jtau @ Jphi -> (n,m)
        term2_2 = (Jtau @ Jphi)
        # # third addend (the red term, with a tensor):
        # Hessian Htau (n,m,m) with phi_eval (m,) -> (n,m)
        # we want for each output dim i: sum_j Htau[i,j,k] * phi_eval[j]  -> result[i,k]
        Htau_phi = torch.einsum('imj,m->ij', Htau, phi_eval)  # (n,m)
        term2 = term2_1 + term2_2 + Htau_phi  # (n,m)

        # multiply by e (m,)
        # term2 @ e -> (n,)
        term2e = term2 @ e

        # loss per sample: norm of (term1 - term2e)
        loss = torch.norm(term1 - term2e)
        return loss
