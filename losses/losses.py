import torch


class OneStepLoss(torch.nn.Module):
    def __init__(self, sys, sys_z, coup):
        super().__init__()
        self.sys = sys
        self.sys_z = sys_z
        self.coup = coup
        self.loss_MSE = torch.nn.MSELoss()

    def forward(self, x_true):
        x_true_ = self.sys.noiseless_forward(0, x_true)
        xi = self.coup(self.coup.lift(x_true))
        xi_ = xi + self.sys_z.h * self.sys_z(t=0, xi=xi, u=self.sys.output(x_true))
        x_hat_ = self.coup.delift(self.coup.forward_inverse(xi_))
        loss_mse = self.loss_MSE(x_hat_, x_true_)
        return loss_mse


class MultiStepLoss(torch.nn.Module):
    def __init__(self, sys, sys_z, coup):
        super().__init__()
        self.sys = sys
        self.sys_z = sys_z
        self.coup = coup
        self.loss_MSE = torch.nn.MSELoss()

    def forward(self, x_init):
        steps = 2
        x, y = self.sys.rollout(x_init, torch.zeros(x_init.shape[0],steps,self.sys.state_dim), torch.zeros(x_init.shape[0],steps,1), steps)
        xi_init = self.coup.forward(self.coup.lift(x_init))
        xi = self.sys_z.rollout(xi_init, y, steps)
        x_hat = self.coup.delift(self.coup.forward_inverse(xi))
        loss_mse = self.loss_MSE(x_hat, x)
        return loss_mse