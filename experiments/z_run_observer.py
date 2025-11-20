import os.path

import torch
import time
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt

from utils import WrapLogger, Params, ROOT_DIR
from plants.sys_sincos import SystemSinCos
from plants.second_order_system import SecondOrderSystem
from plants.duffing import ReverseDuffingOscillator
from plants.lorenz import Lorenz
from plants.predator_prey import PredatorPrey
from plants.van_der_pol import VanDerPol
from models.noderen import ContractiveNodeREN
from models.lru_new import LRU_new
from models.non_linearities import CouplingLayer, HamiltonianSIE, MappingT
from losses.losses import OneStepLoss, MultiStepLoss


# ----- 1. Hyperparameters ----------
p = Params()

p.n_tau = 160
# Observer dimensions
p.nx = 4
p.nq = 8

# Learning hyperparameters
p.seed = 0
p.epochs = 160
p.learning_rate = 5e-3

# Griding for training
p.n_points = 50000  # number of points where to evaluate PDE
p.batch = 1000

# Simulations:
# Noise
p.w_std = 10. /2 *0
p.v_std = 10. /2 *0 +83.
p.t_end = 1000*10
p.plot_batch_size = 100

# ----- 2. Initialization ----------
torch.manual_seed(p.seed)

# ----- 3. Logger and saving folders ----------
logger = WrapLogger()
logger.info(p.text_to_print())
figs_folder = os.path.join(ROOT_DIR, 'experiments','figs')
os.makedirs(figs_folder, exist_ok=True)

# ----- 4. Containers ----------
loss_log_1 = torch.zeros(p.epochs)
loss_log_2 = torch.zeros(p.epochs)
loss_log_multistep = torch.zeros(p.epochs)

# ----- 5. System ----------
# sys = SecondOrderSystem(h=0.001)
# sys = SystemSinCos(h=0.002)
# sys = ReverseDuffingOscillator(h=0.001)
# sys = Lorenz(h=0.0005)  # Not implemented, it has state_dim==3!
# sys = PredatorPrey(h=0.002)
sys = VanDerPol(h=0.003)

# ----- 6. Generate Noise ----------
w = torch.zeros(p.plot_batch_size, p.t_end, sys.state_dim)
v = torch.zeros(p.plot_batch_size, p.t_end, sys.out_dim)

w_noisy = p.w_std * sys.h * torch.randn(p.plot_batch_size, p.t_end, sys.state_dim)
v_noisy = p.v_std * sys.h * torch.randn(p.plot_batch_size, p.t_end, sys.out_dim)

# Let's plot a trajectory of the system
x_init = torch.rand(p.plot_batch_size, 1, sys.state_dim) * (sys.axis_limit['x0high'] - sys.axis_limit['x0low']) + sys.axis_limit['x0low']
plt.figure()
plt.subplot(1, 2, 1)
sys.plot_trajectory(x_init, p.t_end)
plt.subplot(1, 2, 2)
sys.plot_trajectory(x_init, p.t_end, w_noisy, w_noisy)
plt.show()

# ----- 7. Observer -----
p.ny = p.nx
p.nu = sys.out_dim
sys_z = ContractiveNodeREN(p.nx, p.ny, p.nu, p.nq, h=sys.h)
# sys_z = LRU_new(in_features=p.nu, out_features=p.ny, state_features=p.nx//2, h=sys.h)

# coup = CouplingLayer(dim_inputs=p.ny, dim_hidden=p.nq*40, dim_small=sys.state_dim)
# coup = HamiltonianSIE(n_layers=32, dim_inputs=p.ny, dim_small=sys.state_dim)
coup = MappingT(dim_inputs=p.ny, dim_hidden=p.n_tau, dim_small=sys.state_dim)

# ----- 8. Optimizer and losses -----
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(coup.parameters()), lr=p.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p.epochs//4, gamma=.5)
loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()
loss_multistep = MultiStepLoss(sys, sys_z, coup)

# ----- 9. Data for PINN ------
# generate data uniform random
# x_data_all = torch.rand(p.n_points, 1, sys.state_dim) * 2 * p.axis_limit - bias
x_data_all = torch.rand(p.n_points, 1, sys.state_dim) * (sys.axis_limit['high'] - sys.axis_limit['low']) + sys.axis_limit['low']

# ----- 9. Training -----
# if sys.state_dim != 2:
#     raise NotImplementedError("The training is only implemented for 2D systems.")
n_samples = x_data_all.shape[0]
n_batches = n_samples//p.batch

save_folder = os.path.join(ROOT_DIR, "experiments", "trained_models", sys.name)
os.makedirs(save_folder, exist_ok=True)
file_name = "temp" + ".pth"
save_path = os.path.join(save_folder, file_name)

tic = time.time()
for epoch in range(p.epochs):
    idxs = torch.randperm(n_samples)
    loss_log_1[epoch] = 0
    loss_log_2[epoch] = 0
    loss_log_multistep[epoch] = 0
    for i in range(n_batches):
        optimizer.zero_grad()
        idx = idxs[i*p.batch:(i+1)*p.batch]
        x_data = x_data_all[idx]  # (p.batch, 1, sys.state_dim)
        x_data_j = coup.lift(x_data)
        jacobian = vmap(jacrev(coup))(x_data_j).squeeze()
        jacobian = coup.delift(jacobian).transpose(1,2)
        loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian),
                       sys_z(t=0, xi=coup(coup.lift(x_data)), u=sys.output(x_data)))
        loss2 = loss_2(coup.delift(coup.forward_inverse(coup(coup.lift(x_data)))), x_data)
        loss_ms = loss_multistep(x_data) * 100
        loss_reg = loss_1(coup.tau.network[-1].weight, torch.zeros_like(coup.tau.network[-1].weight))
        loss = loss1 + loss2 + loss_reg
        loss.backward()
        optimizer.step()
        sys_z.updateParameters()

        loss_log_1[epoch] += loss1.detach()
        loss_log_2[epoch] += loss2.detach()
        loss_log_multistep[epoch] += loss_ms.detach()
    scheduler.step()
    is_best = epoch == 0 or loss_log_1[epoch] + loss_log_2[epoch] < (loss_log_1[:epoch] + loss_log_2[:epoch]).min()
    if epoch > 0 and is_best:
        checkpoint = {'sys_z': sys_z.to('cpu').state_dict(),
                      'coup': coup.to('cpu').state_dict()}
        torch.save(checkpoint, save_path)
    if epoch % 10 == 0:
        msg =  "Epoch: %4i \t--- " % epoch
        msg += "Loss: %12.6f \t---||--- " % ((loss_log_1[epoch] + loss_log_2[epoch]).detach()/n_batches)
        msg += "Loss 1: %12.6f \t---- " % (loss_log_1[epoch].detach()/n_batches)
        msg += "Loss 2: %12.6f \t---- " % (loss_log_2[epoch].detach()/n_batches)
        msg += "Loss multistep: %8.6f \t---||--- " % (loss_log_multistep[epoch].detach()/n_batches)
        msg += "Elapsed time: %.2f" % (time.time() - tic)
        logger.info(msg)
msg = "\t Model saved at %s" % save_path
logger.info(msg)

# ----- 10. Plots -----
# Let's see how my observer works after training:
time = torch.linspace(0, (p.t_end-1)*sys.h, p.t_end)
x, y = sys.rollout(x_init, w, v, p.t_end)
x_noisy, y_noisy = sys.rollout(x_init, w_noisy, v_noisy, p.t_end)

# Rollouts:
x_hat_init = torch.rand(p.plot_batch_size, 1, sys.state_dim) * (sys.axis_limit['x0high'] - sys.axis_limit['x0low']) + sys.axis_limit['x0low']
xi_init = coup.forward(coup.lift(x_hat_init))
xi = sys_z.rollout(xi_init, y, p.t_end)
x_hat = coup.delift(coup.forward_inverse(xi))
xi_noisy = sys_z.rollout(xi_init, y_noisy, p.t_end)
x_hat_noisy = coup.delift(coup.forward_inverse(xi_noisy))

# x_hat:
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, x[0,:,0], label=r"$x_1(t)$")
plt.plot(time, x_hat[0,:,0].detach(), label=r"$\hat{x}_1(t)$")
plt.legend(loc="right")
plt.subplot(3,1,2)
plt.plot(time, x[0,:,1], label=r"$x_2(t)$")
plt.plot(time, x_hat[0,:,1].detach(), label=r"$\hat{x}_2(t)$")
plt.legend(loc="right")
plt.subplot(3,1,3)
# plt.plot(time, y[0, :, 0], label=r"$y(t)$")
# plt.plot(time, sys.output(x_hat[0,:,:]).detach(), label=r"$\hat{y}(t)$")
mse = ((x - x_hat).detach()**2).mean(axis=0).mean(axis=-1)
plt.plot(time, mse, label='MSE (on %i trajs)' % x.shape[0])
plt.legend(loc="upper right")
plt.yscale("log")
plt.suptitle("nq=%i - T and tau: %i - min of MSE %.2e" % (p.nq, coup.tau.network[0].weight.shape[0], min(mse)))
plt.tight_layout()
plt.savefig(os.path.join(figs_folder, sys.name+"_states_after_train.pdf"), format='pdf')
plt.show()

# -- plot z
plt.figure()
Tx = coup.forward(coup.lift(x)).detach()
for i in range(p.nx):
    plt.subplot(sys_z.nx + 1, 1, i + 1)
    plt.plot(time, Tx[0, :, i], label='$T(x)_%i$' % (i + 1))
    plt.plot(time, xi[0,:,i], label='$z_%i$' % (i + 1))
    plt.legend()
# -- plot error
plt.subplot(sys_z.nx + 1, 1, sys_z.nx + 1)
mse = ((xi - Tx)**2).mean(axis=0).mean(axis=-1)
plt.semilogy(time, mse, label='MSE (on %i trajs)' % xi.shape[0])
plt.legend(loc="upper right")
plt.suptitle("nq=%i - T and tau: %i - min of MSE %.2e" % (p.nq, coup.tau.network[0].weight.shape[0], min(mse)))
plt.tight_layout()
plt.savefig(os.path.join(figs_folder, sys.name+"_latent_after_train.pdf"), format='pdf')
plt.show()

# NOISE:
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, x_noisy[0,:,0], label=r"$x_1(t)$")
plt.plot(time, x_hat_noisy[0,:,0].detach(), label=r"$\hat{x}_1(t)$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(time, x_noisy[0,:,1], label=r"$x_2(t)$")
plt.plot(time, x_hat_noisy[0,:,1].detach(), label=r"$\hat{x}_2(t)$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(time, y_noisy[0, :, 0], label=r"$y(t)$")
plt.plot(time, sys.output(x_hat_noisy[0,:,:]).detach(), label=r"$\hat{y}(t)$")
plt.legend()
plt.savefig(os.path.join(figs_folder, sys.name+"_states_noisy_after_train.pdf"), format='pdf')
plt.show()

# Let's also plot the z-dynamics
plt.figure()
plt.subplot(2,1,1)
for i in range(p.nx):
    plt.plot(time, xi_noisy[0,:,i], label=r"$\xi_{%i}(t)$" % i)
plt.legend()
plt.subplot(2,1,2)
plt.plot(time, y_noisy[0, :, 0], label=r"$y(t)$")
plt.plot(time, sys.output(x_hat_noisy[0,:,:]).detach(), label=r"$\hat{y}(t)$")
plt.legend()
plt.savefig(os.path.join(figs_folder, sys.name+"_latent_noisy_after_train.pdf"), format='pdf')
plt.show()

# Let's also look at the Loss:
plt.figure()
plt.plot(range(p.epochs), loss_log_1, label=r'$\ell_1$')
plt.plot(range(p.epochs), loss_log_2, label=r'$\ell_2$')
plt.plot(range(p.epochs), loss_log_multistep, label=r'$\ell_3$')
plt.legend()
plt.yscale("log")
plt.savefig(os.path.join(figs_folder + sys.name+"_loss.pdf"), format='pdf')
plt.show()

print("Hola")
