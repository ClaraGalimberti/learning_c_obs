import os.path

import torch
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

p.t_end = 1000*10
p.state_dim = 2
p.out_dim = 1
p.plot_batch_size = 1

# Observer dimensions
p.nx = 8
p.nq = 4

# Learning hyperparameters
p.seed = 0
p.epochs = 4000*4
p.learning_rate = 5e-3

# Griding for training
p.n_points = 50  # number of points per dimension
p.axis_limit = 4
p.batch = 50

# Noise
p.w_std = 10. /2
p.v_std = 10. /2

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
loss_log_3 = torch.zeros(p.epochs)

# ----- 5. System ----------
# sys = SecondOrderSystem(h=0.001)
# sys = SystemSinCos(h=0.002)
# sys = ReverseDuffingOscillator(h=0.001)
# sys = Lorenz(h=0.001)  # Not implemented, it has state_dim==3!
sys = PredatorPrey(h=0.002)
# sys = VanDerPol(h=0.003)

# ----- 6. Generate Noise ----------
w = torch.zeros(p.plot_batch_size, p.t_end, p.state_dim)
v = torch.zeros(p.plot_batch_size, p.t_end, p.out_dim)

w_noisy = p.w_std * sys.h * torch.randn(p.plot_batch_size, p.t_end, p.state_dim)
v_noisy = p.v_std * sys.h * torch.randn(p.plot_batch_size, p.t_end, p.out_dim)

# Let's plot a trajectory of the system
x_init = torch.linspace(1, sys.state_dim, sys.state_dim).repeat(p.plot_batch_size, 1, 1)
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
# tau = FCNN(dim_in=p.ny, dim_out=sys.state_dim, dim_hidden=p.nq*40)
# sigma = FCNN(dim_in=sys.state_dim, dim_out=p.ny, dim_hidden=p.nq*40)
coup = MappingT(dim_inputs=p.ny, dim_hidden=p.nq*40, dim_small=sys.state_dim)

# ----- 8. Optimizer and losses -----
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(coup.parameters()), lr=p.learning_rate)
def lr_schedule(epoch):
    if epoch < 0.5*p.epochs:
        return 1.0
    elif epoch < 0.75*p.epochs:
        return 0.25
    else:
        return 0.05
    # elif epoch < 0.5*p.epochs:
    #     return 0.1
    # else:
    #     return 0.01
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()
loss_3 = MultiStepLoss(sys, sys_z, coup)

# ----- 9. Data for PINN ------
if sys.name == "PredatorPrey":
    bias = 0
else:
    bias = p.axis_limit
# x1_vals = torch.rand(p.n_points) * 2 * p.axis_limit - bias
# x2_vals = torch.rand(p.n_points) * 2 * p.axis_limit - bias
# x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
# # stack xs
# x_data_all = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, sys.state_dim)  # ((2*p.n_points)**2, 1, 2)
# Generate coordinate grids depending on system dimension
axes = [
    torch.rand(p.n_points) * 2 * p.axis_limit - bias
    for _ in range(sys.state_dim)
]
# Create an N-dimensional meshgrid
mesh = torch.meshgrid(*axes, indexing="ij")
# Stack all coordinates along the last dimension
x_data_all = torch.stack(mesh, dim=-1).reshape(-1, 1, sys.state_dim)  # (N^d, 1, d)

# ----- 9. Training -----
if sys.state_dim != 2:
    raise NotImplementedError("The training is only implemented for 2D systems.")
for epoch in range(p.epochs):
    optimizer.zero_grad()
    idx = torch.randperm(x_data_all.shape[0])[:p.batch]
    x_data = x_data_all[idx]  # (p.batch, 1, sys.state_dim)
    x_data_j = coup.lift(x_data)
    jacobian = vmap(jacrev(coup))(x_data_j).squeeze()
    jacobian = coup.delift(jacobian).transpose(1,2)
    # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
    # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

    loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian),
                   sys_z(t=0, xi=coup(coup.lift(x_data)), u=sys.output(x_data)))
    loss2 = loss_2(coup.delift(coup.forward_inverse(coup(coup.lift(x_data)))), x_data)
    loss3 = loss_3(x_data) * 100
    loss = loss1 + loss2
    loss_log_1[epoch] = loss1.detach()
    loss_log_2[epoch] = loss2.detach()
    loss_log_3[epoch] = loss3.detach()
    if epoch % 10 == 0:
        msg =  "Epoch: %4i \t--- " % epoch
        msg += "Loss: %12.6f \t---||--- " % loss
        msg += "Loss 1: %12.6f \t---- " % loss1
        msg += "Loss 2: %12.6f \t---- " % loss2
        msg += "Loss 3: %8.6f" % loss3
        logger.info(msg)
    loss.backward()
    optimizer.step()
    scheduler.step()
    sys_z.updateParameters()

# ----- 10. Plots -----
# Let's see how my observer works after training:
time = torch.linspace(0, p.t_end-1, p.t_end)
x, y = sys.rollout(x_init, 0*w_noisy, 0*v_noisy, p.t_end)
x_noisy, y_noisy = sys.rollout(x_init, w_noisy, v_noisy, p.t_end)

# x_hat_log:
xi_init = coup.forward(coup.lift(x_init*(-2)))
# xi_init = coup.forward(coup.lift(torch.randn(1,1,sys.state_dim)))
# xi_init = torch.randn(1,1,sys_z.nx)
xi = sys_z.rollout(xi_init, y, p.t_end)
x_hat = coup.delift(coup.forward_inverse(xi))
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, x[0,:,0], label=r"$x_1(t)$")
plt.plot(time, x_hat[0,:,0].detach(), label=r"$\hat{x}_1(t)$")
plt.legend()
plt.subplot(3,1,2)
plt.plot(time, x[0,:,1], label=r"$x_2(t)$")
plt.plot(time, x_hat[0,:,1].detach(), label=r"$\hat{x}_2(t)$")
plt.legend()
plt.subplot(3,1,3)
plt.plot(time, y[0, :, 0], label=r"$y(t)$")
plt.plot(time, sys.output(x_hat[0,:,:]).detach(), label=r"$\hat{y}(t)$")
plt.legend()
plt.savefig(os.path.join(figs_folder, sys.name+"_states_after_train.pdf"), format='pdf')
plt.show()
#
# Let's also plot the z-dynamics
plt.figure()
plt.subplot(2,1,1)
for i in range(p.nx):
    plt.plot(time, xi[0,:,i], label=r"$\xi_{%i}(t)$" % i)
plt.legend()
plt.subplot(2,1,2)
plt.plot(time, y[0, :, 0], label=r"$y(t)$")
plt.plot(time, sys.output(x_hat[0,:,:]).detach(), label=r"$\hat{y}(t)$")
plt.legend()
plt.savefig(os.path.join(figs_folder, sys.name+"_latent_after_train.pdf"), format='pdf')
plt.show()

# NOISE:
# x_hat_log_noisy:
# TODO: set initialization at the original coordinates and transform
xi_init = torch.randn(1,1,sys_z.nx)
xi_noisy = sys_z.rollout(xi_init, y_noisy, p.t_end)
x_hat_noisy = coup.delift(coup.forward_inverse(xi_noisy))
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
#
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
plt.plot(torch.linspace(0, p.epochs-1, p.epochs), loss_log_1, label=r'$\ell_1$')
plt.plot(torch.linspace(0, p.epochs-1, p.epochs), loss_log_2, label=r'$\ell_2$')
plt.plot(torch.linspace(0, p.epochs-1, p.epochs), loss_log_3, label=r'$\ell_3$')
plt.legend()
ax = plt.gca()
ax.set_yscale('log')
plt.savefig(os.path.join(figs_folder + sys.name+"_loss.pdf"), format='pdf')
plt.show()

print("Hola")
