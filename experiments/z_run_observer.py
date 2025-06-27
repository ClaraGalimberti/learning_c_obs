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
from models.non_linearities import CouplingLayer, HamiltonianSIE



# ----- 1. Hyperparameters ----------
p = Params()

p.t_end = 1000
p.state_dim = 2
p.out_dim = 1
p.batch_size = 1

# Observer dimensions
p.nx = 4
p.nq = 4

# Learning hyperparameters
p.seed = 0
p.epochs = 4000
p.learning_rate = 2.5e-3

# Griding for training
p.n_points = 11  # number of points per dimension
p.axis_limit = 6

# Noise
p.w_std = 0.04
p.v_std = 0.04

# ----- 2. Initialization ----------
torch.manual_seed(p.seed)

# ----- 3. Logger and saving folders ----------
logger = WrapLogger()
figs_folder = os.path.join(ROOT_DIR, 'experiments','figs')
os.makedirs(figs_folder, exist_ok=True)

# ----- 4. Generate Noise ----------
w = torch.zeros(p.batch_size, p.t_end, p.state_dim)
v = torch.zeros(p.batch_size, p.t_end, p.out_dim)

w_noisy = p.w_std * torch.randn(p.batch_size, p.t_end, p.state_dim)
v_noisy = p.v_std * torch.randn(p.batch_size, p.t_end, p.out_dim)

# ----- 5. Containers ----------
loss_log_1 = torch.zeros(p.epochs)
loss_log_2 = torch.zeros(p.epochs)

# ----- 6. System ----------
# sys = SecondOrderSystem()
# sys = SystemSinCos()
sys = ReverseDuffingOscillator()
# sys = Lorenz()
# sys = PredatorPrey()
# sys = VanDerPol()

# Let's plot a trajectory of the system
x_init = torch.tensor([[1., 2.]]).repeat(p.batch_size, 1, 1)
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

coup = CouplingLayer(dim_inputs=p.ny, dim_hidden=p.nq*20, dim_small=sys.state_dim)
# coup = HamiltonianSIE(n_layers=32, dim_inputs=p.ny, dim_small=sys.state_dim)
# tau = FCNN(dim_in=p.ny, dim_out=sys.state_dim, dim_hidden=p.nq*40)
# sigma = FCNN(dim_in=sys.state_dim, dim_out=p.ny, dim_hidden=p.nq*40)

# ----- 8. Optimizer and losses -----
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(coup.parameters()), lr=p.learning_rate)
def lr_schedule(epoch):
    if epoch < 0.5*p.epochs:
        return 1.0
    else:
        return 0.5
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()

# ----- 9. Training -----
if sys.state_dim != 2:
    raise NotImplementedError("The training is only implemented for 2D systems.")
for epoch in range(p.epochs):
    optimizer.zero_grad()
    # Grid  in[-2, 2] x [-2, 2]
    x1_vals = torch.rand(p.n_points) * 2*p.axis_limit - p.axis_limit
    x2_vals = torch.rand(p.n_points) * 2*p.axis_limit - p.axis_limit
    x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
    # stack xs
    x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

    x_data_j = coup.lift(x_data)
    jacobian = vmap(jacrev(coup))(x_data_j).squeeze()
    jacobian = coup.delift(jacobian).transpose(1,2)
    # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
    # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

    loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian),
                   sys_z(t=0, xi=coup(coup.lift(x_data)), u=sys.output(x_data)))
    loss2 = loss_2(coup.delift(coup.forward_inverse(coup(coup.lift(x_data)))),
                   x_data)
    # loss2 = torch.zeros(1)
    loss = loss1 + 0.1*loss2
    loss_log_1[epoch] = loss1.detach()
    loss_log_2[epoch] = loss2.detach()
    if epoch % 10 == 0:
        print("Epoch: %4i \t--- Loss: %12.6f \t---||--- Loss 1: %12.6f \t---- Loss 2: %4.2f"
              % (epoch, loss, loss1, loss2))
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
# TODO: set initialization at the original coordinates and transform
xi_init = torch.randn(1,1,sys_z.nx)
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
plt.savefig(os.path.join(figs_folder,"linear_states_after_training.pdf"), format='pdf')
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
plt.savefig(os.path.join(figs_folder, "linear_latent_after_training.pdf"), format='pdf')
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
plt.savefig(os.path.join(figs_folder, "linear_noisy_states_after_training.pdf"), format='pdf')
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
plt.savefig(os.path.join(figs_folder, "linear_noisy_latent_after_training.pdf"), format='pdf')
plt.show()

# Let's also look at the Loss:
plt.figure()
plt.plot(torch.linspace(0, p.epochs-1, p.epochs), loss_log_1, label=r'$\ell_1$')
plt.legend()
ax = plt.gca()
ax.set_yscale('log')
plt.savefig(os.path.join(figs_folder + "loss.pdf"), format='pdf')
plt.show()

print("Hola")
