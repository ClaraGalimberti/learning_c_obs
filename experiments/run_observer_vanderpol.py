import torch
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt

from plants.van_der_pol import VanDerPol
from models.noderen import ContractiveNodeREN, StableSystem
from models.static_NNs import FCNN



plt.rcParams["text.usetex"] = True
torch.manual_seed(0)

# second_traning_phase = True
second_traning_phase = False

T = 2000
state_dim = 2
out_dim = 1
batch_size = 1
w_log = torch.zeros(batch_size, T, state_dim)
v_log = torch.zeros(batch_size, T, out_dim)

w_log_noisy = 0.05 * torch.randn(batch_size, T, state_dim)
v_log_noisy = 0.05 * torch.randn(batch_size, T, out_dim)

if second_traning_phase:
    folder = ''
else:
    folder = 'figs_no_second_phase/'

# Training parameters:
epochs = 8000
learning_rate = 2.5e-3

sys = VanDerPol()

# Let's plot a trajectory of the system
t = torch.linspace(0, T-1, T)
x_init = torch.tensor([[1., 2.]]).repeat(batch_size, 1, 1)
x_log, y_log = sys.rollout(x_init, w_log, v_log, T)
x_log_noisy, y_log_noisy = sys.rollout(x_init, w_log_noisy, v_log_noisy, T)
plt.plot(t, x_log[0,:,:], label=[r"$x_1(t)$", r"$x_2(t)$"])
plt.legend()
plt.show()
plt.plot(t, y_log[0,:,:], label=r"$y(t)$")
plt.legend()
plt.show()

# Let's create the observer
nx = 8
ny = nx
nu = sys.out_dim
nq = 4
sys_z = ContractiveNodeREN(nx, ny, nu, nq, h=sys.h)
# sys_z = StableSystem(nx, ny, nu, nq, h=sys.h)

tau = FCNN(dim_in=ny, dim_out=sys.state_dim, dim_hidden=nq*10)
sigma = FCNN(dim_in=sys.state_dim, dim_out=ny, dim_hidden=nq*10)

# For training:
# Let's grid the state:
n_points = int(81)  # number of points per dimension

# Grid  in[-2, 2] x [-2, 2]
x1_vals = torch.linspace(-4, 4, n_points)
x2_vals = torch.linspace(-4, 4, n_points)
x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
# stack xs
x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

# Setting the training:
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(tau.parameters()) + list(sigma.parameters()),
                             lr=learning_rate)

loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()

    # Grid  in[-2, 2] x [-2, 2]
    x1_vals = torch.rand(n_points) * 8 - 4
    x2_vals = torch.rand(n_points) * 8 - 4
    x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
    # stack xs
    x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

    jacobian = vmap(jacrev(sigma))(x_data).squeeze().transpose(1,2)

    # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
    # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

    loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian), sys_z(t, sigma(x_data), sys.output(x_data)))
    loss2 = loss_2(tau(sigma(x_data)), x_data)
    loss = 10*loss1 + loss2
    # if epoch % 1000 == 0 and epoch > 4100:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
    print("Epoch: %i \t--- Loss: %.2f \t---||--- Loss 1: %.2f \t---- Loss 2: %.2f"
          % (epoch, 1e6 * loss, 1e6 * loss1, 1e6 * loss2))
    loss.backward()
    optimizer.step()
    sys_z.updateParameters()

if second_traning_phase:
    for epoch in range(int(epochs/10)):
        optimizer.zero_grad()

        # Grid  in[-2, 2] x [-2, 2]
        x1_vals = torch.rand(n_points) * 8 - 4
        x2_vals = torch.rand(n_points) * 8 - 4
        x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
        # stack xs
        x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

        jacobian = vmap(jacrev(sigma))(x_data).squeeze().transpose(1,2)

        # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
        # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

        with torch.no_grad():
            zz = sigma(x_data).detach()
        loss2 = loss_2(tau(zz), x_data)
        loss = loss2

        print("Epoch: %i \t--- Loss: %.2f \t---||--- Loss 1: %.2f \t---- Loss 2: %.2f"
              % (epoch, 1e6 * loss, 1e6 * loss1, 1e6 * loss2))
        loss.backward()
        optimizer.step()
        # sys_z.updateParameters()

# Let's see how my observer works after training:

for i in range(4):
    # x_hat_log:
    xi_init = torch.randn(1,1,sys_z.nx)*2
    xi_log = sys_z.rollout(xi_init, y_log, T)
    x_hat_log = tau(xi_log)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, x_log[0,:,0], label=r"$x_1(t)$")
    plt.plot(t, x_hat_log[0,:,0].detach(), label=r"$\hat{x}_1(t)$")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t, x_log[0,:,1], label=r"$x_2(t)$")
    plt.plot(t, x_hat_log[0,:,1].detach(), label=r"$\hat{x}_2(t)$")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t, y_log[0, :, 0], label=r"$y(t)$")
    plt.plot(t, sys.output(x_hat_log[0,:,:]).detach(), label=r"$\hat{y}(t)$")
    plt.legend()
    plt.savefig(folder + "vanderpol_after_training_%i.pdf" % i, format='pdf')
    plt.show()

    # x_hat_log_noisy:
    xi_init = torch.randn(1,1,sys_z.nx)*2
    xi_log_noisy = sys_z.rollout(xi_init, y_log_noisy, T)
    x_hat_log_noisy = tau(xi_log_noisy)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, x_log_noisy[0,:,0], label=r"$x_1(t)$")
    plt.plot(t, x_hat_log_noisy[0,:,0].detach(), label=r"$\hat{x}_1(t)$")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t, x_log_noisy[0,:,1], label=r"$x_2(t)$")
    plt.plot(t, x_hat_log_noisy[0,:,1].detach(), label=r"$\hat{x}_2(t)$")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t, y_log_noisy[0, :, 0], label=r"$y(t)$")
    plt.plot(t, sys.output(x_hat_log_noisy[0,:,:]).detach(), label=r"$\hat{y}(t)$")
    plt.legend()
    plt.savefig(folder + "vanderpol_noisy_after_training_%i.pdf" % i, format='pdf')
    plt.show()

    x_init = torch.rand(1,1,2)
    w_log_noisy = 0.05 * torch.randn(batch_size, T, state_dim)
    v_log_noisy = 0.05 * torch.randn(batch_size, T, out_dim)
    x_log, y_log = sys.rollout(x_init, w_log, v_log, T)
    x_log_noisy, y_log_noisy = sys.rollout(x_init, w_log_noisy, v_log_noisy, T)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(r"$\xi(t)$")
    for j in range(nx):
        plt.plot(t, xi_log[0, :, j], linewidth=0.5, label=r"$\xi_{%i}(t)$" % j)
    # this is an inset axes over the main axes
    a = plt.axes([.25, .58, .64, .13])
    for j in range(nx):
        plt.plot(t, xi_log[0, :, j], linewidth=0.5, label=r"$\xi_{%i}(t)$" % j)
    a.set_ylim([-0.2, 0.2])
    plt.subplot(2, 1, 2)
    plt.title(r"$\xi(t) - \sigma(x(t))$")
    for j in range(nx):
        plt.plot(t, xi_log[0, :, j] - sigma(x_log).detach()[0, :, 1], linewidth=0.5,
                 label=r"$\xi_{%i}(t) - \sigma(x(t))$" % j)
    a = plt.axes([.25, .16, .64, .13])
    for j in range(nx):
        plt.plot(t, xi_log[0, :, j] - sigma(x_log).detach()[0, :, 1], linewidth=0.5,
                 label=r"$\xi_{%i}(t) - \sigma(x(t))$" % j)
    a.set_ylim([-0.2, 0.2])
    plt.savefig(folder + "vanderpol_latent_after_training_%i.pdf" % i, format='pdf')
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(r"$\xi(t)$")
    for j in range(nx):
        plt.plot(t, xi_log_noisy[0, :, j], linewidth=0.5, label=r"$\xi_{%i}(t)$" % j)
    # a = plt.axes([.25, .58, .64, .13])
    # for j in range(nx):
    #     plt.plot(t, xi_log_noisy[0, :, j], linewidth=0.5, label=r"$\xi_{%i}(t)$" % j)
    # a.set_ylim([-0.4, 0.4])
    plt.subplot(2, 1, 2)
    plt.title(r"$\xi(t) - \sigma(x(t))$")
    for j in range(nx):
        plt.plot(t, xi_log_noisy[0, :, j] - sigma(x_log_noisy).detach()[0, :, 1], linewidth=0.5,
                 label=r"$\xi_{%i}(t) - \sigma(x(t))$" % j)
    # a = plt.axes([.25, .16, .64, .13])
    # for j in range(nx):
    #     plt.plot(t, xi_log_noisy[0, :, j] - sigma(x_log_noisy).detach()[0, :, 1], linewidth=0.5,
    #              label=r"$\xi_{%i}(t) - \sigma(x(t))$" % j)
    # a.set_ylim([-0.4, 0.4])
    plt.savefig(folder + "vanderpol_noisy_latent_after_training_%i.pdf" % i, format='pdf')
    plt.show()

print("Hola")
