import torch
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt

from plants.lorenz import Lorenz
from models.noderen import ContractiveNodeREN
from models.static_NNs import FCNN


# NOT WORKING!

# plt.rcParams['text.usetex'] = True
torch.manual_seed(0)
T = 1000
state_dim = 3
out_dim = 1
batch_size = 1
w_log = 0.01 * torch.randn(batch_size, T, state_dim) *0
v_log = 0.01 * torch.randn(batch_size, T, out_dim) *0

# Training parameters:
epochs = 1000
learning_rate = 5e-3

sys = Lorenz()

# Let's plot a trajectory of the system
t = torch.linspace(0, T-1, T)
x_init = torch.tensor([[1., 2., 1.5]]).repeat(batch_size, 1, 1)
x_log, y_log = sys.rollout(x_init, w_log, v_log, T)
plt.plot(t, x_log[0,:,:], label=[r"$x_1(t)$", r"$x_2(t)$", r"$x_3(t)$"])
plt.legend()
plt.show()
plt.plot(t, y_log[0,:,:], label=r"$y(t)$")
plt.legend()
plt.show()

# Let's create the observer
nx = 4
ny = nx
nu = sys.out_dim
nq = 4
sys_z = ContractiveNodeREN(nx, ny, nu, nq, h=sys.h)

tau = FCNN(dim_in=ny, dim_out=sys.state_dim, dim_hidden=nq*20)
sigma = FCNN(dim_in=sys.state_dim, dim_out=ny, dim_hidden=nq*20)

# For training:
# Let's grid the state:
n_points = int(81)  # number of points per dimension
axis_limit = 20

# Grid  in[-2, 2] x [-2, 2]
x1_vals = torch.linspace(-axis_limit, axis_limit, n_points)
x2_vals = torch.linspace(-axis_limit, axis_limit, n_points)
x3_vals = torch.linspace(-axis_limit, axis_limit, n_points)
x1s, x2s, x3s = torch.meshgrid(x1_vals, x2_vals, x3_vals, indexing="ij")
# stack xs
x_data = torch.stack([x1s, x2s, x3s], dim=-1).reshape(-1, 1, 3)  # (batch_size, 1, 3)

# Setting the training:
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(tau.parameters()) + list(sigma.parameters()),
                             lr=learning_rate)

loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()

    # Grid  in[-2, 2] x [-2, 2]
    x1_vals = torch.rand(n_points) * (2*axis_limit) - axis_limit
    x2_vals = torch.rand(n_points) * (2*axis_limit) - axis_limit
    x3_vals = torch.rand(n_points) * (2*axis_limit) - axis_limit
    x1s, x2s, x3s = torch.meshgrid(x1_vals, x2_vals, x3_vals, indexing="ij")
    # stack xs
    x_data = torch.stack([x1s, x2s, x3s], dim=-1).reshape(-1, 1, 3)  # (batch_size, 1, 2)

    jacobian = vmap(jacrev(sigma))(x_data).squeeze().transpose(1,2)

    # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
    # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

    loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian), sys_z(t, sigma(x_data), sys.output(x_data)))
    loss2 = loss_2(tau(sigma(x_data)), x_data)
    loss = loss1 + loss2

    print("Epoch: %i \t--- Loss: %.2f \t---||--- Loss 1: %.2f \t---- Loss 2: %.2f"
          % (epoch, 1e6 * loss, 1e6 * loss1, 1e6 * loss2))
    loss.backward()
    optimizer.step()
    sys_z.updateParameters()

# Let's see how my observer works after training:

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
plt.plot(t, x_log[0,:,2], label=r"$x_3(t)$")
plt.plot(t, x_hat_log[0,:,2].detach(), label=r"$\hat{x}_3(t)$")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, y_log[0, :, 0], label=r"$y(t)$")
plt.plot(t, sys.output(x_hat_log[0,:,:]).detach(), label=r"$\hat{y}(t)$")
plt.legend()
plt.show()

print("Hola")
