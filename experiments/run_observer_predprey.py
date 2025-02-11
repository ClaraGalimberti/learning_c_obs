import torch
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt

from plants.predator_prey import PredatorPrey
from models.noderen import ContractiveNodeREN
from models.static_NNs import FCNN


torch.manual_seed(0)
T = 1000
state_dim = 2
out_dim = 1
batch_size = 1
w_log = 0.01 * torch.randn(batch_size, T, state_dim)
v_log = 0.01 * torch.randn(batch_size, T, out_dim)

# Training parameters:
epochs = 10000
learning_rate = 5e-2

sys = PredatorPrey()

# Let's plot a trajectory of the system
t = torch.linspace(0, T-1, T)
x_init = torch.tensor([[1., 2.]]).repeat(batch_size, 1, 1)
x_log, y_log = sys.rollout(x_init, w_log, v_log, T)
plt.plot(t, x_log[0,:,:], label=["$x_1(t)$", "$x_2(t)$"])
plt.legend()
plt.show()
plt.plot(t, y_log[0,:,:], label="$y(t)$")
plt.legend()
plt.show()

# Let's create the observer
nx = 10
ny = nx
nu = sys.out_dim
nq = 4
sys_z = ContractiveNodeREN(nx, ny, nu, nq, h=sys.h)

tau = FCNN(dim_in=ny, dim_out=sys.state_dim, dim_hidden=nq)
sigma = FCNN(dim_in=sys.state_dim, dim_out=ny, dim_hidden=nq)

# For training:
# Let's grid the state:
n_points = int(41)  # number of points per dimension

# Grid  in[-2, 2] x [-2, 2]
x1_vals = torch.linspace(1, 7, n_points)
x2_vals = torch.linspace(1, 7, n_points)
x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
# stack xs
x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

# Setting the training:
optimizer = torch.optim.Adam(list(sys_z.parameters()) + list(tau.parameters()) + list(sigma.parameters()),
                             lr=learning_rate)

loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.MSELoss()

log_loss_1 = torch.zeros(epochs)
log_loss_2 = torch.zeros(epochs)

for epoch in range(epochs):
    optimizer.zero_grad()

    # Grid  in[-2, 2] x [-2, 2]
    x1_vals = torch.rand(n_points) * 7
    x2_vals = torch.rand(n_points) * 7
    x1s, x2s = torch.meshgrid(x1_vals, x2_vals, indexing="ij")
    # stack xs
    x_data = torch.stack([x1s, x2s], dim=-1).reshape(-1, 1, 2)  # (batch_size, 1, 2)

    jacobian = vmap(jacrev(sigma))(x_data).squeeze().transpose(1,2)

    # print(torch.bmm(sys.dynamics(x_data), jacobian).shape)
    # print(sys_z(t, sigma(x_data), sys.output(x_data)).shape)

    loss1 = loss_1(torch.bmm(sys.dynamics(x_data), jacobian), sys_z(t, sigma(x_data), sys.output(x_data)))
    loss2 = loss_2(tau(sigma(x_data)), x_data)
    loss = loss1 + loss2
    log_loss_1[epoch] = loss1.detach()
    log_loss_2[epoch] = loss2.detach()

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
plt.subplot(2,1,1)
plt.plot(t, x_log[0,:,0], label="$x_1(t)$")
plt.plot(t, x_hat_log[0,:,0].detach(), label="$\hat{x}_1(t)$")
plt.legend()
plt.subplot(2,1,2)
plt.plot(t, x_log[0,:,1], label="$x_2(t)$")
plt.plot(t, x_hat_log[0,:,1].detach(), label="$\hat{x}_2(t)$")
plt.legend()
plt.savefig("predprey_state_after_training.pdf", format='pdf')
plt.show()

plt.figure()
plt.plot(t, y_log[0, :, 0], label="$y(t)$")
plt.plot(t, sys.output(x_hat_log[0,:,:]).detach(), label="$\hat{y}(t)$")
plt.legend()
plt.savefig("predprey_output_after_training.pdf", format='pdf')
plt.show()

plt.figure()
plt.plot(torch.linspace(0, epochs-1, epochs), log_loss_1, label='$\ell_1$')
plt.plot(torch.linspace(0, epochs-1, epochs), log_loss_2, label='$\ell_2$')
plt.legend()
ax = plt.gca()
ax.set_yscale('log')
plt.savefig("predprey_loss.pdf", format='pdf')
plt.show()

print("Hola")
