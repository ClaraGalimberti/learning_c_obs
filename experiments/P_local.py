import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import lr_scheduler
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt
import os

from models.nn_model_P import Model_Px, Swish

from utils import Params, set_seed, Loss_Stats, ROOT_DIR
from plants.sys_sincos import SystemSinCos
from plants.second_order_system import SecondOrderSystem
from plants.duffing import ReverseDuffingOscillator
from plants.lorenz import Lorenz
from plants.predator_prey import PredatorPrey
from plants.van_der_pol import VanDerPol


def learn_or_load_P(sys_name, q, r, learn=False, test=False):

    # ----- System ----------
    # sys = SecondOrderSystem(h=0.001)
    # sys = SystemSinCos(h=0.002)
    # sys = ReverseDuffingOscillator(h=0.001)
    # sys = Lorenz(h=0.001)  # Not implemented, it has state_dim==3!
    # sys = PredatorPrey(h=0.002)

    if sys_name == "VanderPol":
        json_file = "vdp_params.json"
        sys = VanDerPol(h=0.003)
    elif sys_name == "ReverseDuffingOscillator":
        json_file = "duffing_params.json"
        sys = ReverseDuffingOscillator(h=0.001)
    elif sys_name == "PredatorPrey":
        json_file = "predprey_params.json"
        sys = PredatorPrey(h=0.002)
    elif sys_name == "SinCosSystem":
        json_file = "sincos_params.json"
        sys = SystemSinCos(h=0.002)
    else:
        raise ValueError("Not implemented dataset")

    args = Params(os.path.join(ROOT_DIR, "experiments", "params_learning_P", json_file))
    args.q = q
    args.r = r

    msg = args.text_to_print()
    print(msg)

    # ----- Params ----------
    Q = torch.eye(sys.state_dim) * args.q
    R = torch.eye(sys.out_dim) * args.r
    f = sys.dynamics
    h = sys.output
    # KKLDataset = {"VDP": VanDerPol,
    #               "Param": OscillatorWithParameter,
    #               "Rossler": Rossler}[args.dataset]
    #
    NET_ARCH = [128*2, 128*2]
    ACTIVATION = Swish
    #
    CRITERION = torch.nn.MSELoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    #
    # # %% problem definition
    # Q = torch.eye(KKLDataset.x_dim).to(device) * args.q
    # R = torch.eye(KKLDataset.y_dim).to(device) * args.r
    # f = KKLDataset.get_derivs
    # h = KKLDataset.get_y

    f_jacobian = vmap(jacrev(f))
    h_jacobian = vmap(jacrev(h))

    LEARN_INVERSE = True
    def get_residual(x, P, model):
        """
        Computes the residual of the Riccati equation.
        Parameters:
            x: shape(..., x_dim)
            P: shape(..., x_dim, x_dim)
        Returns:
            residual: shape(..., x_dim, x_dim)
        """
        P_shape = P.shape
        x = x.view(-1, x.shape[-1])
        P = P.view(-1, x.shape[-1], x.shape[-1])
        n_batch, x_dim = x.shape

        # -- algebraic terms
        A = f_jacobian(x) # df/dx
        C = h_jacobian(x) # dh/dx
        if LEARN_INVERSE:
            residual = P @ A.transpose(-2, -1) \
                      + A @ P \
                      - P @ C.transpose(-2, -1) @ torch.inverse(R) @ C @ P \
                      + Q
        else:
            residual = P @ A \
                     + A.transpose(-2, -1) @ P \
                     - C.transpose(-2, -1) @ torch.inverse(R) @ C \
                     + P @ Q @ P

        # -- differential term
        fx = f(x).unsqueeze(-1) # (n_batch, x_dim, 1)
        dP_dx_flat = vmap(jacrev(model))(x).view(n_batch, x_dim**2, x_dim)
        dP_dt = (dP_dx_flat @ fx).view(n_batch, x_dim, x_dim)
        if LEARN_INVERSE:
            residual = residual - dP_dt
        else:
            residual = residual + dP_dt
        return residual.view(*P_shape)


    # %% --- init datasets ---
    set_seed(args.seed)

    if sys_name == "PredatorPrey":
        bias = 0
    else:
        bias = args.axis_limit
    # Generate coordinate grids depending on system dimension
    axes = [
        torch.rand(args.n_points) * 2 * args.axis_limit - bias
        for _ in range(sys.state_dim)
    ]
    # Create an N-dimensional meshgrid
    mesh = torch.meshgrid(*axes, indexing="ij")
    # Stack all coordinates along the last dimension
    x_data_all = torch.stack(mesh, dim=-1).reshape(-1, 1, sys.state_dim)  # (N^d, 1, d)

    dataset = TensorDataset(x_data_all)

    train_size = int(0.8 * len(dataset))   # 80% for training
    valid_size = len(dataset) - train_size  # 20% for test

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    # train_dataset = KKLDataset(use_traj=False,
    #                             n_samples=10000,
    #                             traj_len=-1,
    #                             noise_std=0)
    # # train_dataset = KKLDataset(use_traj=True,
    # #                            n_samples=1000,
    # #                            traj_len=1000,
    # #                            noise_std=0)
    # train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    #
    # valid_dataset = KKLDataset(use_traj=False,
    #                             n_samples=1000,
    #                             traj_len=-1,
    #                             noise_std=0)
    # # valid_dataset = KKLDataset(use_traj=True,
    # #                            n_samples=100,
    # #                            traj_len=1000,
    # #                            noise_std=0)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize)


    # %% --- init model ---
    # model = Model_Px(x_dim=KKLDataset.x_dim,
    #                  net_arch=NET_ARCH,
    #                  activation_fn=ACTIVATION).to(device)
    model = Model_Px(x_dim=sys.state_dim,
                     net_arch=NET_ARCH,
                     activation_fn=ACTIVATION)

    SAVE_FOLDER = os.path.join(ROOT_DIR, "experiments", "trained_models", sys_name)
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    file_name = args.name + "_q%.2f" % args.q + "_r%.2f" % args.r + ".pth"
    SAVE_NAME = os.path.join(SAVE_FOLDER, file_name)

    # Check if file exists
    if os.path.exists(SAVE_NAME) and not learn:
        try:
            model.load_state_dict(torch.load(SAVE_NAME))
            print("Model for P(x) loaded")
        except Exception as e:
            print("Error loading model for P(x): ", e)
            print("Train needed")
            learn = True
    else:
        print("Model for P(x) not found. Train needed")
        learn = True


    # %% --- training ---

    if learn:

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs//4, gamma=.5)

        if args.epochs > 0:
            print("==== training ====")
        loss_stats = Loss_Stats()
        for epoch in range(args.epochs + 1):
            #-- train
            for x, in train_loader:
                P = model(x)
                residual = get_residual(x, P, model)
                loss = CRITERION(residual, torch.zeros_like(residual))
                loss_stats.batch_step(loss.item())
                if epoch > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            loss_train, _ = loss_stats.epoch_step(epoch, 'train')

            #-- valid
            with torch.no_grad():
                for i, (x,) in enumerate(valid_loader):
                    P = model(x)
                    residual = get_residual(x, P, model)
                    loss = CRITERION(residual, torch.zeros_like(residual))
                    loss_stats.batch_step(loss.item())
            loss_valid, is_best = loss_stats.epoch_step(epoch, 'valid')

            #-- verbose / plot evolution
            lr = scheduler.get_last_lr()[0]
            print('epoch %i train %.2e valid %.2e  lr %.2e' %
                  (epoch, loss_train, loss_valid, lr))
            if epoch > 0 and epoch % 10 == 0:
                loss_stats.render()

            # -- best loss ?
            if epoch > 0 and is_best:
                print("\t save model", args.name)
                torch.save(model.state_dict(), SAVE_NAME)

            #-- update learning rate
            if epoch > 0:
                scheduler.step()

        model.load_state_dict(torch.load(SAVE_NAME))

    # %% --- test the observer ---
    if test:
        print("==== test observer ====")
        # -- generate a trajectory
        set_seed(args.seed)

        n_traj = 1
        traj_len = 10000
        w_log = torch.zeros(n_traj, traj_len, sys.state_dim)
        v_log = torch.zeros(n_traj, traj_len, sys.out_dim)
        # w_log = 5. * sys.h * torch.randn(n_traj, traj_len, sys.state_dim)
        # v_log = 5. * sys.h * torch.randn(n_traj, traj_len, sys.out_dim)

        t = torch.linspace(0, traj_len-1, traj_len)
        x_init = torch.tensor([[1., 2.]]).repeat(n_traj, 1, 1)
        x_log, y_log = sys.rollout(x_init, w_log, v_log, traj_len)

        ts, xs, ys = t.squeeze(), x_log.squeeze(), y_log.squeeze()

        # test_dataset = KKLDataset(use_traj=True, n_samples=1)
        # ts, xs, ys, _ = test_dataset.generate_trajectories(n_traj=1,
        #                                                     traj_len=1000,
        #                                                     noise_std=0)
        # ts, xs, ys = ts[0], xs[0], ys[0] # unsqueeze batch dim

        to_torch = lambda a: torch.tensor(a)
        to_numpy = lambda a: a.numpy()

        # xs, ys = to_torch(xs), to_torch(ys)

        # -- init observer
        xs_obs = torch.zeros_like(xs)
        # xs_obs[0] = to_torch(np.random.uniform(0.75*args.axis_limit, KKLDataset.x0_high))
        xs_obs[0] = torch.tensor([-2.,-1.])

        # -- run
        @torch.no_grad()
        def get_dxobs_dt(P, y, x_obs, x):
            # -- require x
            dh_dx_obs = h_jacobian(x_obs.unsqueeze(0)).squeeze(0) # shape(y_dim, x_dim)
            if LEARN_INVERSE:
                dxobs_dt = f(x_obs) + (P @ dh_dx_obs.T @ torch.inverse(R) @ dh_dx_obs @ (x - x_obs).unsqueeze(-1)).squeeze(-1) # TODO: check
            else:
                dxobs_dt = f(x_obs) + (torch.inverse(P) @ dh_dx_obs.T @ torch.inverse(R) @ dh_dx_obs @ (x - x_obs).unsqueeze(-1)).squeeze(-1) # TODO: check

            # -- do not require x (only if h is linear)
            # C = h_jacobian(x_obs.unsqueeze(0)).squeeze(0)
            # dxobs_dt = f(x_obs) + P @ C.T @ torch.inverse(R) @  (y - h(x_obs)) # TODO: check
            return dxobs_dt

        def get_dP_dt(P, x):
            A = f_jacobian(x.unsqueeze(0)).squeeze(0) # (x_dim, x_dim)
            C = h_jacobian(x.unsqueeze(0)).squeeze(0) # (y_dim, x_dim)
            if LEARN_INVERSE:
                dP_dt = P @ A.transpose(-2, -1) \
                      + A @ P \
                      - P @ C.transpose(-2, -1) @ torch.inverse(R) @ C @ P \
                      + Q
            else:
                dP_dt = P @ A \
                      + A.transpose(-2, -1) @ P \
                      - C.transpose(-2, -1) @ torch.inverse(R) @ C \
                      + P @ Q @ P
            return dP_dt

        Ps, residuals = [], [] # (debug)
        P = model(xs_obs[0])
        for k, t in enumerate(ts[:-1]):
            @torch.no_grad()
            def monitor(x):
                # -- debug: store P /residual
                P_model = model(x)
                Ps.append(P_model)
                res = get_residual(x.unsqueeze(0), P_model.unsqueeze(0), model)
                residuals.append((res**2).mean())

            x, y, x_obs = xs[k], ys[k], xs_obs[k]
            dxobs_dt = get_dxobs_dt(P, y, x_obs, x)
            xs_obs[k+1] = x_obs + sys.h * dxobs_dt  # Euler
            # xs_obs[k+1] = RK4(lambda x_obs, _: get_dxobs_dt(P, y, x_obs, x), sys.h, x_obs)
            monitor(xs_obs[k]) # (debug)

            P = model(x_obs) # shape(x_dim, x_dim) # NN based
            # P = P + KKLDataset.dt * get_dP_dt(P, x_obs)  # dinamique de P

        xs, xs_obs, ys = to_numpy(xs), to_numpy(xs_obs), to_numpy(ys)
        Ps = to_numpy(torch.stack(Ps))
        residuals = to_numpy(torch.stack(residuals))

        # -- plot P coeffs / Riccati residual
        plt.figure()
        plt.subplot(211)
        plt.plot(ts[:-1], Ps.reshape((-1, sys.state_dim**2)))
        plt.legend(['P coeffs'])
        plt.subplot(212)
        plt.plot(ts[:-1], residuals)
        plt.legend(['Riccati MSE'])
        plt.tight_layout()
        plt.show()

        # -- plot x / x_obs
        plt.figure()
        for i in range(model.x_dim):
            plt.subplot(model.x_dim + 1, 1, i + 1)
            plt.plot(ts, xs[:, i], label='$x_%i$' % (i + 1))
            plt.plot(ts, xs_obs[:, i], label='$hat x_%i$' % (i + 1))
            if (sys_name == "Rossler" and i == 1) or (sys_name != "Rossler" and i == 0):
                plt.plot(ts, ys, '--', label='y')
            plt.legend()
        plt.subplot(model.x_dim + 1, 1, model.x_dim + 1)
        plt.plot(ts, np.mean((xs - xs_obs)**2, axis=1))
        plt.legend(['MSE'])
        plt.show()

    # Return model
    return model


if __name__ == '__main__':
    sys_name = "VanderPol"
    # sys_name = "ReverseDuffingOscillator"
    learn = False
    learn_or_load_P(sys_name, q=1., r=1., learn=learn, test=True)
