import os
import numpy as np
import paddle
import argparse
import random
import string
from utils import Utils
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# from tensorboardX import SummaryWriter

# 设置设备
# DEVICE = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
DEVICE = 'cpu'
paddle.set_device(DEVICE)

# num_training_samples = 10 * 16384
num_training_samples = 10 * 32
num_test_samples = 128
grid_size = 8
n_test, n_train = 32, 8
checkpoint_dir = './training_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

lr_ = 1.2e-5


def loss(model, n, A_stencil, A_matrices, S_matrices, index=None, pos=-1., phase="Training", epoch=-1, grid_size=8, remove=True):
    A_matrices = paddle.conj(A_matrices)
    S_matrices = paddle.conj(S_matrices)
    pi = np.pi
    theta_x = np.array(([i * 2 * pi / n for i in range(-n // (grid_size * 2) + 1, n // (grid_size * 2) + 1)]))
    if phase == "Test" and epoch == 0:
        P_stencil = model(A_stencil, True)
        P_matrix = utils.compute_p2LFA(P_stencil, n, grid_size)
        P_matrix = paddle.transpose(P_matrix, [2, 0, 1, 3, 4])
        P_matrix_t = paddle.transpose(P_matrix, [0, 1, 2, 4, 3]).conj()
        A_c = paddle.matmul(paddle.matmul(P_matrix_t, A_matrices), P_matrix)

        index_to_remove = len(theta_x) * (-1 + n // (2 * grid_size)) + n // (2 * grid_size) - 1
        A_c = paddle.reshape(A_c, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, (grid_size // 2) ** 2))
        A_c_removed = paddle.concat([A_c[:, :index_to_remove], A_c[:, index_to_remove + 1:]], 1)
        P_matrix_t_reshape = paddle.reshape(P_matrix_t, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, grid_size ** 2))
        P_matrix_reshape = paddle.reshape(P_matrix, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, (grid_size // 2) ** 2))
        A_matrices_reshaped = paddle.reshape(A_matrices, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
        A_matrices_removed = paddle.concat([A_matrices_reshaped[:, :index_to_remove], A_matrices_reshaped[:, index_to_remove + 1:]], 1)
        P_matrix_removed = paddle.concat([P_matrix_reshape[:, :index_to_remove], P_matrix_reshape[:, index_to_remove + 1:]], 1)
        P_matrix_t_removed = paddle.concat([P_matrix_t_reshape[:, :index_to_remove], P_matrix_t_reshape[:, index_to_remove + 1:]], 1)
        # 修复complex128不支持
        if A_c_removed.dtype in [paddle.complex64, paddle.complex128]:
            import scipy.linalg
            A_c_removed_np = A_c_removed.numpy()
            P_matrix_t_removed_np = P_matrix_t_removed.numpy()
            # A_c_removed_np: [batch, num_blocks, N, N], P_matrix_t_removed_np: [batch, num_blocks, N, M]
            A_coarse_inv_removed_np = np.stack([
                np.stack([
                    scipy.linalg.solve(A_c_removed_np[i, j], P_matrix_t_removed_np[i, j])
                    for j in range(A_c_removed_np.shape[1])
                ], axis=0)
                for i in range(A_c_removed_np.shape[0])
            ], axis=0)
            A_coarse_inv_removed = paddle.to_tensor(A_coarse_inv_removed_np)
        else:
            A_coarse_inv_removed = paddle.linalg.solve(A_c_removed, P_matrix_t_removed)
        eye_mat = paddle.eye(grid_size ** 2, dtype='float64')
        eye_mat = paddle.cast(eye_mat, 'complex128')
        CGC_removed = eye_mat - paddle.matmul(paddle.matmul(P_matrix_removed, A_coarse_inv_removed), A_matrices_removed)
        S_matrices_reshaped = paddle.reshape(S_matrices, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
        S_removed = paddle.concat([S_matrices_reshaped[:, :index_to_remove], S_matrices_reshaped[:, index_to_remove + 1:]], 1)
        iteration_matrix = paddle.matmul(paddle.matmul(CGC_removed, S_removed), S_removed)
        loss_test = paddle.mean(paddle.mean(paddle.sum(paddle.square(paddle.abs(iteration_matrix)), axis=[2, 3]), axis=1))
        return paddle.to_tensor([0.]), loss_test.numpy()
    if index is not None:
        P_stencil = model(A_stencil, index=index, pos=pos, phase=phase)
    else:
        P_stencil = model(A_stencil, phase=phase)
    if not (phase == "Test" and epoch == 0):
        P_matrix = utils.compute_p2LFA(P_stencil, n, grid_size)
        P_matrix = paddle.transpose(P_matrix, [2, 0, 1, 3, 4])
        P_matrix_t = paddle.transpose(P_matrix, [0, 1, 2, 4, 3]).conj()
        A_c = paddle.matmul(paddle.matmul(P_matrix_t, A_matrices), P_matrix)
        index_to_remove = len(theta_x) * (-1 + n // (2 * grid_size)) + n // (2 * grid_size) - 1
        A_c = paddle.reshape(A_c, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, (grid_size // 2) ** 2))
        A_c_removed = paddle.concat([A_c[:, :index_to_remove], A_c[:, index_to_remove + 1:]], 1)
        P_matrix_t_reshape = paddle.reshape(P_matrix_t, (-1, int(theta_x.shape[0]) ** 2, (grid_size // 2) ** 2, grid_size ** 2))
        P_matrix_reshape = paddle.reshape(P_matrix, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, (grid_size // 2) ** 2))
        A_matrices_reshaped = paddle.reshape(A_matrices, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
        A_matrices_removed = paddle.concat([A_matrices_reshaped[:, :index_to_remove], A_matrices_reshaped[:, index_to_remove + 1:]], 1)
        P_matrix_removed = paddle.concat([P_matrix_reshape[:, :index_to_remove], P_matrix_reshape[:, index_to_remove + 1:]], 1)
        P_matrix_t_removed = paddle.concat([P_matrix_t_reshape[:, :index_to_remove], P_matrix_t_reshape[:, index_to_remove + 1:]], 1)
        # 修复complex128不支持
        if A_c_removed.dtype in [paddle.complex64, paddle.complex128]:
            import scipy.linalg
            A_c_removed_np = A_c_removed.numpy()
            P_matrix_t_removed_np = P_matrix_t_removed.numpy()
            # A_c_removed_np: [batch, num_blocks, N, N], P_matrix_t_removed_np: [batch, num_blocks, N, M]
            A_coarse_inv_removed_np = np.stack([
                np.stack([
                    scipy.linalg.solve(A_c_removed_np[i, j], P_matrix_t_removed_np[i, j])
                    for j in range(A_c_removed_np.shape[1])
                ], axis=0)
                for i in range(A_c_removed_np.shape[0])
            ], axis=0)
            A_coarse_inv_removed = paddle.to_tensor(A_coarse_inv_removed_np)
        else:
            A_coarse_inv_removed = paddle.linalg.solve(A_c_removed, P_matrix_t_removed)
        eye_mat = paddle.eye(grid_size ** 2, dtype='float64')
        eye_mat = paddle.cast(eye_mat, 'complex128')
        CGC_removed = eye_mat - paddle.matmul(paddle.matmul(P_matrix_removed, A_coarse_inv_removed), A_matrices_removed)
        S_matrices_reshaped = paddle.reshape(S_matrices, (-1, int(theta_x.shape[0]) ** 2, grid_size ** 2, grid_size ** 2))
        S_removed = paddle.concat([S_matrices_reshaped[:, :index_to_remove], S_matrices_reshaped[:, index_to_remove + 1:]], 1)
        iteration_matrix_all = paddle.matmul(paddle.matmul(CGC_removed, S_removed), S_removed)
        if remove:
            if phase != 'Test':
                iteration_matrix = iteration_matrix_all
                for _ in range(0):
                    iteration_matrix = paddle.matmul(iteration_matrix_all, iteration_matrix_all)
            else:
                iteration_matrix = iteration_matrix_all
            loss_val = paddle.mean(paddle.max(paddle.pow(paddle.sum(paddle.square(paddle.abs(iteration_matrix)), axis=[2, 3]), 1), axis=1))
        else:
            loss_val = paddle.mean(paddle.mean(paddle.sum(paddle.square(paddle.abs(iteration_matrix_all)), axis=[2, 3]), axis=1))
            print("Real loss: ", loss_val.numpy())
        real_loss = loss_val.numpy()
        return loss_val, real_loss

def grad(model, n, A_stencil, A_matrices, S_matrices, phase="Training", epoch=-1, grid_size=8, remove=True):
    with paddle.amp.auto_cast():
        loss_value, real_loss = loss(model, n, A_stencil, A_matrices, S_matrices, phase=phase, epoch=epoch, grid_size=grid_size, remove=remove)
    loss_value.backward()
    grads = [p.grad for p in model.parameters()]
    return grads, real_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="")
    parser.add_argument('--use-gpu', action='store_true', default=True, help="")
    parser.add_argument('--grid-size', default=8, type=int, help="")
    parser.add_argument('--batch-size', default=32, type=int, help="")
    parser.add_argument('--n-epochs', default=2, type=int, help="")
    parser.add_argument('--bc', default='periodic')

    args = parser.parse_args()
    paddle.set_device(DEVICE)
    # if args.use_gpu:
    #     paddle.set_device('gpu')
    # else:
    #     paddle.set_device('cpu')

    utils = Utils(grid_size=args.grid_size, device=DEVICE, bc=args.bc)

    random_string = ''.join(random.choices(string.digits, k=5))  # to make the run_name string unique
    run_name = f"regularization_grid_size={args.grid_size}_batch_size={args.batch_size}_{random_string}"
    # writer = SummaryWriter(log_dir='runs/' + run_name)

    if args.bc == 'periodic':
        from model_periodicBC import Pnetwork
    else:
        from model_dirichletBC import Pnetwork

    m = Pnetwork(grid_size=args.grid_size)
    optimizer = paddle.optimizer.Adam(parameters=m.parameters(), learning_rate=lr_)

    pi = np.pi
    ci = 1j

    A_stencils_test, A_matrices_test, S_matrices_test, num_of_modes = utils.get_A_S_matrices(num_test_samples, np.pi, grid_size, n_test)
    A_stencils_test = paddle.to_tensor(A_stencils_test, dtype='float64')
    A_matrices_test = paddle.to_tensor(A_matrices_test, dtype='complex128')
    S_matrices_test = paddle.reshape(paddle.to_tensor(S_matrices_test, dtype='complex128'), (-1, num_of_modes, num_of_modes, grid_size ** 2, grid_size ** 2))

    A_stencils_train = np.array(utils.two_d_stencil(num_training_samples))
    n_train_list = [16, 16, 32]
    initial_epsi = 1e-0

    numiter = -1
    for j in range(len(n_train_list)):
        A_stencils = A_stencils_train.copy()
        n_train = n_train_list[j]

        theta_x = np.array([i * 2 * pi / n_train for i in range(-n_train // (2 * grid_size) + 1, n_train // (2 * grid_size) + 1)])
        theta_y = np.array([i * 2 * pi / n_train for i in range(-n_train // (2 * grid_size) + 1, n_train // (2 * grid_size) + 1)])

        for epoch in range(args.n_epochs):
            print("epoch: {}".format(epoch))
            order = np.random.permutation(num_training_samples)

            _, blackbox_test_loss = grad(model=m, n=n_test, A_stencil=A_stencils_test, A_matrices=A_matrices_test, S_matrices=S_matrices_test, phase="Test", epoch=0, grid_size=grid_size)

            if epoch % 1 == 0:  # change to save once every X epochs
                paddle.save(m.state_dict(), checkpoint_prefix + f"_model_epoch{epoch}.pdparams")
                paddle.save(optimizer.state_dict(), checkpoint_prefix + f"_optim_epoch{epoch}.pdopt")

            for iter in tqdm(range(num_training_samples // args.batch_size)):
                numiter += 1
                idx = np.random.choice(A_stencils.shape[0], args.batch_size, replace=False)
                A_matrices = np.stack([[utils.compute_A(A_stencils[idx], tx, ty, 1j, grid_size=grid_size) for tx in theta_x] for ty in theta_y])
                A_matrices = A_matrices.transpose((2, 0, 1, 3, 4))
                S_matrices = np.reshape(utils.compute_S(A_matrices.reshape((-1, grid_size ** 2, grid_size ** 2))), (-1, theta_x.shape[0], theta_x.shape[0], grid_size ** 2, grid_size ** 2))
                A_stencils_tensor = paddle.to_tensor(A_stencils[idx], dtype='float64')
                A_matrices_tensor = paddle.to_tensor(A_matrices, dtype='complex128')
                S_matrices_tensor = paddle.to_tensor(S_matrices, dtype='complex128')
                _, blackbox_train_loss = grad(m, n_train, A_stencils_tensor, A_matrices_tensor, S_matrices_tensor, epoch=0, grid_size=grid_size, remove=True, phase="Test")
                grads, real_loss = grad(m, n_train, A_stencils_tensor, A_matrices_tensor, S_matrices_tensor, grid_size=grid_size, remove=True, phase="p")
                optimizer.step()
                optimizer.clear_grad()
                # writer.add_scalar('loss', real_loss, numiter)
                # writer.add_scalar('blackbox_train_loss', blackbox_train_loss, numiter)
                # writer.add_scalar('blackbox_test_loss', blackbox_test_loss, numiter)

        # add coarse grid problems:
        if j > 0:
            num_training_samples = num_training_samples // 2
        temp = utils.create_coarse_training_set(m, pi, num_training_samples)
        A_stencils_train = np.concatenate([np.array(utils.two_d_stencil(num_training_samples)), temp], axis=0)
        num_training_samples = A_stencils_train.shape[0]
