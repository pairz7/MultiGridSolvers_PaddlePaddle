import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Pnetwork(nn.Layer):
    def __init__(self, grid_size=8):
        super(Pnetwork, self).__init__()
        self.grid_size = grid_size
        width = 100
        self.linear0 = nn.Linear(9 * 5, width, bias_attr=False)
        self.num_layers = 100
        for i in range(1, self.num_layers):
            setattr(self, f"linear{i}", nn.Linear(width, width, bias_attr=False,
                weight_attr=nn.initializer.TruncatedNormal(std=np.sqrt(2. / width) * i ** (-1 / 2))))
            setattr(self, f"bias_1{i}", self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0)))
            setattr(self, f"linear{i+1}", nn.Linear(width, width, bias_attr=False,
                weight_attr=nn.initializer.Constant(0.0)))
            setattr(self, f"bias_2{i}", self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0)))
            setattr(self, f"bias_3{i}", self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0)))
            setattr(self, f"bias_4{i}", self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0)))
            setattr(self, f"multiplier_{i}", self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(1.0)))
        self.output_layer = nn.Linear(width, 4, bias_attr=True)
        self.new_output = self.create_parameter(
            shape=[2*2*2*8], default_initializer=nn.initializer.Normal(0.5, 1.0))

    def forward(self, inputs, black_box=False, index=None, pos=-1., phase='Training'):
        inputs = paddle.cast(inputs, 'float32')
        batch_size = inputs.shape[0]
        # 方向索引
        right_idx1 = paddle.to_tensor([i for i in range(1, self.grid_size, 2)], dtype='int64')
        right_idx2 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')
        left_idx1 = paddle.to_tensor([(i-1) % self.grid_size for i in range(0, self.grid_size, 2)], dtype='int64')
        left_idx2 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')
        up_idx1 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')
        up_idx2 = paddle.to_tensor([i for i in range(1, self.grid_size, 2)], dtype='int64')
        down_idx1 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')
        down_idx2 = left_idx1
        center_idx1 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')
        center_idx2 = paddle.to_tensor([i for i in range(0, self.grid_size, 2)], dtype='int64')

        right_contributions_input = paddle.index_select(inputs, right_idx1, axis=1)
        right_contributions_input = paddle.index_select(right_contributions_input, right_idx2, axis=2)
        left_contributions_input = paddle.index_select(inputs, left_idx1, axis=1)
        left_contributions_input = paddle.index_select(left_contributions_input, left_idx2, axis=2)
        left_contributions_input = paddle.reshape(left_contributions_input, (-1, self.grid_size//2, self.grid_size//2, 3, 3))
        up_contributions_input = paddle.index_select(inputs, up_idx1, axis=1)
        up_contributions_input = paddle.index_select(up_contributions_input, up_idx2, axis=2)
        up_contributions_input = paddle.reshape(up_contributions_input, (-1, self.grid_size//2, self.grid_size//2, 3, 3))
        down_contributions_input = paddle.index_select(inputs, down_idx1, axis=1)
        down_contributions_input = paddle.index_select(down_contributions_input, down_idx2, axis=2)
        down_contributions_input = paddle.reshape(down_contributions_input, (-1, self.grid_size//2, self.grid_size//2, 3, 3))
        center_contributions_input = paddle.index_select(inputs, center_idx1, axis=1)
        center_contributions_input = paddle.index_select(center_contributions_input, center_idx2, axis=2)
        center_contributions_input = paddle.reshape(center_contributions_input, (-1, self.grid_size//2, self.grid_size//2, 3, 3))

        inputs_combined = paddle.concat([
            right_contributions_input, left_contributions_input,
            up_contributions_input, down_contributions_input,
            center_contributions_input
        ], axis=0)
        flattended = paddle.reshape(inputs_combined, (-1, 9))
        flattended = paddle.cast(flattended, 'float32')
        temp = (self.grid_size//2)**2
        flattended = paddle.concat([
            flattended[:batch_size*temp],
            flattended[temp*batch_size:temp*2*batch_size],
            flattended[temp*2*batch_size:temp*3*batch_size],
            flattended[temp*3*batch_size:temp*4*batch_size],
            flattended[temp*4*batch_size:]
        ], axis=-1)
        x = self.linear0(flattended)
        x = F.relu(x)
        for i in range(1, self.num_layers, 2):
            x1 = getattr(self, f"bias_1{i}") + x
            x1 = getattr(self, f"linear{i}")(x1)
            x1 = x1 + getattr(self, f"bias_2{i}") + x1
            x1 = F.relu(x1)
            x1 = x1 + getattr(self, f"bias_3{i}") + x1
            x1 = getattr(self, f"linear{i+1}")(x1)
            x1 = paddle.multiply(x1, getattr(self, f"multiplier_{i}"))
            x = x + x1
            x = x + getattr(self, f"bias_4{i}")
            x = F.relu(x)
        x = self.output_layer(x)
        if index is not None:
            indices = paddle.to_tensor([[index]], dtype='int64')
            updates = paddle.to_tensor([float(pos)], dtype='float64')
            shape = [2*2*2*8]
            scatter = paddle.scatter_nd(indices, updates, shape)
            x = self.new_output + paddle.reshape(scatter, (-1, 2, 2, 8))
            ld_contribution = x[:, :, :, 0]
            left_contributions_output = x[:, :, :, 1]
            lu_contribution = x[:, :, :, 2]
            down_contributions_output = x[:, :, :, 3]
            up_contributions_output = x[:, :, :, 4]
            ones = paddle.ones_like(up_contributions_output)
            right_contributions_output = x[:, :, :, 6]
            rd_contribution = x[:, :, :, 5]
            ru_contribution = x[:, :, :, 7]
            first_row = paddle.concat([
                paddle.unsqueeze(ld_contribution, -1),
                paddle.unsqueeze(left_contributions_output, -1),
                paddle.unsqueeze(lu_contribution, -1)
            ], -1)
            second_row = paddle.concat([
                paddle.unsqueeze(down_contributions_output, -1),
                paddle.unsqueeze(ones, -1),
                paddle.unsqueeze(up_contributions_output, -1)
            ], -1)
            third_row = paddle.concat([
                paddle.unsqueeze(rd_contribution, -1),
                paddle.unsqueeze(right_contributions_output, -1),
                paddle.unsqueeze(ru_contribution, -1)
            ], -1)
            output = paddle.stack([first_row, second_row, third_row], axis=0)
            output = paddle.transpose(output, [1, 2, 3, 0, 4])
            if not black_box:
                return paddle.cast(output, 'complex128')
        else:
            x = paddle.reshape(x, (-1, self.grid_size//2, self.grid_size//2, 4))
        if black_box:
            up_contributions_output = paddle.index_select(inputs, up_idx1, axis=1)
            up_contributions_output = paddle.index_select(up_contributions_output, up_idx2, axis=2)
            up_contributions_output = -paddle.sum(up_contributions_output[:, :, :, :, 0], axis=-1) / paddle.sum(up_contributions_output[:, :, :, :, 1], axis=-1)
            left_contributions_output = paddle.index_select(inputs, left_idx1, axis=1)
            left_contributions_output = paddle.index_select(left_contributions_output, left_idx2, axis=2)
            left_contributions_output = -paddle.sum(left_contributions_output[:, :, :, 2, :], axis=-1) / paddle.sum(left_contributions_output[:, :, :, 1, :], axis=-1)
            right_contributions_output = paddle.index_select(inputs, right_idx1, axis=1)
            right_contributions_output = paddle.index_select(right_contributions_output, right_idx2, axis=2)
            right_contributions_output = -paddle.sum(right_contributions_output[:, :, :, 0, :], axis=-1) / paddle.sum(right_contributions_output[:, :, :, 1, :], axis=-1)
            down_contributions_output = paddle.index_select(inputs, down_idx1, axis=1)
            down_contributions_output = paddle.index_select(down_contributions_output, down_idx2, axis=2)
            down_contributions_output = -paddle.sum(down_contributions_output[:, :, :, :, 2], axis=-1) / paddle.sum(down_contributions_output[:, :, :, :, 1], axis=-1)
        else:
            jm1 = [(i - 1) % (self.grid_size // 2) for i in range(self.grid_size // 2)]
            jp1 = [(i + 1) % (self.grid_size // 2) for i in range(self.grid_size // 2)]
            right_contributions_output = x[:, :, :, 0] / (paddle.index_select(x[:, :, :, 1], paddle.to_tensor(jp1, dtype='int64'), axis=1) + x[:, :, :, 0])
            left_contributions_output = x[:, :, :, 1] / (x[:, :, :, 1] + paddle.index_select(x[:, :, :, 0], paddle.to_tensor(jm1, dtype='int64'), axis=1))
            up_contributions_output = x[:, :, :, 2] / (x[:, :, :, 2] + paddle.index_select(x[:, :, :, 3], paddle.to_tensor(jp1, dtype='int64'), axis=2))
            down_contributions_output = x[:, :, :, 3] / (paddle.index_select(x[:, :, :, 2], paddle.to_tensor(jm1, dtype='int64'), axis=2) + x[:, :, :, 3])
        ones = paddle.ones_like(down_contributions_output)
        # 斜对角方向
        up_right_contribution = paddle.index_select(inputs, right_idx1, axis=1)
        up_right_contribution = paddle.index_select(up_right_contribution, right_idx2, axis=2)
        up_right_contribution = up_right_contribution[:, :, :, 0, 1]
        right_up_contirbution = paddle.index_select(inputs, right_idx1, axis=1)
        right_up_contirbution = paddle.index_select(right_up_contirbution, right_idx2, axis=2)
        right_up_contirbution_additional_term = right_up_contirbution[:, :, :, 0, 0]
        right_up_contirbution = right_up_contirbution[:, :, :, 1, 0]
        ru_center_ = paddle.index_select(inputs, right_idx1, axis=1)
        ru_center_ = paddle.index_select(ru_center_, right_idx2, axis=2)
        ru_center_ = ru_center_[:, :, :, 1, 1]
        ru_contribution = -paddle.unsqueeze((right_up_contirbution_additional_term +
            paddle.multiply(right_up_contirbution, right_contributions_output) +
            paddle.multiply(up_right_contribution, up_contributions_output)) / ru_center_, -1)
        up_left_contribution = paddle.index_select(inputs, left_idx1, axis=1)
        up_left_contribution = paddle.index_select(up_left_contribution, right_idx2, axis=2)
        up_left_contribution = up_left_contribution[:, :, :, 2, 1]
        left_up_contirbution = paddle.index_select(inputs, left_idx1, axis=1)
        left_up_contirbution = paddle.index_select(left_up_contirbution, right_idx2, axis=2)
        left_up_contirbution_addtional_term = left_up_contirbution[:, :, :, 2, 0]
        left_up_contirbution = left_up_contirbution[:, :, :, 1, 0]
        lu_center_ = paddle.index_select(inputs, left_idx1, axis=1)
        lu_center_ = paddle.index_select(lu_center_, right_idx2, axis=2)
        lu_center_ = lu_center_[:, :, :, 1, 1]
        lu_contribution = -paddle.unsqueeze((left_up_contirbution_addtional_term +
            paddle.multiply(up_left_contribution, up_contributions_output) +
            paddle.multiply(left_up_contirbution, left_contributions_output)) / lu_center_, -1)
        down_left_contribution = paddle.index_select(inputs, left_idx1, axis=1)
        down_left_contribution = paddle.index_select(down_left_contribution, left_idx2, axis=2)
        down_left_contribution = down_left_contribution[:, :, :, 2, 1]
        left_down_contirbution = paddle.index_select(inputs, left_idx1, axis=1)
        left_down_contirbution = paddle.index_select(left_down_contirbution, left_idx2, axis=2)
        left_down_contirbution_additional_term = left_down_contirbution[:, :, :, 2, 2]
        left_down_contirbution = left_down_contirbution[:, :, :, 1, 2]
        ld_center_ = paddle.index_select(inputs, left_idx1, axis=1)
        ld_center_ = paddle.index_select(ld_center_, left_idx2, axis=2)
        ld_center_ = ld_center_[:, :, :, 1, 1]
        ld_contribution = -paddle.unsqueeze((left_down_contirbution_additional_term +
            paddle.multiply(down_left_contribution, down_contributions_output) +
            paddle.multiply(left_down_contirbution, left_contributions_output)) / ld_center_, -1)
        down_right_contribution = paddle.index_select(inputs, right_idx1, axis=1)
        down_right_contribution = paddle.index_select(down_right_contribution, left_idx2, axis=2)
        down_right_contribution = down_right_contribution[:, :, :, 0, 1]
        right_down_contirbution = paddle.index_select(inputs, right_idx1, axis=1)
        right_down_contirbution = paddle.index_select(right_down_contirbution, left_idx2, axis=2)
        right_down_contirbution_addtional_term = right_down_contirbution[:, :, :, 0, 2]
        right_down_contirbution = right_down_contirbution[:, :, :, 1, 2]
        rd_center_ = paddle.index_select(inputs, right_idx1, axis=1)
        rd_center_ = paddle.index_select(rd_center_, left_idx2, axis=2)
        rd_center_ = rd_center_[:, :, :, 1, 1]
        rd_contribution = -paddle.unsqueeze((right_down_contirbution_addtional_term +
            paddle.multiply(down_right_contribution, down_contributions_output) +
            paddle.multiply(right_down_contirbution, right_contributions_output)) / rd_center_, -1)
        first_row = paddle.concat([
            ld_contribution, paddle.unsqueeze(left_contributions_output, -1),
            lu_contribution
        ], -1)
        second_row = paddle.concat([
            paddle.unsqueeze(down_contributions_output, -1),
            paddle.unsqueeze(ones, -1),
            paddle.unsqueeze(up_contributions_output, -1)
        ], -1)
        third_row = paddle.concat([
            rd_contribution, paddle.unsqueeze(right_contributions_output, -1),
            ru_contribution
        ], -1)
        output = paddle.stack([first_row, second_row, third_row], axis=0)
        output = paddle.transpose(output, [1, 2, 3, 0, 4])
        return paddle.cast(output, 'complex128')