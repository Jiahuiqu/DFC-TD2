import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import nn
from transformers.models.mamba.modeling_mamba import MambaRMSNorm
from transformers.models.mamba.modeling_mamba import causal_conv1d_fn, selective_scan_fn
import torch.nn.functional as F

#sobel边缘特征增强
class EdgeExtract(nn.Module):
    def __init__(self, input_height=11, input_width=11):
        super(EdgeExtract, self).__init__()

        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.sobel_x.weight = nn.Parameter(sobel_x_kernel, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel, requires_grad=False)


    def forward(self, x):

        sobel_x_result = self.sobel_x(x)
        sobel_y_result = self.sobel_y(x)

        edge_feat = torch.sqrt(sobel_x_result ** 2 + sobel_y_result ** 2)

        return edge_feat

class pathSSM(nn.Module):

    def __init__(
        self, hidden_size, state_size, kernel_size, intermediate_size, time_step_rank
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.use_bias = False
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.act = nn.SiLU()
        self.activation = "silu"

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=self.use_bias
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )


        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        self.linear_hid2 = nn.Linear(
            self.intermediate_size, 2 * self.intermediate_size, bias=True
        )


        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, gate):

        hidden_states = hidden_states.transpose(1, 2)


        hidden_states = self.in_proj(hidden_states).transpose(1, 2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        hidden_states = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
        )

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = discrete_time_step.transpose(1, 2)

        A = -torch.exp(self.A_log.float())

        time_proj_bias = (
            self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        )
        scan_outputs, ssm_state = selective_scan_fn(
            hidden_states,
            discrete_time_step,
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            self.D.float(),
            gate,  # None,
            time_proj_bias,
            delta_softplus=True,
            return_last_state=True,
        )

        return scan_outputs

    def forward(self, hidden_states, gate):
        return self.cuda_kernels_forward(hidden_states, gate)


class SSM(nn.Module):

    def __init__(
        self,
        hidden_size,
        state_size=16,
        kernel_size=4,
        intermediate_size=64,
        time_step_rank=48,
    ):
        super(SSM, self).__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = kernel_size
        self.intermediate_size = intermediate_size
        self.time_step_rank = time_step_rank
        self.use_conv_bias = False
        self.use_bias = False
        self.norm = MambaRMSNorm(hidden_size, eps=1e-5)
        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=self.use_bias
        )
        self.act = nn.SiLU()
        self.path = pathSSM(
            hidden_size, state_size, kernel_size, intermediate_size, time_step_rank
        )
        self.path_back = pathSSM(
            hidden_size, state_size, kernel_size, intermediate_size, time_step_rank
        )

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=self.use_bias
        )
        self.out_LN = nn.LayerNorm(self.intermediate_size)

    def cuda_kernels_forward(self, hidden_states: torch.Tensor):
        b, c, h, w = hidden_states.shape
        gate = hidden_states.flatten(2).transpose(1, 2)
        gate = self.in_proj(gate)
        gate = self.act(gate)

        gate = gate.transpose(1, 2)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_back = hidden_states.flip(-1)
        gate_back = gate.flip(-1)

        scan_outputs = self.path(hidden_states, gate)
        scan_outputs_back = self.path_back(hidden_states_back, gate_back)
        scan_outputs = scan_outputs + scan_outputs_back.flip(-1)

        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        hidden_states = self.norm(contextualized_states).transpose(1, 2)
        hidden_states = hidden_states.view(b, c, h, w)

        return hidden_states

    def forward(self, hidden_states):
        return self.cuda_kernels_forward(hidden_states)


class Lispe(nn.Module):
    def __init__(self, hidden_size, state_size=16, kernel_size=4, intermediate_size=64, time_step_rank=48):
        super(Lispe, self).__init__()
        self.edge_extract = EdgeExtract()
        self.edge = nn.Conv2d(1, out_channels=hidden_size, kernel_size=1)
        self.input_proj = nn.Conv2d(1,out_channels = hidden_size, kernel_size=1)
        self.ssm = SSM(hidden_size=hidden_size, state_size=state_size, kernel_size=kernel_size,
                       intermediate_size=intermediate_size, time_step_rank=time_step_rank)

    def forward(self, x):
        edge_features = self.edge_extract(x)
        edge_features = self.edge(edge_features)
        x_proj = self.input_proj(x)
        ssm_features = self.ssm(x_proj)
        Lis_features = torch.cat((edge_features, ssm_features), dim=1)

        return Lis_features
