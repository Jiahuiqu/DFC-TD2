import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from tqdm import tqdm
from transformers.models.mamba.modeling_mamba import causal_conv1d_fn, selective_scan_fn
from transformers.models.mamba.modeling_mamba import MambaRMSNorm
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torch.utils.data import DataLoader
from dataset import load_data
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from text import SimpleTokenizer
from text import get_loss_clip
from text import tokenize
from text import Text_Net
from reconstruction_Lidar import rec,SharedEncoder
import psutil
from ptflops import get_model_complexity_info

#HSI+LiDAR/重建LiDAR

class HSI_CNN_extract(nn.Module):
    def __init__(self, in_channels=144, output_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, output_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x

class LiDAR_CNN_extract(nn.Module):
    def __init__(self, in_channels=1, output_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, output_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, H, W]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, H, W]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, H, W]
        return x

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
        # projection of the input hidden states
        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=self.use_bias
        )
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )

        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        self.linear_hid2 = nn.Linear(
            self.intermediate_size, 2 * self.intermediate_size, bias=True
        )

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
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
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

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
        gate == self.act(gate)
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


class HSIspe(nn.Module):
    def __init__(self, channels=144, hidden_size=64, state_size=16, kernel_size=4, intermediate_size=64,
                 time_step_rank=48):
        super(HSIspe, self).__init__()
        self.cnn_extract = HSI_CNN_extract(in_channels=channels)
        self.input_proj = nn.Conv2d(channels, hidden_size, kernel_size=1)
        self.ssm = SSM(hidden_size=hidden_size, state_size=state_size, kernel_size=kernel_size,
                       intermediate_size=intermediate_size, time_step_rank=time_step_rank)

    def forward(self, x):
        cnn_features = self.cnn_extract(x)

        x_proj = self.input_proj(x)
        ssm_features = self.ssm(x_proj)

        HSI_features = torch.cat((cnn_features, ssm_features), dim=1)

        return HSI_features


class Lispe(nn.Module):
    def __init__(self, hidden_size, state_size=16, kernel_size=4, intermediate_size=64, time_step_rank=48):
        super(Lispe, self).__init__()
        self.cnn_extract = LiDAR_CNN_extract()
        self.edge = nn.Conv2d(1, out_channels=hidden_size, kernel_size=1)
        self.input_proj = nn.Conv2d(1, out_channels=hidden_size, kernel_size=1)
        self.ssm = SSM(hidden_size=hidden_size, state_size=state_size, kernel_size=kernel_size,
                       intermediate_size=intermediate_size, time_step_rank=time_step_rank)

    def forward(self, x):
        edge_features = self.cnn_extract(x)

        x_proj = self.input_proj(x)
        ssm_features = self.ssm(x_proj)

        Lis_features = torch.cat((edge_features, ssm_features), dim=1)

        return Lis_features


class FuseMamba(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))
        self.mamba = Mamba(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        b, c, h, _ = x1.shape
        x1 = x1.reshape(b, c, -1).permute(0, 2, 1).contiguous()
        x2 = x2.reshape(b, c, -1).permute(0, 2, 1).contiguous()
        x1 = self.mamba(x1) + x1
        x2 = self.mamba(x2) + x2
        x = self.weight1 * x1 + self.weight2 * x2

        return x


class Classification(nn.Module):
    def __init__(self, num_classes, fuse_params):
        super().__init__()
        self.hsi = HSIspe()
        self.lispe = Lispe(hidden_size=64)

        self.fuse = FuseMamba()

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x1, x2):

        hsi_feat = self.hsi(x1)

        lispe_feat = self.lispe(x2)

        fused_feat1 = self.fuse(hsi_feat, lispe_feat)
        fused_feat2 = fused_feat1.permute(0, 2, 1)
        fused_feat3 = fused_feat2.mean(dim=2)
        return fused_feat3, self.classifier(fused_feat3)


class Class(nn.Module):
    def __init__(self, num_classes, fuse_params):
        super().__init__()
        self.hsi = HSIspe()
        self.lispe = Lispe(hidden_size=64)
        self.shared=SharedEncoder()
        with torch.no_grad():
            self.li_re = rec(
                device=device,
                img_size=128,
                loss_type='l1',
                dataloader=train_loader,
                testloader=test_loader,
                schedule_opt=schedule_opt,
                save_path='/houston/',
                load_path='/houston/',
                load=False,
                inner_channel=64,
                norm_groups=16,
                channel_mults=(1, 2, 2, 2),
                dropout=0,
                res_blocks=2,
                lr=1e-4,
                distributed=False
            ).to(device)
            self.li_re.load_state_dict(torch.load('/houston/rec_Lidar.pth'))
        self.fuse = FuseMamba()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x1, x2):
        hsi_feat = self.hsi(x1)
        lispe_feat = self.lispe(x2)
        shared_feat,cmd_loss=self.shared(x1,x2)
        with torch.no_grad():
            li_feat = self.li_re.encode_test_result(x2_restored=lispe_feat, condition=hsi_feat)
        re_feat = self.fuse(hsi_feat, li_feat)
        re_feat = re_feat.permute(0, 2, 1)
        re_feat = re_feat.flatten(2).mean(2)

        return re_feat, self.classifier(re_feat)

def calculate_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    OA = np.trace(cm) / np.sum(cm)
    class_accuracy = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    AA = np.mean(class_accuracy)
    kappa = cohen_kappa_score(y_true, y_pred)
    return OA * 100, AA * 100, kappa * 100

def compute_model_cost(model, name, device, input_shapes, repeat=50):
    model.to(device)
    model.eval()
    print(f"\n=== {name} Model Analysis ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f} M | Trainable params: {trainable_params/1e6:.2f} M")
    class MultiInputModel(torch.nn.Module):
        def __init__(self, model, input_shapes):
            super().__init__()
            self.model = model
            self.input_shapes = input_shapes

        def forward(self, x):
            if isinstance(x, tuple) or isinstance(x, list):
                return self.model(*x)
            else:
                return self.model(x)
    try:
        from thop import profile
        dummy_inputs = tuple(torch.randn((1,) + s).to(device) for s in input_shapes)
        flops, params = profile(model, inputs=dummy_inputs, verbose=False)

        flops_g = flops / 1e9
        params_m = params / 1e6

        print(f"FLOPs: {flops_g:.3f} GFLOPs | Params (thop): {params_m:.3f} M")
    except Exception as e:
        print("FLOPs calculation failed:", e)
        flops_g = None

    dummy_inputs = [torch.randn((1,) + s).to(device) for s in input_shapes]
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(*dummy_inputs)
    torch.cuda.synchronize()
    avg_time_ms = (time.time() - start_time) / repeat * 1000
    print(f"Average inference time: {avg_time_ms:.2f} ms per sample")

    allocated = torch.cuda.memory_allocated(device)/1024**2
    reserved = torch.cuda.memory_reserved(device)/1024**2
    print(f"GPU memory allocated: {allocated:.2f} MB | reserved: {reserved:.2f} MB")

    cpu_mem = psutil.Process().memory_info().rss / 1024**2
    print(f"CPU memory used: {cpu_mem:.2f} MB")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'FLOPs': flops if 'flops' in locals() else None,
        'inference_time_ms': avg_time_ms,
        'gpu_allocated_MB': allocated,
        'gpu_reserved_MB': reserved,
        'cpu_mem_MB': cpu_mem
    }

def get_config():
    config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "num_classes": 15,
        "batch_size": 1024,
        "context_length": 77,
        "vocab_size": 49408,
        "embed_dim": 128,
        "num_epochs": 500,
        "teacher_epochs": 100,
        "fuse_params": {"dim": 128},
        "data_path": "/datasets/houston13/",
        "text_path": "/houston/train_text.txt"
    }
    return config

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]

def build_dataloader(config):
    train_dataset, test_dataset, IGNORED_LABELS = load_data(
        dataset_name="Houston13",
        data_path=config["data_path"],
        patchsize=11,
        samples_type='num',
        train_size=0.8,
        num_samples_per_class=40
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    return train_loader, test_loader

def build_text_encoder(config):
    model = Text_Net(
        embed_dim=config["embed_dim"],
        context_length=config["context_length"],
        vocab_size=config["vocab_size"],
        transformer_width=128,
        transformer_heads=8,
        transformer_layers=12
    )
    return model.to(config["device"])

def build_teacher(config):
    model = Classification(
        num_classes=config["num_classes"],
        fuse_params=config["fuse_params"]
    )
    return model.to(config["device"])

def build_student(config):
    model = Class(
        num_classes=config["num_classes"],
        fuse_params=config["fuse_params"]
    )
    return model.to(config["device"])
def teacher_train(text_encoder,texts,train_loader,model1,optimizer1,criterion,device):

    model1.train()
    text_encoder.eval()

    total = 0
    correct = 0
    running_loss = 0

    all_preds = []
    all_labels = []

    tokens_tensor = tokenize(
        texts,
        context_length=77,
        truncate=True
    ).to(device)

    with torch.no_grad():
        text_features = text_encoder(tokens_tensor)

    for batch_idx, ((x1, x2), labels, indices) in enumerate(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        optimizer1.zero_grad()
        image_features, outputs = model1(x1, x2)
        loss_clip = get_loss_clip(40, image_features.shape[1],labels,image_features,text_features) * 0.005
        loss_ce = criterion(outputs, labels)
        loss = loss_ce + loss_clip

        loss.backward()
        optimizer1.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    OA, AA, kappa = calculate_metrics(
        all_labels,
        all_preds,
        outputs.size(1)
    )
    print(f"Teacher Train | "f"Loss:{running_loss/len(train_loader):.4f} "f"| OA:{OA:.2f} "f"| AA:{AA:.2f} "f"| Kappa:{kappa:.2f}")

def teacher_test(test_loader, model1, device):
    model1.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (x1, x2), labels, indices in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            _, outputs = model1(x1, x2)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    OA, AA, kappa = calculate_metrics(
        all_labels,
        all_preds,
        outputs.size(1)
    )

    print(f"Teacher Test | OA:{OA:.2f} "f"| AA:{AA:.2f} "f"| Kappa:{kappa:.2f}")

def student_train(train_loader,model1,model2,optimizer2,device,alpha=0.7,beta=0.01,T=5):

    model1.eval()
    model2.train()

    all_preds = []
    all_labels = []

    running_loss = 0

    for ((x1, x2), labels, indices) in tqdm(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)

        optimizer2.zero_grad()

        s_feature, s_outputs = model2(x1, x2)

        with torch.no_grad():
            t_feature, t_outputs = model1(x1, x2)

        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(s_outputs / T, dim=1),
            torch.softmax(t_outputs / T, dim=1)
        ) * (T * T)

        loss_feat = nn.functional.mse_loss(
            s_feature,
            t_feature.detach()
        )

        loss_ce = nn.functional.cross_entropy(
            s_outputs,
            labels
        )

        loss = (
                alpha * loss_kd +
                (1 - alpha) * loss_ce +
                beta * loss_feat
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model2.parameters(),
            max_norm=5.0
        )
        optimizer2.step()
        running_loss += loss.item()
        _, predicted = s_outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    OA, AA, kappa = calculate_metrics(
        all_labels,
        all_preds,
        s_outputs.size(1)
    )

    print(f"Student Train | "f"Loss:{running_loss/len(train_loader):.4f} "f"| OA:{OA:.2f} "f"| AA:{AA:.2f} "f"| Kappa:{kappa:.2f}")

def student_test(test_loader, model2, device):
    model2.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for ((x1, x2), labels, indices) in test_loader:

            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            _, outputs = model2(x1, x2)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    OA, AA, kappa = calculate_metrics(
        all_labels,
        all_preds,
        outputs.size(1)
    )

    print(f"Student Test | "f"OA:{OA:.2f} "f"| AA:{AA:.2f} "f"| Kappa:{kappa:.2f}")

def main():
    config = get_config()
    device = config["device"]
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = build_dataloader(config)

    texts = read_text_file(config["text_path"])

    text_encoder = build_text_encoder(config)

    print("Start Training Teacher...")
    model1 = build_teacher(config)
    optimizer1 = optim.Adam(model1.parameters(),lr=1e-3,weight_decay=1e-4)

    for epoch in range(config["teacher_epochs"]):
        print(
            f"\nTeacher Epoch "
            f"[{epoch+1}/{config['teacher_epochs']}]"
        )

        teacher_train(text_encoder,texts,train_loader,model1,optimizer1,criterion,device)

    torch.save(model1.state_dict(),"Teacher_LiDAR_model.pth")
    print("Teacher Saved.")

    model1.load_state_dict(torch.load("Teacher_LiDAR_model.pth",map_location=device))

    model1.eval()

    for param in model1.parameters():
        param.requires_grad = False

    print("\nStart Training Student...")

    model2 = build_student(config)

    optimizer2 = optim.Adam(model2.parameters(),lr=1e-3,weight_decay=1e-4)

    for epoch in range(config["num_epochs"]):
        print(
            f"\nStudent Epoch "
            f"[{epoch+1}/{config['num_epochs']}]"
        )

        student_train(train_loader,model1,model2,optimizer2,device)
        if (epoch + 1) % 100 == 0:
            print("\nTesting Student...")
            student_test(test_loader,model2,device)


if __name__ == "__main__":
    main()