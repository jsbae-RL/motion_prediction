# ResNet101을 사용해 실 데이터로 학습하려고 함 (임시 테스트 용도가 아님)
from torchvision.models.resnet import resnet101
from torchvision.models import ResNet101_Weights
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from typing import Dict
import os
import numpy as np
from tqdm import tqdm
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import compute_metrics_csv, write_pred_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

# L5Kit 데이터 경로를 환경 변수로 설정
os.environ["L5KIT_DATA_FOLDER"] = "/media/minsu/Windows-SSD/ROKEY/LyftMotion_predicition_autonomous_vehicles/test/L5kit_agent_model/lyft-motion-prediction-autonomous-vehicles"

# LocalDataManager를 추상화된 형태로 초기화
dm = LocalDataManager(None)

# config.yaml 파일을 불러와 딕셔너리 형태로 저장
cfg = load_config_data("/media/minsu/Windows-SSD/ROKEY/LyftMotion_predicition_autonomous_vehicles/test/L5kit_agent_model/config.yaml")

# 학습 데이터셋 초기화
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

# AgentDataset 설정: 에이전트 중심의 데이터셋 생성
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, 
                              shuffle=train_cfg["shuffle"], 
                              batch_size=train_cfg["batch_size"], 
                              num_workers=train_cfg["num_workers"])

# 통계 정보 확인 예시 (train_dataset의 __str__에서 제공)
'''
train_dataset의 통계 예시:
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
| Num Scenes | Num Frames | Num Agents | Num TR lights | Total Time (hr) | Avg Frames per Scene | Avg Agents per Frame | Avg Scene Time (sec) | Avg Frame frequency |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
|   16265    |  4039527   | 320124624  |    38735988   |      112.19     |        248.36        |        79.25         |        24.83         |        10.00        |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
'''

# 모델 정의 (다중 모드 적용)
'''추가 요소
num_modes 예측할 (모드)수 -> 보통 3~6개 사이로 하는데 직진, 좌, 우 정도로 생각해서 3개로 설정함
모드수 만큼 예측값을 반환하고, 각 모드에 대한 확률을 반환
'''
def build_model(cfg: Dict, num_modes: int = 3) -> nn.Module:
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    num_targets = num_modes * cfg["model_params"]["future_num_frames"] * 2 + num_modes
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)
    return model

# 다중 모드 순전파 함수 정의
def forward(data, model, device, criterion, num_modes: int = 3):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    outputs = model(inputs)
    future_num_frames = cfg["model_params"]["future_num_frames"]
    # 모델 출력의 앞부분을 여러모드의 좌표로 변환(3개의 모드, 12프레임, xy2채널)
    coords = outputs[:, :-num_modes].reshape(-1, num_modes, future_num_frames, 2)
    # 마지막 3개 값의 신뢰도 추출
    confidences = outputs[:, -num_modes:]
    # 신뢰도를 확률로 정규화(softmax)
    confidences = torch.softmax(confidences, dim=-1)
    loss = criterion(targets, coords, confidences, target_availabilities)
    # 단일모드에서는 loss, coords만 반환 하였으나 다중모드에서는 confidences도 반환
    # confidences는 (batch_size, num_modes)의 형태로 반환각모드의 신뢰도
    return loss, coords, confidences

# 다중 모드 NLL 손실 함수 정의
'''
preds: (batch_size, num_modes, future_num_frames, 2)
targets: (batch_size, future_num_frames, 2)  
confidences: (batch_size, num_modes)
availabilities: (batch_size, future_num_frames)
'''
def multi_mode_nll_loss(targets, preds, confidences, availabilities):
    # 각 모드와 실제 경로 간의 MSE 계산
    mse = torch.mean((preds - targets.unsqueeze(1)) ** 2, dim=(2, 3))
    # 가장 작은 MSE를 가진 모드 선택
    best_mode = torch.argmin(mse, dim=1)
    # 가장 작은 MSE를 가진 모드의 신뢰도 추출
    best_conf = confidences[torch.arange(len(confidences)), best_mode]
    # NLL 계산
    nll = -torch.log(best_conf + 1e-6)
    nll = nll * availabilities.mean(dim=1)
    return nll.mean()

# 모델 초기화
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg, num_modes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = multi_mode_nll_loss

# 학습 루프
tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    loss, _, _ = forward(data, model, device, criterion)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

# 평가용 데이터셋 준비
val_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)

# 평가 데이터셋을 10,000 샘플로 제한 너무 크면 진행하다가 끝나지 않고 에러 떠서 줄임.
val_dataset = torch.utils.data.Subset(val_dataset, range(10000))

# DataLoader 설정
val_dataloader = DataLoader(val_dataset,
                            shuffle=val_cfg["shuffle"],
                            batch_size=24,
                            num_workers=4,
                            pin_memory=True)

# 모델 설정
model.eval()

# 예측값 및 Ground Truth 저장을 위한 리스트
predictions = []
ground_truths = []
confidences = []
timestamps = []
track_ids = []
availabilities = []

# 평가 루프
'''
단일모드에서는 pred의 coords만 반환하였으나 다중모드에서는 confidences도 반환
availabilities는 각 프레임의 유효성을 나타내는 값
target_availabilities는 각 에이전트의 유효성을 나타내는 값
confidences는 각 모드의 신뢰도를 나타내는 값
'''
with torch.no_grad():
    for data in tqdm(val_dataloader, desc="Evaluating"):
        inputs = data["image"].to(device)
        targets = data["target_positions"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        loss, coords, confs = forward(data, model, device, criterion)
        predictions.append(coords.cpu().numpy())
        ground_truths.append(targets.cpu().numpy())
        confidences.append(confs.cpu().numpy())
        timestamps.append(data["timestamp"].numpy())
        track_ids.append(data["track_id"].numpy())
        availabilities.append(target_availabilities.cpu().numpy())

# 리스트를 넘파이 배열로 결합
predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)
confidences = np.concatenate(confidences, axis=0)
timestamps = np.concatenate(timestamps, axis=0)
track_ids = np.concatenate(track_ids, axis=0)
availabilities = np.concatenate(availabilities, axis=0)

# 모델이 예측한 좌표와 신뢰도 --> 다중경로
pred_path = "predictions.csv"
write_pred_csv(pred_path, timestamps=timestamps, track_ids=track_ids, coords=predictions, confs=confidences)
# 실제 차량의 좌표와 가용성 데이터(availabilities) -->단일 경로
gt_path = "ground_truth.csv"
write_gt_csv(gt_path, timestamps=timestamps, track_ids=track_ids, coords=ground_truths, avails=availabilities)

# 메트릭 계산
metrics = compute_metrics_csv(gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

# ADE 계산 (첫 번째 모드 기준)
ade = np.mean(np.linalg.norm(predictions[:, 0] - ground_truths, axis=-1))
print(f"Average Displacement Error (ADE, Mode 0): {ade}")