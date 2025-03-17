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
from tempfile import gettempdir
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from pathlib import Path
from matplotlib.animation import FuncAnimation

# 학습 데이터 iter로 반복하여 학습에 넣기위해 가공
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization import build_rasterizer
# 학습한 데이터 csv파일로 만들어서 성능 평가
from l5kit.evaluation import compute_metrics_csv, write_pred_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

# L5Kit 데이터 경로를 환경 변수로 설정
os.environ["L5KIT_DATA_FOLDER"] = "/media/minsu/Windows-SSD/ROKEY/LyftMotion_predicition_autonomous_vehicles/test/L5kit_agent_model/lyft-motion-prediction-autonomous-vehicles"

# LocalDataManager를 추상화된 형태로 초기화 (구체적인 경로 대신 None 사용)
dm = LocalDataManager(None)

# config.yaml 파일을 불러와 딕셔너리 형태로 저장
cfg = load_config_data("/media/minsu/Windows-SSD/ROKEY/LyftMotion_predicition_autonomous_vehicles/test/L5kit_agent_model/config.yaml")

# 데이터셋 초기화 시작
# 지도 데이터를 벡터화하기 위한 vectorizer 생성
# ChunkedDataset은 경로와 키를 통해 데이터에 접근하며, LocalDataManager는 키만 전달하는 추상화된 구조
train_zarr = ChunkedDataset(dm.require(
    # train_data_loader의 key 값인 "scenes/train.zarr"을 참조
    cfg["train_data_loader"]["key"]
    # open()은 읽기 모드로 파일을 열고, ChunkedDataset에서 정의된 패턴으로 데이터를 로드
    )).open()

'''vectorizer, train_dataset 관련 설명
빌드 벡터라이져함수 딕셔너리와 추상화 클래스를 받아서 벡터라이저형태로 반환함.

cfg에서["raster_params"]["dataset_meta_key"]를 읽음 -> raster_params:  dataset_meta_key: "meta.json" 임
meta.json파일을 읽어서 딕셔너리 형태로 저장함.
"world_to_ecef": [
        [
            0.846617444,
            0.323463078,
            -0.422623402,
            -2698767.44
        ],
        [
            -0.532201938,
            0.514559352,
            -0.672301845,
            -4293151.58
        ],
        [
            -3.05311332e-16,
            0.794103464,
            0.6077826,
            3855164.76
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]
이걸 읽어와서 넘파이 어레이로 저장함.

cfg에서["raster_params"]["semantic_map_key"]를 읽음 -> raster_params:  semantic_map_key: "semantic_map/semantic_map.pb"

가공된 world_to_ecef와 시멘틱맵경로를 MapAPI에 넣음.
대강 pb파일을 world_to_ecef가지고 활용하는 함수인거 같은데... 나중에 이부분 유의 하고 사용하면 될듯 
return 된 값이 Vectorizer(cfg, MapAPI)라서 활용 하면될듯.

Vectorizer는 
self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]
self.mapAPI = mapAPI
self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
self.history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
self.future_num_frames = cfg["model_params"]["future_num_frames"]
self.history_num_frames_max = max(cfg["model_params"]["history_num_frames_ego"], self.history_num_frames_agents)
self.other_agents_num = cfg["data_generation_params"]["other_agents_num"]
로 초기화를 하는데.....
벡터 매개 변수 전부랑 
model_params:
  history_num_frames_ego: 1 
  history_num_frames_agents: 3
  future_num_frames: 12
이렇게 3개는 self 변수로 저장함.
'''

# vectorizer 객체 생성 (cfg와 dm을 사용해 지도 데이터를 벡터화)
vectorizer = build_vectorizer(cfg, dm)

# EgoDatasetVectorized를 통해 학습 데이터셋 생성 (자율주행 차량 중심의 벡터화된 데이터)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

''' 
둘다 같은걸 출력함..
print(train_zarr)
print(train_dataset)
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
| Num Scenes | Num Frames | Num Agents | Num TR lights | Total Time (hr) | Avg Frames per Scene | Avg Agents per Frame | Avg Scene Time (sec) | Avg Frame frequency |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
|   16265    |  4039527   | 320124624  |    38735988   |      112.19     |        248.36        |        79.25         |        24.83         |        10.00        |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
'''

# AgentDataset 설정: 에이전트(다른 차량 등) 중심의 데이터셋 생성
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)  # 지도와 데이터를 래스터 이미지로 변환
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, 
                            shuffle=train_cfg["shuffle"], 
                            batch_size=train_cfg["batch_size"], 
                            num_workers=train_cfg["num_workers"])
# 데이터 내부를 확인하려 했으나 크기가 너무 커서 읽기 어려움
# for epoch in range(2):
#     print(f'----------epoch : {epoch}----------')  # 에포크 출력
#     for i, (data, target) in enumerate(train_dataloader):
#         print(target)  # 타겟 데이터 출력 (에이전트 데이터와 yaml 설정값 포함 추정)

# 모델 정의
def build_model(cfg: Dict) -> nn.Module:
    # 사전 학습된 ResNet101 모델 로드
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

    # 입력 채널 수를 config에 맞게 조정 (과거 프레임 수를 반영) (과거 프레임수+현재프레임(1))*채널수(2)
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    # 첫 컨볼루션 레이어 새로운 입력 채널 수에 맞추기위해 재정의
    # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)로 기본값 설정되어 있다고 함.
    # 채널만 바꾸고 나머진 유지했음.
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )

    # 출력 레이어 설정-> 미래 프레임 수에 맞는 좌표(x, y) 예측
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)
    return model

# 순전파 함수 정의
def forward(data, model, device, criterion): # DataLoader, model, device, loss function
    '''data["image"].to(device)
    data["image"]:
    data는 DataLoader에서 제공하는 배치 데이터(딕셔너리 형태).
    "image" 키는 지도 이미지를 나타냄(L5Kit의 AgentDataset에서 생성).
    .to(device):
    device는 torch.device("cuda:0") 또는 "cpu"로 정의됨.
    데이터를 해당 디바이스로 이동시켜 모델 연산 준비.
    '''
    inputs = data["image"].to(device)
    # target_availabilities은 유효성을 나타냄
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    # 타겟 위치 데이터 디바이스로 이동
    targets = data["target_positions"].to(device)
    ''' 모델 예측 후 타겟 형태로 재구성'
    원래 출력: (batch_size, num_targets)
    .reshape(targets.shape):
    (batch_size, 24) → (batch_size, 12, 2)로 재구성. ->12프레임의 (x, y) 좌표
    예측값을 타겟과 직접 비교 가능하도록 형태 맞춤.
    '''
    outputs = model(inputs).reshape(targets.shape)
    # 손실 계산 -> MSE 손실 함수로 일딴 되어 있음.(예제꺼 그대로 쓰는중)
    loss = criterion(outputs, targets)
    loss = loss * target_availabilities #-> 유효한 즉 사용가능한 프레임만 손실에 반영;
    loss = loss.mean()
    return loss, outputs

''' 모델 초기화
GPU 사용 가능 시 CUDA, 아니면 CPU
모델 생성 후 디바이스로 이동
Adam 옵티마이저 설정 (학습률 0.001)
MSE 손실 함수 (감소 없음)
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model = build_model(cfg).to(device)  
optimizer = optim.Adam(model.parameters(), lr=0.001)  
criterion = nn.MSELoss(reduction="none")  

# 학습 루프 시작
# iter로 반복 가능한 객체로 변환
tr_it = iter(train_dataloader)  
# 진행을 표시해서 보여주기 위함.
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
#손실 리스트 초기화
losses_train = []

for _ in progress_bar:
    try:
        #다음 데이터 가져화오기
        data = next(tr_it)
    # 반복자 멈추기 끝에 왔을때 예외 처리로 끝내려고 쓰는것.
    except StopIteration:
        # 데이터 새로이 활성화
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # 학습 모드로 전환
    model.train()  
    torch.set_grad_enabled(True)  # 그래디언트 계산 활성화
    loss, _ = forward(data, model, device, criterion)   # 순전파 손실 계산 _ : output 값은 사용하지 않음.

    # 역전파
    # 그래디언트 초기화
    optimizer.zero_grad()
    # 손실에 대한 그레디언트를 계산하고 계산된 그래디언트를 사용해 파라미터 조정
    loss.backward()
    optimizer.step()

    # 손실값 저장
    losses_train.append(loss.item())  
    # 진행상황 바
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")


# 데이터 평가 및 분석

# 평가 데이터를 잘라서 준비하기 위한 설정
# num_frames_to_chop: 각 장면(scene)을 몇 개의 프레임으로 자를지 결정 (여기서는 100프레임으로 설정)
num_frames_to_chop = 100

# 평가용 설정을 config에서 가져옴
eval_cfg = cfg["val_data_loader"]

'''평가 데이터셋 생성
dm.require(eval_cfg["key"]): LocalDataManager를 통해 "scenes/validate.zarr" 같은 평가 데이터 경로 반환
cfg["raster_params"]["filter_agents_threshold"]: 에이전트 필터링 임계값 (예: 0.5, 너무 작은 에이전트 제외)
num_frames_to_chop: 잘라낼 프레임 수 (100)
cfg["model_params"]["future_num_frames"]: 예측해야 할 미래 프레임 수 (예: 12)
MIN_FUTURE_STEPS: 최소 미래 스텝 수 (L5Kit에서 정의된 상수, 보통 50)
원본 데이터셋을 num_frames_to_chop 단위로 잘라 새로운 .zarr 파일과 마스크, 실제값(gt) CSV 생성
잘린 데이터는 예측과 비교에 적합하도록 미래 프레임과 최소 스텝 조건을 만족
출력: eval_base_path (str) - 잘린 데이터셋이 저장된 기본 디렉토리 경로'
'''
eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                                        num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)


# 데이터셋의 .zarr 파일 경로 (예: "eval_base_path/validate.zarr")
eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)

# 에이전트 필터링 마스크 파일 경로 (예: "eval_base_path/mask.npz")
# 마스크는 어떤 에이전트가 유효한지 나타내는 이진 데이터로, 불필요한 에이전트를 제외
eval_mask_path = str(Path(eval_base_path) / "mask.npz")
eval_gt_path = str(Path(eval_base_path) / "gt.csv")
eval_zarr = ChunkedDataset(eval_zarr_path).open()

# 마스크 파일 로드
# np.load로 .npz 파일을 읽고 "arr_0" 키로 이진 마스크 배열 추출
# eval_mask (np.array) - 에이전트 유효성을 나타내는 배열 (예: [True, False, True, ...])
eval_mask = np.load(eval_mask_path)["arr_0"]

'''데이터셋 초기화 및 마스크 로드
평가용 AgentDataset 생성
입력: 전체 설정 딕셔너리,평가 데이터셋, 지도와 에이전트 데이터를 래스터 이미지로 변환하는 객체, 유효한 에이전트만 포함하도록 필터링

rasterizer를 사용해 데이터를 이미지 형태로 변환하고, eval_mask로 유효하지 않은 에이전트 제외
'''
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)

'''평가용 DataLoader 생성
  - eval_dataset: 평가 데이터셋
  - shuffle=eval_cfg["shuffle"]: 섞기 여부 (보통 False)
  - batch_size=eval_cfg["batch_size"]: 배치 크기 (예: 32)
  - num_workers=eval_cfg["num_workers"]: 데이터 로딩 스레드 수 (예: 4)
'''
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)

# 평가 루프
# 모델을 평가 모드로 전환
model.eval()
# 그래디언트 계산 비활성화 : 평가 중에는 파라미터 업데이트가 필요 없으므로 계산 자원 절약
torch.set_grad_enabled(False)

# 평가 결과를 저장할 리스트 초기화
future_coords_offsets_pd = []
timestamps = []
agent_ids = []

# 평가 진행 상황 표시를 위한 진행바 설정
progress_bar = tqdm(eval_dataloader)

# 평가 루프 실행
for data in progress_bar:
    # 순전파 수행
    _, outputs = forward(data, model, device, criterion)

    agents_coords = outputs.numpy()

    # 좌표 변환을 위한 변환 행렬과 중심점 추출  에이전트 좌표계를 세계 좌표계로 변환하는 행렬 (batch_size x 3 x 3), centroids: 에이전트의 중심점 좌표 (batch_size x 2)
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()

    # 예측 좌표를 세계 좌표계로 변환하고 오프셋 계산
    #   - transform_points: 좌표를 세계 좌표계로 변환 (batch_size x future_num_frames x 2)
    #   - centroids[:, None, :2]: 중심점을 (batch_size x 1 x 2)로 확장해 빼기
    # np.array 형태로 저장됨.
    coords_offset = transform_points(agents_coords, world_from_agents) - centroids[:, None, :2]

    # 평가 결과 리스트에 추가
    future_coords_offsets_pd.append(coords_offset)
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

# CSV 저장

print("Collected data lengths:", len(timestamps), len(agent_ids), len(future_coords_offsets_pd))

try:
    # 예측 결과를 저장할 CSV 파일 경로 설정
    pred_path = "./pred.csv"

    # 리스트를 하나의 배열로 결합
    #   - timestamps_concat: 모든 타임스탬프 (total_samples,)
    #   - agent_ids_concat: 모든 에이전트 ID (total_samples,)
    #   - coords_concat: 모든 예측 좌표 (total_samples x future_num_frames x 2)
    timestamps_concat = np.concatenate(timestamps)
    agent_ids_concat = np.concatenate(agent_ids)
    coords_concat = np.concatenate(future_coords_offsets_pd)
    # 결합된 배열의 형태 확인 (디버깅용)
    # 각 배열의 shape (total_samples가 모두 동일해야 함)
    print("Concatenated shapes:", timestamps_concat.shape, agent_ids_concat.shape, coords_concat.shape)

    # 예측 결과를 CSV로 저장
    write_pred_csv(pred_path,
                   timestamps=timestamps_concat,
                   track_ids=agent_ids_concat,
                   coords=coords_concat)
    print("CSV saved at:", pred_path)

except Exception as e:
    print("Error saving CSV:", str(e))

# ===== 메트릭 계산 =====
'''CSV 평가 메트릭
  - neg_multi_log_likelihood: 예측의 로그 우도(정확도 관련) 계산, 값이 작을수록 좋음
  - time_displace: 시간별 예측 오차 계산
'''
metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])

# 계산된 메트릭 출력
for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)