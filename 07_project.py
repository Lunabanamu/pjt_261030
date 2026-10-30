"""1. 이전 Gradient 초기화
optimizer.zero_grad()

2. 순전파: 모델이 예측 (문제를 풀고 답을 내는 과정)
pred = model(X_train)

3. 손실 계산
loss = loss_fn(pred, y_train)

4. 역전파: Gradient 계산 (틀린 이유를 분석하고 각 가중치를 어떻게 조정할지 계산하는 과정)
loss.backward()

5. 가중치 업데이트
optimizer.step()"""


from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt
df = pd.read_csv(base / "points.csv") #csv 파일 읽기

#데이터 학습(train)과 검증(valid) 분리
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32) #입력 데이터 x1, x2
y_train = torch.tensor(train["label"].values, dtype=torch.long) #정답 데이터 y 정수형(Long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32) #검증데이터 동일
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

loss_fn = nn.CrossEntropyLoss() 
summary = [] #실험 결과 저장 리스트
#두 가지 학습률 비교
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]: 
    torch.manual_seed(42) #난수 고정
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)) #신경망 모델 생성
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = [] #epoch별 학습 및 검증 손실을 저장할 리스트
    """1. 학습모드"""
    for epoch in range(1, 1001): #학습 횟수
        model.train()  
        optimizer.zero_grad() #기울기 초기화
        loss = loss_fn(model(X_train), y_train) 

        # loss.backward() 다음 optimizer.step()을 호출하세요.
        # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
       
        # TODO B08: 위 요구사항에 맞게 이 블록만 완성합니다.
        loss.backward() #역전파 실행 
        optimizer.step() #계산된 기울기로 가중치 업데이트

        """2. 평가모드"""
        model.eval() 
        with torch.no_grad(): #평가 단계에서는 기울기 계산 비활성화
            train_loss = loss_fn(model(X_train), y_train).item() #학습 데이터 평가모드 재계산
            valid_loss = loss_fn(model(X_valid), y_valid).item() #검증 데이터 이용 검증 손실 계산

        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
        # TODO B09: 위 요구사항에 맞게 이 블록만 완성합니다.
        history.append([epoch, train_loss, valid_loss])
                        #epoch번호, 학습손실, 검증손실

    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"]) #학습 기록 DataFrame 변환
    table.to_csv(base / f"history_{name}.csv", index=False) #학습기록 CSV 저장
    best = table.loc[table["valid_loss"].idxmin()] #검증 손실이 가장 낮았던 epoch 찾기
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]]) #실험 결과 요약

    plt.figure(figsize=(8, 5)) #학습곡선 그래프
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)") #학습 손실 그래프
    plt.plot(table["epoch"], table["valid_loss"], label="valid") #검증 손실 그래프
    plt.axvline(best["epoch"], linestyle="--", label="best epoch") #검증 손실이 가장 낮았던 epoch 표시
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"curves_{name}.png", dpi=150) #그래프 이미지 저장
    plt.close()

result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"]) #결과를 표로 정리
result.to_csv(base / "experiment_table.csv", index=False) #실험 최종 csv 저장
print(result.round(6).to_string(index=False)) #결과 출력
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")

"""
20 epoch
1. 실험목적 : 학습률을 다르게 설정했을 때 차이 확인
2. 비교조건: Baseline = 0.001 / Candidate = 0.01
3. 실험결과
setting     lr        best_epoch     best_valid_loss
 baseline 0.001          20         0.480205
candidate 0.010          20         0.307332
4. 결과비교
Candidate의 loss가 Beseline 보다 낮게 나타남
학습률을 0.001에서 0.01로 높였을 때 검증 손실이 약 36% 감소


50 epoch
3. 실험결과
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001          50         0.442497
candidate 0.010          50         0.161767
4. 결과비교
Baseline (lr 0.001): 약 7.85% 감소
Candidate (lr 0.01): 약 47.38% 감소
Candidate가 더 효과적인 학습 결과인 것은 동일


100 epoch
3. 실험결과
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001         100         0.391903
candidate 0.010         100         0.031692
4. 결과비교
학습 횟수가 증가하면서 Validation Loss가 지속적으로 감소
100 Epoch에서도 최저점이 마지막에 나타났기 때문에 실제 최적 Epoch는 아직 확인되지 않음


200 epoch
3. 실험결과
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001         200         0.310185
candidate 0.010         200         0.005890
4. 결과비교
아직 과적합 여부 확정 불가

500 epoch
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001         500         0.123618
candidate 0.010         500         0.001407

1000 epoch
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001        1000         0.026759
candidate 0.010        1000         0.000510

다 좋아..?
"""