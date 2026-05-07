# 🌧️ Rain2FloodNet

> **Pix2Pix + FiLM 기반 조건부 생성 모델을 활용한  
> 실시간 도시 침수흔적도 예측 프로젝트**

---

## 📌 Project Overview

최근 이상기후와 도시화로 인해 도심 침수 피해가 지속적으로 증가하고 있습니다.  
기존의 물리 기반 침수 예측 모델은 높은 계산 비용과 복잡한 시뮬레이션 과정으로 인해 실시간 대응에 한계가 존재합니다.

**Rain2FloodNet**은 이러한 문제를 해결하기 위해, 실제 침수흔적도 데이터와 AWS 강수량 데이터를 활용하여 강수 조건에 따른 침수흔적도를 생성하는 **Conditional GAN 기반 침수 예측 모델**입니다.

본 프로젝트에서는 단순 cGAN 구조를 넘어, **Pix2Pix 기반 U-Net Generator**에 다양한 조건 반영 기법을 추가하여 강우 조건 변화에 대한 반응성과 생성 안정성을 개선했습니다.

특히 다음과 같은 모듈들을 직접 구현 및 실험했습니다.

- FiLM (Feature-wise Linear Modulation)
- Condition Embedder
- Minibatch Discrimination
- Condition Regressor
- WGAN-GP Loss
- White Test 기반 일반화 평가

---

## 🏗️ Model Architecture

### ✅ Baseline: Conditional GAN (cGAN)

비교 모델로 기본 **Conditional GAN(cGAN)** 구조를 구현했습니다.

Generator는 다음 정보를 입력받아 침수흔적도 이미지를 생성합니다.

- 강수량 조건
- 공간 정보
- 랜덤 노이즈 벡터

Discriminator는 다음 정보를 함께 입력받아 이미지의 진위를 판별합니다.

- 실제 침수흔적도 이미지
- 생성된 침수 예측 이미지
- 조건 벡터

학습 안정성을 높이기 위해 다음 손실을 함께 적용했습니다.

- WGAN-GP Loss
- L1 Reconstruction Loss

---

### ✅ Proposed Model: Rain2FloodNet

**Rain2FloodNet**은 Pix2Pix 기반 U-Net 구조를 확장한 조건부 이미지 생성 모델입니다.

기존 Pix2Pix 구조에 조건 정보를 보다 효과적으로 반영하기 위해 다음 모듈들을 추가했습니다.

---

#### 🔹 Condition Embedder

Condition Embedder는 입력 강수량 조건을 고차원 임베딩 벡터로 변환합니다.

변환된 조건 임베딩은 Generator 내부의 encoder 및 decoder feature map에 전달되어, 모델이 강수 조건에 따른 침수 양상 변화를 학습할 수 있도록 합니다.

---

#### 🔹 FiLM (Feature-wise Linear Modulation)

FiLM은 임베딩된 조건 정보를 feature map에 직접 반영하는 모듈입니다.

조건 임베딩을 기반으로 각 feature map에 대해 다음 두 값을 생성합니다.

- Scaling factor: γ
- Shifting factor: β

이를 통해 Generator 내부 feature map을 조건에 따라 조절합니다.

단순히 조건 벡터를 입력 이미지나 latent vector에 붙이는 방식보다, 네트워크의 중간 표현에 조건 정보를 직접 주입할 수 있어 강우 조건 변화에 대한 반응성을 높일 수 있습니다.

---

#### 🔹 Minibatch Discrimination

Discriminator 내부에 **Minibatch Discrimination(MBD)** 모듈을 추가했습니다.

이 모듈은 미니배치 내 생성 결과들의 유사성을 함께 고려하여 판별하도록 설계되었습니다.

이를 통해 다음 효과를 기대했습니다.

- 생성 결과 다양성 확보
- Mode Collapse 완화
- 침수 패턴 판별력 향상

---

#### 🔹 Condition Regressor

Condition Regressor는 생성된 이미지로부터 입력 강수 조건을 다시 추정하는 모듈입니다.

생성 이미지가 실제 입력 조건을 충분히 반영하고 있는지 학습 과정에서 직접적으로 유도하기 위해 사용했습니다.

이를 통해 다음 효과를 기대했습니다.

- 조건 정보 반영력 향상
- 강수량 변화에 따른 생성 결과의 민감도 개선
- 조건과 생성 이미지 간의 일관성 강화

---

#### 🔹 Loss Function

최종 학습은 다음 손실 함수들을 결합하여 수행했습니다.

```python
Generator Loss =
Adversarial Loss
+ L1 Reconstruction Loss
+ Condition Regression Loss
```

Discriminator 학습에는 WGAN-GP를 적용하여 학습 안정성을 높였습니다.

---

## 📊 Dataset

### 📍 Data Sources

본 프로젝트에서는 침수 예측 모델 학습을 위해 다음 데이터를 활용했습니다.

#### AWS Rainfall Data

- 1시간 최대 강수량
- 3시간 최대 강수량
- 5시간 최대 강수량

#### Flood Footprint Data

- 실제 침수흔적도 이미지
- 침수심 정보
- 침수 범위 정보

#### Map Data

- 지도 이미지
- 공간 단위 분해번호 기반 매칭 데이터

---

## 🌧️ Condition Vector

강수 조건은 1시간, 3시간, 5시간 최대 강수량을 기반으로 구성했습니다.

```python
condition = 0.5 * max_1h + 0.3 * max_3h + 0.2 * max_5h
```

단기 집중호우의 영향을 크게 반영하면서도, 중장기 강수의 영향을 함께 고려하기 위해 가중합 기반 복합 조건 벡터를 사용했습니다.

---

## 🧪 Data Split Strategy

단순 랜덤 분할이 아닌, 강수량 기반 **Case Split 전략**을 직접 설계했습니다.

테스트 데이터는 강수 조건의 분포를 고려하여 다음 케이스로 구성했습니다.

| Case | Description |
|---|---|
| Median | 중간 강수 조건 |
| Min | 최소 강수 조건 |
| Max | 최대 강수 조건 |
| Random | 무작위 강수 조건 |

이를 통해 특정 강수 조건에 편향되지 않고, 다양한 조건에서 모델의 일반화 성능을 평가할 수 있도록 했습니다.

---

## 🤍 White Test

모델의 일반화 성능을 평가하기 위해 **White Test Set**을 별도로 구성했습니다.

White Test는 특정 강수 조건에서 침수가 발생하지 않는 상황을 가정하고, 흰색 배경 이미지를 타깃으로 사용하는 평가 방식입니다.

각 공간 단위별로 다음 강수 조건을 적용했습니다.

- 0mm
- 5mm
- 10mm
- 15mm
- 20mm

이를 통해 모델이 단순히 학습 데이터의 침수 패턴을 복사하는 것이 아니라, 조건 변화에 따라 비침수 상황도 적절히 반영할 수 있는지 확인했습니다.

---

## 📈 Evaluation Metrics

모델 성능은 이미지 품질과 침수 영역 예측 성능을 함께 고려하여 평가했습니다.

| Metric | Description |
|---|---|
| PSNR | 생성 이미지와 실제 이미지 간 픽셀 단위 복원 품질 |
| SSIM | 구조적 유사도 기반 이미지 품질 |
| mIoU | 전체 영역 기준 평균 IoU |
| fg-mIoU | 침수 전경 영역 기준 평균 IoU |
| Grade-wise IoU | 침수 등급별 IoU |

특히 실제 침수 영역은 전체 이미지에서 차지하는 비율이 작기 때문에, PSNR과 SSIM뿐만 아니라 fg-mIoU와 Grade-wise IoU를 함께 확인했습니다.

---

## 🏆 Experimental Results

| Model | PSNR | SSIM | fg-mIoU | Best IoU-Grade |
|---|---:|---:|---:|---:|
| Rain2FloodNet | 25.72 | 0.9759 | 0.0124 | 0.1662 |
| cGAN | 25.80 | 0.9869 | 0.0003 | 0.0033 |

실험 결과, PSNR과 SSIM은 두 모델 모두 높은 수치를 보였습니다.  
이는 전체 이미지에서 비침수 배경 영역이 큰 비중을 차지하기 때문입니다.

반면 실제 침수 영역 예측 성능을 나타내는 fg-mIoU와 Best IoU-Grade에서는 Rain2FloodNet이 기존 cGAN 대비 더 높은 성능을 보였습니다.

이를 통해 제안 모델이 단순한 이미지 복원뿐만 아니라, 강수 조건에 따른 침수 영역 및 침수 등급 예측에서도 더 나은 조건 반영 성능을 가진다는 것을 확인했습니다.

---

## 🛠️ Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| Framework | PyTorch |
| Data Processing | NumPy, Pandas |
| Image Processing | PIL, scikit-image |
| Visualization | Matplotlib |
| Training Method | Pix2Pix, WGAN-GP |
| Model Type | Conditional GAN |

---

## ⚙️ Key Features

- Conditional Flood Map Generation
- Pix2Pix 기반 Image-to-Image Translation
- FiLM 기반 조건 임베딩
- WGAN-GP 기반 학습 안정화
- Condition Regressor 기반 조건 반영성 강화
- Minibatch Discrimination 기반 생성 다양성 개선
- White Test 기반 일반화 성능 평가
- Case-based Test Split
- Grade-wise IoU Evaluation
- Automatic Visualization Pipeline
- Logging & Metric Tracking

---

## 📂 Project Structure

```bash
Rain2FloodNet/
│
├── model/
│   ├── generator.py
│   ├── discriminator.py
│   └── condition_regressor.py
│
├── dataset/
│   ├── rainfall_csv/
│   ├── flood_images/
│   └── map_images/
│
├── train.py
├── evaluation.py
├── visualization.py
├── metrics.py
├── README.md
└── requirements.txt
```

> 실제 공개 레포지토리에서는 데이터 보안 문제로 `dataset/` 내부 원본 데이터는 포함하지 않습니다.

---

## 🚨 Data & Weight Policy

본 프로젝트는 실제 침수흔적도 데이터를 활용한 연구 프로젝트입니다.  
따라서 데이터 보안 및 원본 자료 보호를 위해 다음 항목은 공개하지 않습니다.

- Raw Dataset
- Flood Footprint Images
- Rainfall CSV Data
- Trained Weights (`.pth`, `.pt`, `.ckpt`)
- Checkpoint Files
- Original QGIS / GIS Source Files

공개 레포지토리에는 모델 구조, 학습 파이프라인, 평가 코드, 실험 설계 내용을 중심으로 정리했습니다.

---

## 📚 Research Keywords

- Conditional GAN
- Pix2Pix
- FiLM
- WGAN-GP
- Minibatch Discrimination
- Condition Regression
- Flood Prediction
- Urban Flood Prediction
- Flood Footprint Map
- Image-to-Image Translation
- Deep Learning

---

## 👨‍💻 Author

### Hyeon-Woo Seo

- Backend / AI Research
- Deep Learning 기반 침수 예측 연구
- PyTorch 기반 생성 모델 구현
- Conditional GAN Architecture Research

---

## 📝 Note

This repository is organized as a portfolio and research implementation project.  
Due to data confidentiality, raw datasets and trained model weights are not included.