# Overview

---

- 피부기록 앱 스킨로그와 화장품 추천 앱 매니폴드를 서비스하는 아트랩의 기업 과제 수행
- 제공받은 피부 데이터를 바탕으로 피부 평가, 분석 및 XAI
- 사용자에게 피부에 대한 정보를 제공하고, 화장품 선택에 도움주는 것이 목적

- **Input** : 피부 부위별 사진(측면, 중앙부, 이마)
- **Output** : 유분, 민감도, 주름, 색소침착 네 가지 기준에 대한 0~4점의 라벨링

## 평가방법

---

- ***Macro-recall***
    - data Imbalance 문제를 해결하기 위해 F1 score와 macro Recall 고려
    - Macro-recall : 각 라벨에 대한 recall의 평균
    - F1-score : 각 라벨에 대한 F1 score의 평균
    - ex) wrinkle에서 4점에 대한 label
        
        ![image](https://user-images.githubusercontent.com/57162812/172381370-2038ea09-7390-40d5-976d-1ea0f5a8f82e.png)
        
        위 경우 병의 진단과 같은 맥락으로 1~4점을 0점으로 예측하는 경우를 줄이는 것보다, 0점을 1~4점으로 예측하는 경우을 줄이는 것을 중요하게 생각해서 ***F1 score***가 아닌 ***Macro Recall***를 평가지표로 선정했습니다.
        

# **Archive contents**

---

```python
.
├── 📂 naverboostcamp_dataset/
│   ├── 📂 naverboostcamp_train/
│   │   ├── 📂 JPEGImages/
│   │   │   ├── 📝 00000NBC.jpg
│   │   │   ├── 📝 00000NBC.json
│   │   │   └── 📝      ⋮
│   │   └── 📝 annotations.json
│   └── 📂 naverboostcamp_val/
│       ├── 📂 JPEGImages
│       │   ├── 📝 00000NBC_val.jpg
│       │   ├── 📝 00000NBC_val.json
│       │   └── 📝      ⋮
│       └── 📝 annotations.json
├── 📂 MWML/
│   ├── 📂 configs/
│   ├── 📂 libs/
│   ├── 📂 main/
│   │   ├── 📝 train.py
│   │   └── 📝 valid.py
│   ├── 📂 outputs/
│   └── 📂 pretrained_models/
│       └── 📝 RegNetY-3.2GF_dds_8gpu.pyth
└── 📂 baseline/
    ├── 📂 customs/{feature name}/settings/
    │   ├── 📝 arg.py
    │   ├── 📝 dataloader.py
    │   ├── 📝 loss.py
    │   ├── 📝 model.py
    │   ├── 📝 optimizer.py
    │   ├── 📝 scheduler.py
    │   └── 📝 transform.py			
    ├── 📂 utils
    │   └── 📝 set_seed.py
    └── 📝 train.py
```

# Dataset

---

- 피부 부위별 사진을 유분, 민감도, 주름, 색소침착 네 가지 기준에 대해 0~4점으로 라벨링

![image](https://user-images.githubusercontent.com/57162812/172381415-09dafb45-155a-4140-ad45-63819a4d6969.png)

![image](https://user-images.githubusercontent.com/57162812/172381453-785a2c92-6a11-4e9c-89c2-9e3f44a17558.png)

# Experiment

---

- ***CLAHE***
    
    ![image](https://user-images.githubusercontent.com/57162812/172381517-b23ae784-7d85-4531-b345-528b82709717.png)
    
- ***Loss Weight***

![image](https://user-images.githubusercontent.com/57162812/172381555-d926232b-8382-4124-93e1-b5a37ff805fa.png)

- ***Ordinal Classification***

![image](https://user-images.githubusercontent.com/57162812/172381585-c24a3514-b7d7-4ae8-aec3-09e037c6e8aa.png)

각각의 클래스를 원핫인코딩 하는 대신, 각 클래스를 ordinal하게 인코딩하여 클래스에 순서를 부여

![image](https://user-images.githubusercontent.com/57162812/172381619-2947b60c-94b8-4cf1-81c9-48355d4cbafa.png)

# Results

---

|  | macro recall | f1 |
| --- | --- | --- |
| pigmentation | 74.172 | 0.7258 |
| oil | 61.3 | 0.5024 |
| sensitive | 77.967 | 0.7158 |
| wrinkle | 61.83 | 0.5149 |

# XAI

---

- ***Oil***
    
    <img src="https://user-images.githubusercontent.com/57162812/172381695-2e8c3e5b-6d60-45ad-b07a-4a2795112d3c.png" width="30%">
    
- ***Pigmentation***
    
    <img src="https://user-images.githubusercontent.com/57162812/172381717-ad7dddca-d4c6-47f2-b520-ad2fac581250.png" width="30%">
    
- ***Sensitive***
    
    <img src="https://user-images.githubusercontent.com/57162812/172381745-f6cf8466-8fcf-4dc3-ad70-4d0298717a8c.png" width="30%">
    
- ***Wrinkle***
    
    <img src="https://user-images.githubusercontent.com/57162812/172381765-4fdc0b3f-b5dc-4c47-b47f-0b90c5c4a352.png" width="30%">
    

# Requirements

---

```jsx
pip install -r requirements.txt
```

# Train.py

---

```python
# MWML
python main/train.py --cfg {config 경로}

# baseline
python train.py --dir {feature 이름} --arg_n {arg 이름}
```
