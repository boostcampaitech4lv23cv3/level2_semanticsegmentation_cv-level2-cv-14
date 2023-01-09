# BoostCamp AI Tech4 level-2-재활용 품목 분류를 위한 Semantic Segmentation
***
## Member🔥
| [김지훈](https://github.com/kzh3010) | [원준식](https://github.com/JSJSWON) | [송영섭](https://github.com/gih0109) | [허건혁](https://github.com/GeonHyeock) | [홍주영](https://github.com/archemist-hong) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/kzh3010" width="100"> | <img src="https://avatars.githubusercontent.com/JSJSWON" width="100"> | <img src="https://avatars.githubusercontent.com/gih0109" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> | <img src="https://avatars.githubusercontent.com/archemist-hong" width="100"> |
***
## Index
* [Project Summary](#Project-Summary)
* [Team role](#Team-role)
* [Procedures](#Procedures)
* [Model](#model)
* [Result](#result)
* [Command](#Command)
* [Wrap UP Report](#Wrap-UP-Report)  
***


## Project-Summary

<img width="150%" src="./image/model_visual.png"/>

### 개요
- 주제: 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 **쓰레기로 인한 환경 문제를 해결**해보고자 합니다.
- 기대효과: 우수한 성능의 모델은 쓰레기장에 설치되어 **정확한 분리수거**를 돕거나, 어린아이들의 **분리수거 교육** 등에 사용될 수 있을 것입니다.
- Input: 쓰레기 객체가 담긴 이미지
- Output: segmentation 결과 (pixel별 classification 결과)
### 데이터 셋 구조
- annotation: COCO format


***

## Team role
- 김지훈: Augmentation 실험, Ensemble 구현 및 실험
- 원준식: baseline 작성, cross validation 실험
- 송영섭: mmseg를 위한 여러 Augmentation 구현 및 실험
- 허건혁: EDA를 위한 시각화 tool 개발 & Copy Paste augmentation 실험
- 홍주영: multi-scale TTA , Pseudo-labeling 실험
---

## Procedures
대회 기간 : 2022.12.19 ~ 2023.01.05

| 날짜 | 내용 |
| :---: | :---: |
| 12.19 ~ 12.25 | BoostCamp 강의 수강 및 Segmentation 이론 학습|
| 12.26 ~ 01.01 | Data EDA & Model Experiment |
| 01.02 ~ 01.05 | HyperParameter Tuning & model Ensemble |
---
## Model

[지난 object detection 대회](https://github.com/boostcampaitech4lv23cv3/level2_objectdetection_cv-level2-cv-14)에서 좋은 성능을 보여주었던 Swin Transformer - L 모델과 Convnext - XL 모델을 UperNet의 backbone으로 사용
| Model | mIoU (val) | mIoU (LB) | Training Time |
| :---: | :---: | :---: | :---: |
| swin - l | 0.7673 | 0.7325 | 8h 29m 32s |
| convnext - xl | 0.7123 | 0.6792 | 2h 22m 56s |

convnext 모델이 swin 모델과 비슷한 성능을 보여주지만 훈련 시간이 매우 적기 때문에 convnext 모델도 함께 baseline으로 차용


### Augmentation
#### mmsegmentation에 구현된 aug
- Resize
- Random Flip
- PhotoMetricDistortion
- Normalize Augmentation \
#### Ablumentation Augmentation
- Random Rotate90
- One of (Blur, GaussianBlur, MotionBlur

### Optimization

- AdamW
    
    mmsegemntation에서 convnext backbone을 사용하는 경우 기본적으로 정의된 AdamW사용
    
    - lr = 0.0001 → 6e-5 (마지막 제출 전 fine tuning)
- Warm Up linear Scheduler
    - mmsegemntation에서 convnext backbone을 사용하는 경우 기본적으로 정의된 warm up linear scheduler 사용

## 기타

모델이 클수록 모델 성능 향상에 큰 도움이 되나 학습 속도가 기하급수적으로 증가하여 **실험 시간 및 성능의 균형을 잡기 위해 convnext-tiny 모델 사용**

기본 학습 20,000 iter로 실험하였으나, 대회 후반 리더보드 제출을 위해서는 60epoch, 80epoch 등 길게 학습

input size는 크기가 클수록 성능 향상이 되어 (512, 512)로 학습하고, inference 시에도 (512, 512)로 추론한 결과를 (256, 256)으로 resize 하여 제출

## TestTimeAugmentation

|  | LB Score |
| :---: | :---: |
| Convnext-xl without multi-scale TTA | 0.7601 |
| Convnext-xl with multi-scale TTA | 0.7618 |
- **Multi-scale TestTimeAugmentation** 적용 (256, 256) ~ (640, 640)까지 (128, 128) 씩 증가시켜 적용
    - Random Flip, Normalize

## Pseudo Labeling

| type | LBscore | val mIoU | val mIoU best | 시간 | step | dataset |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| base | 0.6792 | 0.7104 | 0.7123 | 약 1h | 20000 | train only |
| pseudo-label | 0.7510 | 0.7561 | 0.7561 | 약 9h | 100000 | train + test |
| pseudo-label2 | 0.7601 | 0.9664 | 0.9664 | 약 14h | 163600 | train + test + val + 남은 데이터 |
| pseudo-label3 | 0.7733 | 0.9646 | 0.9646 | 약 14h | 163600 | train + test + val + 남은 데이터 |

train data 뿐만 아니라, validation data와 annotation이 되지 않은 기타 데이터, test data에 pseudo label을 사용하여 모델을 학습시키고, 모델 성능에 개선이 있는 경우 반복해서 pseudo label을 다시 달아 학습시키는 방식으로 모델 성능을 크게 끌어 올림



---
## Result
| Leaderboard | Public | Private |
| :---: | :---: | :---: |
| Score(mIoU) | 0.7733 | 0.7601 |
| Ranking | 6th / 19th | 7th / 19th |

## Command
- mmdetection train command
```
cd mmsegmentation
python tools/train.py {config file}
```

- submission csv 생성 command

```
cd mmsegmentation
python tools/inference.py {model} {checkpoint} --out {저장경로}
```

- Data Visual을 위한 streamlit command
```
streamlit run visual/visual.py --server.port=[port 번호]
```


---
## Wrap UP Report
- [Report](https://howmuchisit.notion.site/Wrap-Up-f39d8b3501dc40c49b61a82b0a3756a2)
