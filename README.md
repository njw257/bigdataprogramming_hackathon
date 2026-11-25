# 쓰레기 이미지 분류와 4블록 잔차 CNN

계명대학교 경영정보학과 남종욱 · 빅데이터프로그래밍 1차 해커톤

PyTorch로 직접 구현한 12클래스 잔차 CNN입니다. 교수님 평가 사이트의 최종 모델 **#27**은 전체 **61.67% (37/60)**, 독립 생성 이미지 **77.78% (14/18)**를 기록했습니다.

## 실행과 제출

[Colab에서 열기](https://colab.research.google.com/github/njw257/bigdataprogramming_hackathon/blob/main/garbage_classifier_residual_final.ipynb)

1. [`garbage_classifier_residual_final.ipynb`](garbage_classifier_residual_final.ipynb)을 열고 GPU 런타임에서 위부터 실행합니다. 데이터는 Kaggle과 Hugging Face에서 가져옵니다. 기본 실행은 내장된 #27 가중치로 재평가합니다.
2. #17 가중치부터 재현하려면 `RUN_REPLAY = True`로 설정합니다. 잔차 CNN을 처음부터 학습하려면 `RUN_FROM_SCRATCH = True`를 사용합니다. 선택 실행 셀에서 하이퍼파라미터를 다시 비교하거나 외부 이미지를 분류할 수 있습니다.
3. 교수님 사이트에서는 [`model.py`](model.py)와 [`weights.pt`](weights.pt)를 함께 업로드합니다. 이 파일들은 #27 제출 당시의 파일입니다.

## 데이터와 학습

| 출처 | 학습 | 검증 | 시험 |
|---|---:|---:|---:|
| [Kaggle V1](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | 12,394 | 1,549 | 1,550 |
| [Kaggle V2 v12](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/versions/12) | 1,253 | 156 | 156 |
| [TrashBusters](https://huggingface.co/datasets/TrashBusters/combined) | 3,421 | 774 | 777 |
| 합계 | **17,068** | 2,479 | 2,483 |

V1 원본 15,515장에서 레이블 충돌 6장과 같은 클래스의 동일 파일 16장을 제외했습니다. V2는 biological, plastic, trash, metal을 보강했고, TrashBusters는 원본 출처와 레이블을 확인해 paper, trash, plastic만 선별했습니다. 추가 데이터는 파일·RGB 픽셀 해시로 중복을 확인하고 pHash 유사 후보를 보류했습니다. TrashBusters 리비전은 `8c6e9226b01fc8a349750b69911b28d7d7c11443`입니다.

최종 모델의 채널은 `32 → 64 → 128 → 192`입니다. 입력 `128×128`, BatchNorm, 전역 평균 풀링, Cross-Entropy를 사용했습니다. V1·V2에서 만든 가중치에서 시작해 TrashBusters를 합쳐 Adam `0.0001`, 배치 `32`로 5 epoch 학습했습니다. 학습 중 Dropout과 무작위 증강은 0이었습니다. 검증에서는 원본·변형 Kaggle 평균 Macro F1과 TrashBusters Macro F1에 같은 비중을 주어 5번째 epoch를 선택했습니다.

학습률, Dropout, BatchNorm, 기하·조명 증강, 블록 수, 필터 폭과 입력 해상도를 단계적으로 비교했습니다. 저해상도·크롭 증강 비교, 과적합 점검, 실제 프롬프트는 [보고서](빅데이터프로그래밍_1차_해커톤_보고서_남종욱.docx)에서, 초기 비교 코드는 [별도 실험 저장소](https://github.com/njw257/bigdata)에서 확인할 수 있습니다.

## #27 성능

| 평가 | 이미지 수 | 정확도 | 평균 손실 | Macro F1 |
|---|---:|---:|---:|---:|
| 학습 중 · 선택 epoch | 17,068 | 93.65% | 0.2034 | — |
| Kaggle 원본 검증 | 1,705 | 87.80% | 0.4606 | 85.89% |
| Kaggle 변형 검증 | 1,705 | 80.65% | 0.7107 | 78.49% |
| TrashBusters 검증 | 774 | 72.09% | 0.8187 | 73.41% |
| V1 원본 시험 | 1,550 | 87.35% | 0.3865 | 85.08% |
| V2 보류 시험 | 156 | 87.82% | 0.439 | 86.28% |
| TrashBusters 보류 시험 | 777 | 73.23% | 0.8298 | 75.33% |

| 교수님 평가 | 정답 | 정확도 | 평균 손실 |
|---|---:|---:|---:|
| 전체 | **37 / 60** | **61.67%** | **1.8831** |
| 콜라주 확대 | 23 / 42 | 54.76% | 2.1689 |
| 독립 생성 | 14 / 18 | 77.78% | 1.2163 |

교수님 평가의 60장은 모두 AI 생성 이미지입니다. 저해상도 콜라주가 포함되어 전체 점수는 독립 생성 18장과 함께 해석했습니다. 위 Kaggle·TrashBusters 수치는 보관된 학습·재평가 기록입니다.
