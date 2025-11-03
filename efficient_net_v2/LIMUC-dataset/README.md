## Efficient Net V2 with LIMUC-dataset
This model is a family of Convolutional neural networks designed for image classification. The model is trained with LIMUC-dataset to classify severity scores for ulcerative colitis using different images. 

#### LIMUC-dataset/first-implementation
**Description**:
- Uses LIMUC-dataset to categorize endoscopic images into 4 Mayo score classes (0-3) on the basis of severity levels (lowest to highest)
- Loads pre-trained EfficientNetV2-S model and fine-tunes it by replacing the final classifier layer, adding dropout for regularization.
- Computes balanced class weights to address uneven distribution of images across severity classes to remove bias.
- Implements data augmentation (flips, rotations, color jitter), early stopping, learning rate scheduling, and an 80-10-10 train/validation/test split.
- Calculates per-class and overall metrics like precision, recall, f1-score, accuracy

**Findings**
Overall performance:
- Accuracy: 74.0 %
- Macro Average F1 Score: 0.64
- Weighted Average F1 Score: 0.73

Per-Class Performance:
| Class  | Precision | Recall | F1-Score | Support |
|--------|------------|---------|-----------|----------|
| Mayo 0 | 0.84       | 0.87    | 0.85      | 925      |
| Mayo 1 | 0.62       | 0.64    | 0.63      | 464      |
| Mayo 2 | 0.54       | 0.42    | 0.47      | 177      |
| Mayo 3 | 0.66       | 0.56    | 0.60      | 120      |

- Strong performance on Mayo 0 with 87 % overall.
- Moderate performance on Mayo 1 with balanced precision and recall.
- Weakest performance on Mayo 2 with only 42% recall.
- Mayo 3 with moderate recall (56%) despite smallest class.

#### LIMUC-dataset/second-implementation
**Description**:
- The script is same. However, the neural network training is stochastic causing the results to be random in this implementation.
- We have shuffle=True in the train_loader causing the batches to be present in different orders.
- We have random flips, rotations, and color jittering that produces different variations of images each epoch.
- The 0.3 dropout randomly drops different neurons in training.
- Presence of non-deterministic CUDA operations.

**Findings**
- Accuracy: 74.5 (0.5% improvement)
- Overall Precision: 0.6653
- Overall Recall: 0.6769
- Overall F1-score: 0.6571

Per-Class Performance:
| Class  | Precision | Recall | F1-Score | Support |
|--------|------------|---------|-----------|----------|
| Mayo 0 | 0.9042     | 0.8065  | 0.8526    | 925      |
| Mayo 1 | 0.5946     | 0.7651  | 0.6692    | 464      |
| Mayo 2 | 0.5524     | 0.3277  | 0.4113    | 177      |
| Mayo 3 | 0.6101     | 0.8083  | 0.6953    | 120      |

- Mayo 0 precision improved significantly (84%->90%) but recall decreased slightly (87%->81%)
- Mayo 1 recall improved substantially (64%->77%) but at the cost of precision (62%->59%).
- Mayo 2 recall deteriorated further (42%->33%) making this class the most challenging.
- Mayo 3 showed remarkable improvement in recall (56%->81%), suggesting better minority class mapping.

## Conclusion
- Both implementation achieved similar overall accuracy ( ~74%) demonstrating model stability.
- Variability in results comes from the stochastic nature of neural network training.
- Mayo 0 is most consistently identified class in both runs due to large sample size.
- Mayo 1 showed improved senstivity (+13% recall) with slight precision trade off.
- Mayo 2 is the most inconsistent and poorst performing class. Might be due to lowest sample size (n=177) and recall dropped 9% in second run, indicating high variability.
- Mayo 3 showest largest improvement despite smallest sample size (n=120). Recall increased by 25% in second implementation and F1-score improved from 0.60 to 0.70.

## Clinical Implications
- Second implementation's higher recall for severe cases (Mayo 3) could be preferable in medical settings where missing severe disease is more costly than false alarms.
- The persistent difficulty in Mayo 2 suggests potential class overlap or need for more training samples.
