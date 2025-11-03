## Swin Transformer with LIMUC-dataset
A swin transformer is an advanced type of Vision Transformer(ViT) designed for better efficiency and scalability in computer vision tasks. It divides an image into small windows and applies self-attention locally within each window, then shifts the windows between layers to let them interact. This makes it hierarchical like a CNN and efficient for high-resolution images. This model is powerful for tasks like classification, detection, and segmentation.

####LIMUC-dataset/swin_transformer.py
**Description**:
- Uses the Swin Base Patch4 Window12 model with 384x384 input resolution from the timm library, loading pretrained weights from a local file to leverage transfer learning for ulcerative colitis Mayo score classification (Mayo 0-3).
- Implements mixed precision training (AMP) for memory efficiency, reduced batch size (16) to accommodate the larger Swin Base model, and longer training schedule (100 epochs) with warmup period (5 epochs) and cosine learning rate decay from 1e-4 to 1e-6.
- Custom dataset loader supports multiple image formats (.bmp, .jpg, .png) with flexible directory structure validation, checks for missing classes, and provides detailed logging of data distribution across train/validation/test splits.
- Includes early stopping with 15-epoch patience, automatic mixed precision for faster training, configurable multi-GPU distributed training support, and organized output directory structure for saving results, metrics, and visualizations.
- Applies clinically appropriate augmentations including horizontal/vertical flips (simulating different scope orientations), rotation, color jittering, and affine transformations while using ImageNet normalization statistics to match the pretrained model's expected input distribution.

**Findings**:

Overall performance:
- Test Accuracy: 75.09%
- Test Loss: 1.1537
- Weighted Average: Precision (74%), Recall (75%), F1-Score (74%) show balanced performance accounting for class distribution
- Macro Average: Precision (69%), Recall (66%), F1-Score (66%) reveal challenges with minority classes
- Improvement over ViT from scratch:17.3% point gain (75.09% vs 57.82%), demonstrating significant benefit of pretrained weights and hierarchical architecture

Per-Class performance:

| Class  | Precision | Recall | F1-Score | Support |
|---------|------------|---------|-----------|----------|
| Mayo 0  | 0.80       | 0.93    | 0.86      | 925      |
| Mayo 1  | 0.69       | 0.51    | 0.59      | 464      |
| Mayo 2  | 0.66       | 0.40    | 0.49      | 177      |
| Mayo 3  | 0.61       | 0.82    | 0.70      | 120      |
| **accuracy**     |            |         | **0.75**  | **1686**  |
| **macro avg**    | 0.69       | 0.66    | 0.66      | 1686      |
| **weighted avg** | 0.74       | 0.75    | 0.74      | 1686      |

- Mayo 0 has strongest performance.Highest support (925 samples) provides substantial training data. Exceptional recall (93%) means only 7% of remission cases are miss. 80% precision indicates some false positives from other Mayo grades.
- Mayo 1 had moderate challenges. Low recall (51%) means nearly half of mild disease cases are missed. When model predicts Mayo 1, it's correct 69% of the time. 464 samples provide reasonable representation but confusion with adjacent grades evident. Critical gap for early disease detection.
- Mayo 2 had the poorest performance. Lowest F1-score (49%) indicates significant classification difficulty. Extremely low recall (40%) means 60% of moderate disease cases are misclassified. Small sample size (177) contributes to learning difficulties. Frequently confused with Mayo 1 or Mayo 3 severity levels.
- Mayo 3 had surprisingly high recall. Highest recall (82%) among pathological classes despite smallest sample size (120). Lower precision (61%) suggests over-prediction of severe disease. Severe inflammation features may be more visually distinctive. Better than Mayo 2 performance despite less training data.

Class imbalance impact
- Skewed distribution: Mayo 0 (925) >> Mayo 1 (464) > Mayo 2 (177) > Mayo 3 (120)
- 5.4:1 ratio: Largest to smallest class shows extreme imbalance
- Mayo 2 and Mayo 3 collectively represent only 17.6% of test set (297/1686)
- 8 percentage point difference in F1 (74% vs 66%) highlights minority class struggles

### Comparison with previous ViT architectures
- Accuracy: 75.09% vs 57.82% (+17.3 points)
- Mayo 0 F1: 86% vs 72.2% (+13.8 points)
- Mayo 1 F1: 59% vs 34.4% (+24.6 points)
- Mayo 2 F1: 49% vs 0% (+49 points - dramatic improvement)
- Mayo 3 F1: 70% vs 3.1% (+66.9 points - massive improvement)

### Dataset-Specific Challenges
- LIMUC limitations: Relatively small dataset (~2,000 total images) for deep learning
- Class imbalance severity: 7.7x ratio between Mayo 0 and Mayo 3
- Label quality: Mayo scoring subjectivity means training labels may contain inconsistencies
- Image quality variation: Different endoscopes, lighting, and cleaning quality affect input consistency

## Clinical Implications

Diagnostic Accuracy
- 93% recall adequate for confirming disease quiescence in maintenance therapy.
- 51% recall for Mayo 1 means half of mild disease goes undetected.
- 60% of Mayo 2 cases misclassified could delay appropriate therapy intensification.
- 82% recall reasonable but 18% missed cases still concerning for urgent interventions.
- 75% accuracy approaches but doesn't reach clinical deployment threshold (typically >85%).

Misclassification Patterns
- Mayo 1 missed cases (49%) likely classified as Mayo 0.
- Mayo 2 missed cases (60%) likely classified as Mayo 1.
- Could lead to inadequate treatment and disease progression.
- 39% false positive rate for Mayo 3 (100% - 61% precision).
- Could result in unnecessary treatment escalation.

Practical Applications
- Flag cases requiring expert review, particularly for Mayo 2-3.
- Prioritize urgent cases with predicted Mayo 3 for expedited gastroenterology consultation.
- Help fellows learn Mayo scoring with AI-suggested grades for discussion.
- However, 75% accuracy insufficient for autonomous diagnosis. Misdiagnosis rates too high for primary diagnostic tool
- Single-center dataset limits generalizability to other institutions.

Comparison with Human Expert Performance
- Published studies show 80-90% inter-observer agreement on Mayo scores. 75% accuracy below expert level but better than untrained observers.
- AI provides reproducible assessments without fatigue or bias.
- Model lacks patient history, symptoms, and histology that inform human decisions.






