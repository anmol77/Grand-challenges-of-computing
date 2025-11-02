## Vision Transformer with Kvasir-dataset

#### Kvasir-dataset/vision_transformation_kvasir.py
**Description**:
- Builds a ViT architecture from scratch with 16x16 patch embeddings, 12 transformer blocks, 12 attention heads, and 768 embedding dimensions, specifically designed to classify 8 types of gastrointestinal conditions from endoscopy images.
- Unlike the EfficientNetV2 approach, this model trains the vision transformers completely from scratch without using pretrained weights, relying solely on the given dataset for visual features.
- Implements the 80-10-10 stratified split (training, validation, test) with moderate data augmentation including random flips, rotations, and color jittering, while maintaining reproducibility through fixed random seeds.
- Uses AdamW optimizer with weight decay (1e-4), ReduceLROnPlateau scheduler that reduces learning rate when validation loss plateaus, early stopping with 10-epoch patience, and saves the best model based on validation loss rather than accuracy.
- Saves extensive training history(loss, accuracy, learning rate per epoch), generates classification reports with per-class metrics, creates confusion matrices, and stores all results as CSV files for comprehensive post-training analysis and comparison with other models.

**Findings**

Overall Performance:
- Test Accuracy: 69.5% representing moderate performance
- Balanced Metrics: Precision (69.82%), Recall (69.5%), and F1-Score (68.96%) are closely aligned, indicating consistent performance without major bias
- Macro vs weighted averages: Identical values (69.82%) suggest relatively balanced class distribution and no significant class imbalance effects.
- Performance gap: 23.5 percentage points lower than EfficientNetV2 (93% vs 69.5%), highlighting the challenge of training transformers from scratch
- Training from scratch limitation: Lack of pretrained weights significantly hampered the model's ability to learn effective visual representations from the limited medical dataset

Per Class Performance:

| Class                 | Precision | Recall | F1-Score |
|------------------------|------------|---------|-----------|
| dyed-lifted-polyps     | 0.7273     | 0.64    | 0.6809    |
| dyed-resection-margins | 0.6727     | 0.74    | 0.7048    |
| esophagitis            | 0.6842     | 0.65    | 0.6667    |
| normal-cecum           | 0.7200     | 0.90    | 0.8000    |
| normal-pylorus         | 0.7838     | 0.87    | 0.8246    |
| normal-z-line          | 0.6020     | 0.59    | 0.5960    |
| polyps                 | 0.6250     | 0.70    | 0.6604    |
| ulcerative-colitis     | 0.7705     | 0.47    | 0.5839    |
| **accuracy**           | **0.695**  | **0.695** | **0.695** |
| **macro avg**          | 0.6982     | 0.695   | 0.6896    |
| **weighted avg**       | 0.6982     | 0.695   | 0.6896    |

- Normal-pylorus has the best overall performance. High recall shows excellent sensitivity for identifying normal pyloric tissue. Strong precision minimizes false positives.
- Normal-cecum has the second best performance. Exceptional recall indicates very few missed cases of noral cecum. High performance on normal anatomical structures consistent across models. Clear visual characteristics of cecal landmarks facilitate classification.
- Ulcerative colitis has high precision but low recall. When model predicts ulcerative colitis, it's correct 77% of the time. Critical weakness: misses 53% of actual ulcerative colitis cases. Conservative prediction strategy may prioritize avoiding false positives.
- Dyed-resection-margins have balanced performance. Highest recall among non-normal classes suggests good sensitivity. Artificial dye markers may provide distinctive visual cues. Moderate precision indicates some confusion with other classes
- Dyed-lifted-polyps have adequate performance. Lower recall (64%) means 36% of cases are missed. Decent precision suggests predictions are reasonably reliable. Dye-enhanced features not fully leveraged by the model
- Esophagitis have consistent moderate performance. Balanced precision and recall indicate no strong directional bias. Better than ViT on LIMUC (87% F1) but still room for improvement. Inflammatory patterns may require more training data to learn effectively.
- Polyps have below average performance (Precision: 62.5%, Recall: 70%, F1: 66%). Lowest precision among all classes indicates high false positive rate. 70% recall means 30% of polyps are missed. Clinically concerning given polyps are precancerous lesions requiring detection.

Confusion Matrix Analysis:

- Dyed-lifted-polyps:
  - 64 correctly classified.
  - 36 were misclassified as dyed-resection-margins.
 
- Esophagitis:
  - 65 correctly classified.
  - 32 misclassified as normal-z-line, meaning these two classes look visually similar in your data.
 
- Normal-cecum:
  - 90 correct.
  - 10 confused as polyps.
 
- Ulcerative-colitis:
  - 47 correct
  - 31 misclassified as polyps and 20 as normal-cecum, meaning some inflammation features might resemble those patterns.

### Training Dynamics Analysis
- Model stopped improving relatively early, suggesting limited learning capacity from scratch.
- ReduceLROnPlateau scheduler indicates optimization challenges.
- 10-epoch patience for early stopping may prevent sufficient convergence.
- AdamW with 3e-5 learning rate may be suboptimal for this architecture and dataset size.
- Moderate augmentation cannot compensate for lack of pretrained weights.

## Comparison ViT from Scratch vs EfficientNetV2 Pretrained
- ViT achieved 69.5% vs EfficientNetV2's 93% on identical dataset
- 23.5 percentage point gap demonstrates critical importance of transfer learning.
- CNNs with pretraining significantly outperform transformers trained from scratch on small datasets.
- EfficientNetV2 leverages millions of ImageNet images; ViT learns only from ~8,000 Kvasir images.
- EfficientNetV2 approaches clinical utility (93%); ViT requires substantial improvement (69.5%)

## Clinical Implications
- 69.5% accuracy falls well below acceptable standards for medical diagnostic tools.
- 30% missed polyps (70% recall) unacceptable for colorectal cancer screening programs.
- 53% missed cases (47% recall) renders model unsuitable for disease activity assessment.
- Performance ranging from 59.6% to 82.5% F1 creates unpredictable clinical behavior.
- Nearly one-third of cases misclassified could lead to inappropriate clinical decisions.
- Current performance suitable for academic studies comparing architectures, not clinical practice.
- Demonstrates ViT applicability to GI endoscopy but requires substantial improvements.
- Could illustrate AI capabilities and limitations to medical trainees.

## Future Potential
- Pretrained ViTs could potentially reach 85-90% accuracy range.
- Transformer efficiency could enable live endoscopy assistance after optimization.
- ViTs excel at combining imaging with clinical text data.
- Attention mechanisms could track disease progression across multiple exams



