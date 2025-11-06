## Vision Transformer with LIMUC-dataset
Vision Transformer(ViT) is a deep learning model that applies the Transformer architecture designed for language tasks and image recognition. We are using the LIMUC-dataset to train this model for image classification of GI diseases.

#### LIMUC-dataset/vision_transformer.py
**Description**:
- Implements a Vision Transformer(ViT) from scratch with patch embeddings(8x8 patches), multi-head self-attention mechanisms and 12 transformer blocks with 768 embedding dimensions, designed specifically for ulcerative colitis severity classification (Mayo 0-Mayo 3).
- Addresses dataset imbalance through multiple strategies including computed class weights applied to the loss function, label smoothing(0.1) and extensive data augmentation(random crops, flips, rotations, color jittering, and random erasing) to improve model robustness across all Mayo severity scores.
- Uses AdamW optimizer with weight decay (0.05) for regularization, cosine annealing learning rate scheduler that gradually reduces learning rate over 150 epochs, gradient clipping (max norm 1.0) to prevent exploding gradients, and early stopping with 15-epoch patience to prevent overfitting.
- Trains on the LIMUC-dataset to classify colonoscopy images into four Mayo scores representing ulcerative colitis severity levels, with a small batch size (8) suitable for the limited medical imaging dataset and high-resolution analysis requirements.
- Unlike stadard ViT implementations, includes an additional pre-classification layer that reduces dimensionality (768-> 384) with layer normalization and GELU activation before the final classificaiton layer, potentially improving feature refinement for the fine-grained medical classification task.

**Findings**

Overall Performance:
- Validation accuracy: 57.82%
- Training instability: Validation accuracy ranging from 5.91% to 57.82% suggesting model struggled to find stable patterns.
- Early stopping: training terminated at epoch 40 out of 150 due to no improvement for 15 consecutive epochs, indicating the model reached its learning capacity early.
- Training time: Process required 4.3 hours which is computationally very high
- Training accuracy plateaued around 32-36% while best validation accuracy reached 57.82%, suggesting inconsistent learning patterns.

Per Class Performance Analysis
- Mayo 0
  - Best Performing class with 61.7% precision, 87.1% recall, and 72.2% F1-Score.
  - High recall indicates the model successfully identifies most of these cases.
  - Lower precision suggests false positives where other severity grades are misclassified as Mayo 0.
 
- Mayo 1
  - Poor performance with 46.5% precision, 27.3% recall, 34.4% F1-Score.
  - Extremely low recall means the model misses nearly 73 % of mild ulcerative colitis cases.
  - The model struggles to distinguish mid inflammation from other severity levels.
  - Represents a critical clinical gap for early disease detection.
 
- Mayo 2
  - Complete failure for Mayo 2 classification.
  - Model failed completely to correctly identify any Mayo 2 cases in validation.
  - All moderate diseases images were misclassified into other categories.
  - Represents severe limitation for clinical capability.
 
- Mayo 3
  - Near complete failure with 7.1% precision, 2.0% recall, and 3.1% F1-Score.
  - Model correctly identified only 2% of severe ulcerative colitis cases.
  - Of the few predictions made for Mayo 3, only 7.1 % were correct.
  - Critical failre given the urgency of identifying severe disease.
 
Training Challenges:
- Validation Accuracy fluctuated wildly (epoch 3: 8.13% to epoch 4: 49.92%) indicating unstable learning.
- Training loss decreased minimally (1.52->1.47), suggesting optimization difficulties.
- Despite using class weights, the model heavily biased towards Mayo 0 for most cases.
- Low training accuracy (32-36%) suggests the model failed to learn discriminative features even from training data.

## Clinical Implications
- 57.82 % accuracy is below the threshold for any clinical decision support system.
- 2% recall for Mayo 3 means 98% of severe ulcerative colitis cases would be missed.
- Complete failure on Mayo 2 creates a "blind spot" for moderately active disease.
- High false positive rate for Mayo 0 could lead to inappropriate treatment de-escalation.
- Cannot replace endoscopic judgement due to performance being far below human expert accuracy which is usually above 85 %.

## Model Architectural Challenges
- Custom ViT architecture may lack the inductive biases needed for medical images.
- 8x8 patches might miss subtle mucosal inflammation patterns critical for Mayo scoring.
- Transformers typically requires large datasets and LIMUC-dataset may be insufficient.
- Unlike Efficient Net V2 model which used image pretrained weights, this ViT was trained from scratch.
