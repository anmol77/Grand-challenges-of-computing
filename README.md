**Efficient Net V2**

    LIMUC-Dataset
    
    /first_implementation/
      - training accuracy ~98%
      - validation accuracy - 75.35 %
      - test accuracy - 73.84
      - large gap between training and validation accuracy suggest overfitting
      - Per class performance (Mayo 0: f1 0.85, Mayo 1: f1 0.63, Mayo 2: f1 0.47, Mayo 3 f1 0.60)
      - As per class performance (f1 score) Mayo 0 performed the best.
    
    /second_implementation/
      - addresses issues of first implementation
      - adds more data augmentation
      - implementation of early stopping
      - Since Mayo 0 (925 samples), Mayo 1 (464 samples), Mayo 2 (177 samples), Mayo 3 (120 samples), this implementation addresses class imbalance
    
    Kvasir-Dataset
    
    /first_implementation/
      - Best validation accuracy: 93.50%
      - Training completed in 1.7 hours (6,225 seconds)
      - Final test metrics:
          - Accuracy: 93.00%
          - Precision: 93.02%
          - Recall: 93.00%
          - F1-score: 93.00%

**Vision Transformers**

    LIMUC-Dataset
    - Best validation accuracy: 57.82%
    - Training completed in 4.3 hours (15,550 seconds)
    - Early stopping triggered at epoch 40/150
    - Class-wise F1-scores:
      - Mayo 0: 0.722
      - Mayo 1: 0.344
      - Mayo 2: 0.000
      - Mayo 3: 0.031
    - Strong performance on Mayo 0 class (normal cases)
    - Significant challenge in identifying Mayo 2 and 3 cases
    - High class imbalance evident from performance metrics
    - Validation accuracy showed high fluctuation during training (ranging from ~6% to 57%)
    
    Kvasir-Dataset
    - Best validation accuracy: 70.00%
    - Training completed in 1.5 hours (5,558 seconds)
    - Early stopping triggered at epoch 45/100
    - Final test metrics:
      - Accuracy: 69.50%
      - Macro Precision: 69.82%
      - Macro Recall: 69.50%
      - Macro F1-score: 68.96%
    - Most balanced performance across all classes compared to other models
    - Strong performance on normal-pylorus (F1: 0.825) and normal-cecum (F1: 0.800)
    - Relatively consistent performance across different disease classes
    - Model showed stable convergence with cosine learning rate scheduling


  
