## Efficient Net V2 with Kvasir-dataset
#### kvasir-dataset/first-implementation
**Description**:
- Uses pretrained Efficient Net V2-small model (trained on Imagenet) to classify 8 types of gastrointestinal conditions from endoscopy images, including polyps, ulcerative colitis, esophagitis, and normal tissue samples from the Kvasir medical imaging dataset.
- Replaces the final classification layer to adapt it for the 8 medical classes, allowing model to leverage existing visual features learned from millions of images.
- Divides the dataset into 80% training, 10% validation, and 10 % test sets with stratified splitting to maintain class balance. Training data gets augmented with random flips, rotations, and color adjustments to improve model generalization.
- Trains for 60 epochs using Adam optimizer with initial learning rate of 0.001, employing a learning rate scheduler that reduces the rate when validation accuracy plateaus, and saves the best-performing model based on validation accuracy.
- Evaluates the best model on the test set using multiple metrics (accuracy, precision, recall, F1-score) and generates a detailed classification report showing per-class performance for all 8 gastrointestinal conditions.

**Findings**

Overall performance:
- Test accuracy: 93.02%
- Precision: 93.02%
- Recall: 93%
- F1-Score: 93%

Per-Class Performance:

| Class                  | Precision | Recall | F1-Score | Support |
|-------------------------|------------|---------|-----------|----------|
| dyed-lifted-polyps      | 0.93       | 0.89    | 0.91      | 100      |
| dyed-resection-margins  | 0.90       | 0.93    | 0.92      | 100      |
| esophagitis             | 0.86       | 0.87    | 0.87      | 100      |
| normal-cecum            | 0.98       | 0.98    | 0.98      | 100      |
| normal-pylorus          | 0.98       | 1.00    | 0.99      | 100      |
| normal-z-line           | 0.87       | 0.87    | 0.87      | 100      |
| polyps                  | 0.97       | 0.94    | 0.95      | 100      |
| ulcerative-colitis      | 0.95       | 0.96    | 0.96      | 100      |
| **accuracy**            |            |         | **0.93**  | **800**  |
| **macro avg**           | 0.93       | 0.93    | 0.93      | 800      |
| **weighted avg**        | 0.93       | 0.93    | 0.93      | 800      |

- Perfect recall indicates no missed cases of normal pylorus.
- Normal-caecum shows second-best performance with consistently high metrics across all measures.
- Polyps have strong pathology detection with high precision minimizing false alarms. Clinically critical given polyps are precancerous lesions.
- Ulcerative colitis have excellent inflammatory disease detection.
- Dyed-resection-margins have a moderate performance. Post procedural images with artificial dye may present unique challenges. However, it still maintains > 90% accuracy.
- Dyed-lifted polyps have solid results. However, lower recall (89%)  suggests some missed detections. Dye-enhanced images may have greater visual variability.
- Esophagitis shows the lowest performance with 86% precision. 13% error rate may reflect subtle inflammatory changes. Also, the visual overlap with normal esophageal tissue could cause confusion.
- Normal-z line is facing similar challenges. It might be due to Z-line and esophagitis occuring in same anatomical region.


## Clinical Implications
- 94% recall for polyps and 96% for ulcerative colitis supports use in screening programs.
- Model could flag suspicious cases for physician review during routine colonoscopies and serve as a secondary check to catch findings overlooked by human readers.
- Near-perfect accuracy on normal structures (pylorus, cecum) assists in procedure documentation.
- 97 % precision minimizes false positives that could lead to unnecessary biopsies.
- Strong ulcerative colitis performance (96% recall) valuable for tracking disease progression.
- 93 % accuracy is impressive bt 7 % error rate requires human oversight.
- Should complement, not replace,expert clinical judgement.
- Early detection of polyps and inflammatory conditions may reduce healthcare costs through prevetion.
- Might not be that useful for esophagitis and z-line classification and require more dataset for this class.
- Transfer learning from Efficient V@ effectively adapted to medical domain.
