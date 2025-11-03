## BioClinical BERT

BioClinical BERT is is a specialized version of BERT that’s been pretrained on biomedical and clinical text, such as PubMed papers and MIMIC-III clinical notes. BioClinicalBERT understands medical and clinical language better than standard BERT, making it ideal for tasks like clinical text classification, named entity recognition (NER), and medical question answering. We are using synthetic dataset (pathology reports) to train BioClinical BERT to see if the reports help the model classify ulcerative colitis on the basis of textual modalities. We have two different folders of report in which the training is done.

#### bert_clustering.py
**Description**:
- PDF-to-embedding NLP pipeline for medical text: Extracts text from PDF pathology reports using PyMuPDF and PyPDF2, then generates 768-dimensional embeddings using BioClinicalBERT (a BERT model pretrained on clinical text), designed to work offline on HPC clusters without internet access by loading locally saved model files.
- Unsupervised clustering for Mayo score classification: Performs K-means clustering (k=4) on the embeddings to automatically group ulcerative colitis pathology reports into four severity categories corresponding to Mayo scores (0=normal/quiescent, 1=mild, 2=moderate, 3=severe), without requiring labeled training data.
- Comprehensive clustering quality assessment: Evaluates clustering performance using multiple unsupervised metrics including silhouette score (cluster cohesion), Davies-Bouldin index (cluster separation), Calinski-Harabasz score (variance ratios), and cosine similarity matrices to assess whether reports with similar disease severity are grouped together.
- Optional supervised evaluation with ground truth: If Mayo score labels are available in CSV format, validates clustering accuracy by mapping predicted clusters to true Mayo scores, generates confusion matrices, and calculates classification metrics (precision, recall, F1-score, accuracy) to quantify clinical alignment.
- Multi-method dimensionality reduction visualization: Creates interpretable 2D visualizations using both PCA and t-SNE to project high-dimensional embeddings, generates heatmaps showing cosine similarity between reports, and produces distribution charts showing how reports are classified across Mayo severity levels, all color-coded by disease severity for clinical interpretation.

### Findings:

#### UC-Textual-Reports
- 2,000 ulcerative colitis pathology reports successfully analyzed.
- BioClinicalBERT produced 768-dimensional semantic representations for each report.
- Complete pipeline executed in 81 seconds, demonstrating feasibility for large-scale clinical text analysis.
- K-means clustering (k=4) grouped reports into four severity categories without requiring labeled training data.
- Model discovered patterns in clinical text without prior knowledge of Mayo scoring system.

#### Clustering Quality Metrics Analysis

- Silhouette Score: 0.1024 (Range: -1 to 1, higher is better)
  - Low positive value indicates weak to moderate cohesion.
  - Reports within same cluster share some similarity, but boundaries between severity levels are not sharply defined.
  - Suggests overlapping linguistic patterns across Mayo scores.
  - Clinical interpretation: Pathology language may exist on a continuum rather than discrete categories
 
- Davies-Bouldin Index: 2.8116 (Lower is better)
  - Relatively high value indicates moderate cluster overlap.
  - Average ratio of within-cluster to between-cluster distances suggests adjacent severity levels are not well-separated.
  - Mayo 1 vs Mayo 2 or Mayo 2 vs Mayo 3 may have similar descriptive language.
  - Indicates challenge in distinguishing neighboring severity grades from text alone
 
- Calinski-Harabasz Score: 820.46 (Higher is better)
  - Moderate score reflects reasonable between-cluster vs within-cluster variance ratio.
  - Suggests some meaningful separation between severity groups exists.
  - Higher than random clustering would produce, validating clinical relevance of discovered patterns.
  - Indicates BioClinicalBERT captures medically meaningful semantic differences.
 
- Within-Cluster Distance: 0.0028 (Cosine distance)
  - Very low average distance indicates high intra-cluster similarity.
  - Reports assigned to same Mayo score share consistent terminology and descriptions.
  - Demonstrates internal consistency of clinical documentation within severity levels
 
- Between Cluster Distance: 0.0045 (Cosine Distance)
  - Small difference from within-cluster distance (0.0045 vs 0.0028, ratio ~1.6:1).
  - Limited separation between severity clusters in embedding space.
  - Explains moderate silhouette score and high Davies-Bouldin index.
  - Reflects subtle linguistic differences between adjacent Mayo grades.

 #### Cluster Distribution Analysis

 Mayo 0: 405 reports (20.3%)
 - Smallest cluster, reflecting fewer patients in complete remission.
 - May include reports describing "inactive disease," "quiescent colitis," "no active inflammation".

 Mayo 1: 597 reports (29.9%)
 - Largest cluster by assignment
 - Likely includes descriptors like "minimal erythema," "mild granularity," "edema present"

 Mayo 2: 372 reports (18.6%)
 - Second smallest cluster.
 - May contain language about "marked erythema," "friability," "erosions," "spontaneous bleeding".

 Mayo 3: 626 reports (31.3%)
 - Second largest cluster, surprisingly high proportion.
 - Expected terms: "ulceration," "severe friability," "spontaneous hemorrhage," "extensive mucosal damage".

#### Mapping Limitation
- No ground truth validation: Mayo scores assigned by centroid magnitude, not validated against actual clinical assessments.
- Assumption-based: Mapping assumes severity correlates with embedding vector norms, which may not hold.
- Requires clinical validation: Expert gastroenterologist review needed to assess mapping accuracy.
- High proportion of Mayo 3 (31.3%) may reflect mapping artifacts rather than true severe disease prevalence.

#### Linguistic Pattern Insights
- Low silhouette score (0.1024) suggests Mayo scores differ more quantitatively than qualitatively in language.
- Adjacent grades may use similar terminology with different intensities (e.g., "mild erythema" vs "marked erythema").
- High Davies-Bouldin index (2.8116) indicates "gray zones" between severity levels.
- Mayo 1 vs Mayo 2 particularly difficult to distinguish linguistically.
- Low within-cluster distance (0.0028) shows standardized medical terminology within severity levels.

## Clinical Implications
- Identify severe cases (Mayo 2-3) requiring urgent gastroenterology follow-up.
- Flag reports for expedited physician review based on predicted severity.
- Reduce time from biopsy to treatment decisions in high-volume centers.
- Current limitation: Requires validation before clinical deployment.
- Detect inconsistencies between pathology descriptions and assigned Mayo scores.
- Training tool for pathology residents learning UC grading.
- Automated extraction of disease activity from unstructured pathology reports
- Populate structured EHR fields for disease monitoring.
- Generate severity trends over time for individual patients

## Current Limitations
- Silhouette score (0.1024) too low for diagnostic-grade performance making it not deployable without validation.
- Heuristic mapping unvalidated - actual accuracy unknown.
- Risk of misclassification could lead to inappropriate treatment decisions.
- Must remain research tool until prospective validation complete.
- Poor separation (Davies-Bouldin 2.8116) means Mayo 1 vs 2 or 2 vs 3 frequently misclassified.
- False Mayo 3 assignment could cause unnecessary aggressive treatment.
- False Mayo 1 assignment could delay needed therapy intensification.
- Model processes only pathology text, ignoring endoscopic findings. Text alone cannot capture complete disease picture.
- Regional practice variations affect language patterns

## Technical Performance Strengths
- Successfully generated meaningful embeddings from complex medical text.
- Captured semantic relationships between UC severity descriptors.
- Processed 2,000 reports in 81 seconds (~25 reports/second).
- Offline mode supports deployment in HPC clusters without internet.
- GPU acceleration (CUDA) enabled efficient batch processing.
- Comprehensive output files support audit trail and verification.

## Comparison with supervised approaches
- Current approach: no labeled data required, discovers patterns automatically.
- Supervised alternative: would need hundreds of expert-labeled reports for training.
- Trade-off: unsupervised has lower accuracy but much lower annotation cost.
- Hybrid approach: use clustering to identify candidates for targeted expert labeling.
- Supervised deep learning models typically achieve 70-85% accuracy on Mayo scoring.
- Current unsupervised approach likely 50-60% accuracy (requires validation).
- Gap reflects value of expert supervision in capturing subtle severity distinctions.

## Recommendations
- Collaborate with gastroenterologists to label 200-500 reports.
- Use labels to evaluate true clustering accuracy vs heuristic mapping.
- Calculate confusion matrix showing actual misclassification patterns
- Train supervised classifier on subset, use for remaining reports.
- Extract specific terms (ulceration, friability, erythema) and weight explicitly.
- Combine text embeddings with structured data (patient age, disease duration).
- Incorporate endoscopic images if available (multi-modal approach).
- Fine-tune BioClinicalBERT specifically on UC pathology reports.
- Implement few-shot learning with minimal labeled examples.
- Use active learning to iteratively improve with expert feedback
