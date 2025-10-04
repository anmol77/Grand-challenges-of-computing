import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json

# PDF processing
import PyPDF2
import fitz  # PyMuPDF - better for complex PDFs

# NLP and embeddings
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Clustering and metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    pairwise_distances, classification_report, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setting seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS
# ============================================================================

# TODO: UPDATE THIS PATH - Path to your UC pathology PDF reports
PDF_DIRECTORY = "/home/agauta01/anmol_work/uc_textual_reports"  # <-- CHANGE THIS

# TODO: OPTIONAL - If you have ground truth labels (Mayo scores), provide CSV path
# CSV should have columns: 'filename', 'mayo_score' (0, 1, 2, or 3)
GROUND_TRUTH_CSV = None  # Set to "/path/to/labels.csv" if you have labels

# Output directory for results
OUTPUT_DIR = "./uc_clustering_results"

# Model configuration
# TODO: UPDATE THIS PATH - Path to your locally saved model directory
# Download model first on a machine with internet, then transfer to HPC
# See offline_model_setup guide for instructions
MODEL_PATH = "/home/agauta01/anmol_work/bioclinicalbert_local"  # <-- CHANGE THIS

# If you have internet access, you can use model name instead:
# MODEL_PATH = "emilyalsentzer/Bio_ClinicalBERT"

# Alternative models you can download and use:
# "dmis-lab/biobert-base-cased-v1.1" - BioBERT
# "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" - PubMedBERT
# "bert-base-uncased" - Standard BERT (often pre-installed on HPC)
# "sentence-transformers/all-MiniLM-L6-v2" - General purpose, faster, smaller

# Set to True if using local model files (no internet on HPC)
USE_OFFLINE_MODE = True  # <-- Set to True for HPC clusters without internet

# Processing parameters
MAX_LENGTH = 512  # Maximum token length for BERT
BATCH_SIZE = 8  # Adjust based on your GPU memory
NUM_CLUSTERS = 4  # Mayo 0, 1, 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# PDF TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using multiple methods for robustness.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text as string
    """
    text = ""
    
    # Method 1: Try PyMuPDF (better for complex PDFs)
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if text.strip():
            return text
    except Exception as e:
        print(f"PyMuPDF failed for {pdf_path}: {e}")
    
    # Method 2: Fallback to PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"PyPDF2 failed for {pdf_path}: {e}")
    
    return text

def preprocess_text(text):
    """
    Clean and preprocess extracted text.
    
    Args:
        text: Raw text from PDF
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep medical terminology
    # You can customize this based on your PDF format
    
    return text

def load_all_reports(pdf_directory):
    """
    Load and extract text from all PDF reports in directory.
    
    Args:
        pdf_directory: Directory containing PDF files
    
    Returns:
        Dictionary: {filename: extracted_text}
    """
    reports = {}
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    print(f"\nFound {len(pdf_files)} PDF files in {pdf_directory}")
    print("Extracting text from PDFs...")
    
    for idx, filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_directory, filename)
        try:
            text = extract_text_from_pdf(pdf_path)
            text = preprocess_text(text)
            
            if text.strip():
                reports[filename] = text
                print(f"[{idx}/{len(pdf_files)}] ✓ {filename} - {len(text)} characters")
            else:
                print(f"[{idx}/{len(pdf_files)}] ✗ {filename} - No text extracted")
        except Exception as e:
            print(f"[{idx}/{len(pdf_files)}] ✗ {filename} - Error: {e}")
    
    print(f"\nSuccessfully processed {len(reports)} reports")
    return reports

# ============================================================================
# EMBEDDING GENERATION WITH BioClinicalBERT
# ============================================================================

class BioClinicalBERTEmbedder:
    """
    Generate embeddings for medical text using BioClinicalBERT.
    """
    
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, max_length=MAX_LENGTH, offline=USE_OFFLINE_MODE):
        """
        Initialize the embedder.
        
        Args:
            model_path: Path to model (local directory or HuggingFace model name)
            device: CPU or CUDA
            max_length: Maximum token length
            offline: If True, only use local files (no internet download)
        """
        print(f"\nLoading model from {model_path} on {device}...")
        if offline:
            print("Running in OFFLINE mode (local files only)")
        
        self.device = device
        self.max_length = max_length
        
        try:
            # Load tokenizer and model
            if offline:
                # HPC cluster mode - use local files only
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            else:
                # Internet available - can download if needed
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\nTroubleshooting steps:")
            print("1. If on HPC without internet:")
            print("   - Download model on local machine first")
            print("   - Transfer to HPC cluster")
            print("   - Update MODEL_PATH to point to local directory")
            print("   - Set USE_OFFLINE_MODE = True")
            print("\n2. Check that the model directory contains:")
            print("   - config.json")
            print("   - pytorch_model.bin")
            print("   - tokenizer files (vocab.txt, tokenizer_config.json)")
            print(f"\n3. Verify path exists: {model_path}")
            raise
    
    def get_embedding(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            768-dimensional embedding vector (for BERT-base models)
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use [CLS] token embedding (first token)
            # This represents the entire sequence
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]  # Return 1D array
    
    def get_batch_embeddings(self, texts, batch_size=BATCH_SIZE):
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        print(f"\nGenerating embeddings for {len(texts)} reports...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
            
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} reports")
        
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings

# ============================================================================
# CLUSTERING AND ANALYSIS
# ============================================================================

def perform_clustering(embeddings, n_clusters=NUM_CLUSTERS, method='kmeans'):
    """
    Cluster embeddings into groups.
    
    Args:
        embeddings: numpy array of embeddings
        n_clusters: Number of clusters (4 for Mayo scores)
        method: 'kmeans' or 'hierarchical'
    
    Returns:
        Cluster labels
    """
    print(f"\nPerforming {method} clustering into {n_clusters} groups...")
    
    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,  # Multiple initializations for stability
            max_iter=300
        )
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Ward minimizes within-cluster variance
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    labels = clusterer.fit_predict(embeddings)
    
    print(f"Clustering complete!")
    print(f"Cluster distribution: {np.bincount(labels)}")
    
    return labels, clusterer

def calculate_clustering_metrics(embeddings, labels):
    """
    Calculate various clustering quality metrics.
    
    Args:
        embeddings: numpy array of embeddings
        labels: Cluster labels
    
    Returns:
        Dictionary of metrics
    """
    print("\nCalculating clustering metrics...")
    
    metrics = {}
    
    # Silhouette Score (higher is better, range: -1 to 1)
    # Measures how similar an object is to its own cluster vs other clusters
    metrics['silhouette_score'] = silhouette_score(embeddings, labels)
    
    # Davies-Bouldin Index (lower is better)
    # Ratio of within-cluster to between-cluster distances
    metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
    
    # Calinski-Harabasz Score (higher is better)
    # Ratio of between-cluster to within-cluster dispersion
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
    
    # Calculate within-cluster and between-cluster distances
    unique_labels = np.unique(labels)
    within_cluster_distances = []
    
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        if len(cluster_points) > 1:
            # Average pairwise distance within cluster
            distances = pairwise_distances(cluster_points, metric='cosine')
            within_cluster_distances.append(np.mean(distances[np.triu_indices_from(distances, k=1)]))
    
    metrics['avg_within_cluster_distance'] = np.mean(within_cluster_distances)
    
    # Calculate cluster centroids and between-cluster distances
    centroids = []
    for label in unique_labels:
        centroids.append(np.mean(embeddings[labels == label], axis=0))
    centroids = np.array(centroids)
    
    between_distances = pairwise_distances(centroids, metric='cosine')
    metrics['avg_between_cluster_distance'] = np.mean(between_distances[np.triu_indices_from(between_distances, k=1)])
    
    return metrics

def calculate_cosine_similarity_matrix(embeddings):
    """
    Calculate pairwise cosine similarity between all embeddings.
    
    Args:
        embeddings: numpy array of embeddings
    
    Returns:
        Cosine similarity matrix
    """
    print("\nCalculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def map_clusters_to_mayo_scores(labels, embeddings, ground_truth=None):
    """
    Map cluster IDs to Mayo severity scores (0-3).
    
    If ground truth is available, use it to map.
    Otherwise, use heuristics based on cluster characteristics.
    
    Args:
        labels: Cluster labels
        embeddings: Embedding vectors
        ground_truth: Optional dictionary {filename: mayo_score}
    
    Returns:
        Dictionary mapping cluster_id -> mayo_score
    """
    print("\nMapping clusters to Mayo severity scores...")
    
    if ground_truth:
        # Use ground truth to determine mapping
        cluster_to_mayo = {}
        for cluster_id in np.unique(labels):
            cluster_filenames = [f for i, f in enumerate(ground_truth.keys()) if labels[i] == cluster_id]
            cluster_mayo_scores = [ground_truth[f] for f in cluster_filenames]
            # Assign most common Mayo score in cluster
            cluster_to_mayo[cluster_id] = max(set(cluster_mayo_scores), key=cluster_mayo_scores.count)
    else:
        # Heuristic: Order clusters by average distance from origin
        # Assumption: More severe cases might have different embedding magnitudes
        cluster_centroids = []
        for cluster_id in np.unique(labels):
            centroid = np.mean(embeddings[labels == cluster_id], axis=0)
            cluster_centroids.append((cluster_id, np.linalg.norm(centroid)))
        
        # Sort by magnitude and assign Mayo scores
        cluster_centroids.sort(key=lambda x: x[1])
        cluster_to_mayo = {cluster_id: mayo_score for mayo_score, (cluster_id, _) in enumerate(cluster_centroids)}
        
        print("Note: Without ground truth, Mayo scores are assigned heuristically.")
        print("Consider validating with clinical experts or labeled data.")
    
    return cluster_to_mayo

def evaluate_with_ground_truth(labels, ground_truth_dict, filenames, cluster_to_mayo):
    """
    Evaluate clustering performance against ground truth Mayo scores.
    
    Args:
        labels: Predicted cluster labels
        ground_truth_dict: Dictionary {filename: true_mayo_score}
        filenames: List of filenames in order
        cluster_to_mayo: Mapping from cluster to Mayo score
    
    Returns:
        Metrics dictionary
    """
    print("\n" + "="*70)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("="*70)
    
    # Convert cluster labels to Mayo scores
    predicted_mayo = np.array([cluster_to_mayo[label] for label in labels])
    true_mayo = np.array([ground_truth_dict[f] for f in filenames])
    
    # Calculate metrics
    report = classification_report(
        true_mayo,
        predicted_mayo,
        target_names=['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'],
        output_dict=True,
        zero_division=0
    )
    
    # Print classification report
    print("\nClassification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())
    
    # Confusion matrix
    cm = confusion_matrix(true_mayo, predicted_mayo, labels=[0, 1, 2, 3])
    
    # Calculate additional metrics
    accuracy = np.mean(predicted_mayo == true_mayo)
    
    metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'confusion_matrix': cm,
        'classification_report': report_df
    }
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall: {metrics['recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_score']:.4f}")
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(embeddings, labels, cluster_to_mayo, filenames, output_dir):
    """
    Create visualizations of clustering results.
    
    Args:
        embeddings: Embedding vectors
        labels: Cluster labels
        cluster_to_mayo: Mapping from cluster to Mayo score
        filenames: List of filenames
        output_dir: Directory to save plots
    """
    print("\nGenerating visualizations...")
    
    # Convert labels to Mayo scores for coloring
    mayo_scores = np.array([cluster_to_mayo[label] for label in labels])
    
    # 1. PCA visualization (2D)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=mayo_scores,
        cmap='RdYlGn_r',  # Red (severe) to Green (mild)
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    plt.colorbar(scatter, label='Mayo Score', ticks=[0, 1, 2, 3])
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('UC Pathology Reports - PCA Visualization\n(Colored by Mayo Severity Score)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300)
    plt.close()
    
    # 2. t-SNE visualization (2D)
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_tsne[:, 0],
        embeddings_tsne[:, 1],
        c=mayo_scores,
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    plt.colorbar(scatter, label='Mayo Score', ticks=[0, 1, 2, 3])
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('UC Pathology Reports - t-SNE Visualization\n(Colored by Mayo Severity Score)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
    
    # 3. Cosine similarity heatmap
    similarity_matrix = cosine_similarity(embeddings)
    
    # Sort by Mayo scores for better visualization
    sorted_indices = np.argsort(mayo_scores)
    sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_mayo = mayo_scores[sorted_indices]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sorted_similarity,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Cosine Similarity Matrix\n(Sorted by Mayo Score)')
    plt.xlabel('Reports (sorted by Mayo score)')
    plt.ylabel('Reports (sorted by Mayo score)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=300)
    plt.close()
    
    # 4. Cluster distribution
    plt.figure(figsize=(10, 6))
    mayo_counts = np.bincount(mayo_scores, minlength=4)
    colors = ['green', 'yellow', 'orange', 'red']
    plt.bar(['Mayo 0\n(Normal)', 'Mayo 1\n(Mild)', 'Mayo 2\n(Moderate)', 'Mayo 3\n(Severe)'],
            mayo_counts, color=colors, edgecolor='black', alpha=0.7)
    plt.ylabel('Number of Reports')
    plt.title('Distribution of UC Severity Scores')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mayo_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline for UC pathology report clustering.
    """
    print("="*70)
    print("UC PATHOLOGY REPORT CLUSTERING WITH BioClinicalBERT")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Offline mode: {USE_OFFLINE_MODE}")
    print(f"Number of clusters: {NUM_CLUSTERS}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load PDF reports
    print("\n" + "="*70)
    print("STEP 1: LOADING PDF REPORTS")
    print("="*70)
    reports_dict = load_all_reports(PDF_DIRECTORY)
    
    if len(reports_dict) == 0:
        print("\nError: No PDF reports found or processed successfully.")
        print(f"Please check the PDF_DIRECTORY path: {PDF_DIRECTORY}")
        return
    
    filenames = list(reports_dict.keys())
    texts = list(reports_dict.values())
    
    # Save extracted texts
    with open(os.path.join(OUTPUT_DIR, 'extracted_texts.json'), 'w') as f:
        json.dump(reports_dict, f, indent=2)
    
    # Step 2: Generate embeddings
    print("\n" + "="*70)
    print("STEP 2: GENERATING EMBEDDINGS")
    print("="*70)
    embedder = BioClinicalBERTEmbedder()
    embeddings = embedder.get_batch_embeddings(texts)
    
    # Save embeddings
    np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
    
    # Step 3: Perform clustering
    print("\n" + "="*70)
    print("STEP 3: CLUSTERING")
    print("="*70)
    labels, clusterer = perform_clustering(embeddings)
    
    # Step 4: Calculate metrics
    print("\n" + "="*70)
    print("STEP 4: CALCULATING METRICS")
    print("="*70)
    metrics = calculate_clustering_metrics(embeddings, labels)
    
    print("\nClustering Quality Metrics:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, range: -1 to 1)")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
    print(f"  Avg Within-Cluster Distance: {metrics['avg_within_cluster_distance']:.4f}")
    print(f"  Avg Between-Cluster Distance: {metrics['avg_between_cluster_distance']:.4f}")
    
    # Step 5: Load ground truth if available
    ground_truth_dict = None
    if GROUND_TRUTH_CSV and os.path.exists(GROUND_TRUTH_CSV):
        print("\n" + "="*70)
        print("STEP 5: LOADING GROUND TRUTH")
        print("="*70)
        gt_df = pd.read_csv(GROUND_TRUTH_CSV)
        ground_truth_dict = dict(zip(gt_df['filename'], gt_df['mayo_score']))
        print(f"Loaded ground truth for {len(ground_truth_dict)} reports")
    
    # Step 6: Map clusters to Mayo scores
    print("\n" + "="*70)
    print("STEP 6: MAPPING TO MAYO SCORES")
    print("="*70)
    cluster_to_mayo = map_clusters_to_mayo_scores(labels, embeddings, ground_truth_dict)
    
    print("\nCluster to Mayo Score Mapping:")
    for cluster_id, mayo_score in sorted(cluster_to_mayo.items()):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id} -> Mayo {mayo_score} ({count} reports)")
    
    # Step 7: Evaluate if ground truth available
    if ground_truth_dict:
        eval_metrics = evaluate_with_ground_truth(labels, ground_truth_dict, filenames, cluster_to_mayo)
        metrics.update(eval_metrics)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(
            eval_metrics['confusion_matrix'],
            index=['True Mayo 0', 'True Mayo 1', 'True Mayo 2', 'True Mayo 3'],
            columns=['Pred Mayo 0', 'Pred Mayo 1', 'Pred Mayo 2', 'Pred Mayo 3']
        )
        cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(eval_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'],
                    yticklabels=['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'])
        plt.ylabel('True Mayo Score')
        plt.xlabel('Predicted Mayo Score')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    # Step 8: Create results dataframe
    print("\n" + "="*70)
    print("STEP 8: SAVING RESULTS")
    print("="*70)
    
    mayo_scores = [cluster_to_mayo[label] for label in labels]
    results_df = pd.DataFrame({
        'filename': filenames,
        'cluster_id': labels,
        'mayo_score': mayo_scores,
        'text_length': [len(text) for text in texts]
    })
    
    if ground_truth_dict:
        results_df['true_mayo_score'] = [ground_truth_dict.get(f, None) for f in filenames]
        results_df['correct'] = results_df['mayo_score'] == results_df['true_mayo_score']
    
    results_df = results_df.sort_values('mayo_score')
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'clustering_results.csv'), index=False)
    
    # Step 9: Visualizations
    print("\n" + "="*70)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("="*70)
    visualize_results(embeddings, labels, cluster_to_mayo, filenames, OUTPUT_DIR)
    
    # Step 10: Save all metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)
    
    # Save detailed report
    with open(os.path.join(OUTPUT_DIR, 'clustering_report.txt'), 'w') as f:
        f.write("UC PATHOLOGY REPORT CLUSTERING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Offline mode: {USE_OFFLINE_MODE}\n")
        f.write(f"Number of reports: {len(filenames)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"Number of clusters: {NUM_CLUSTERS}\n\n")
        
        f.write("CLUSTERING QUALITY METRICS:\n")
        f.write(f"  Silhouette Score: {metrics['silhouette_score']:.4f}\n")
        f.write(f"  Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}\n")
        f.write(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}\n\n")
        
        if ground_truth_dict:
            f.write("CLASSIFICATION METRICS:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
        
        f.write("CLUSTER DISTRIBUTION:\n")
        for mayo_score in range(4):
            count = np.sum(np.array(mayo_scores) == mayo_score)
            percentage = (count / len(mayo_scores)) * 100
            f.write(f"  Mayo {mayo_score}: {count} reports ({percentage:.1f}%)\n")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - clustering_results.csv: Main results with Mayo scores")
    print("  - embeddings.npy: BioClinicalBERT embeddings")
    print("  - extracted_texts.json: Extracted PDF texts")
    print("  - metrics_summary.csv: All clustering metrics")
    print("  - clustering_report.txt: Detailed text report")
    print("  - pca_visualization.png: 2D PCA plot")
    print("  - tsne_visualization.png: 2D t-SNE plot")
    print("  - similarity_heatmap.png: Cosine similarity matrix")
    print("  - mayo_distribution.png: Distribution bar chart")
    if ground_truth_dict:
        print("  - confusion_matrix.csv: Confusion matrix")
        print("  - confusion_matrix.png: Confusion matrix heatmap")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total reports processed: {len(filenames)}")
    print(f"Mayo 0 (Normal/Quiescent): {np.sum(np.array(mayo_scores) == 0)} reports")
    print(f"Mayo 1 (Mild): {np.sum(np.array(mayo_scores) == 1)} reports")
    print(f"Mayo 2 (Moderate): {np.sum(np.array(mayo_scores) == 2)} reports")
    print(f"Mayo 3 (Severe): {np.sum(np.array(mayo_scores) == 3)} reports")
    
    if ground_truth_dict:
        print(f"\nClassification Accuracy: {metrics['accuracy']:.2%}")
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'cluster_to_mayo': cluster_to_mayo,
        'metrics': metrics,
        'results_df': results_df
    }

if __name__ == "__main__":
    results = main()