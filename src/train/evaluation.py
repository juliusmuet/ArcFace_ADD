# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
os.environ["NUMBA_NUM_THREADS"] = "1"   # Prevent UMAP from using all cores / threads
import re
import random
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score
from sklearn.preprocessing import normalize
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datasets.dataset_evaluation import TrialDataset
from utils.utils import get_elapsed_remaining_time_from_progress_bar

logger = logging.getLogger(__name__)
LOGGING_INT = 500


def calculate_eer(scores, labels):
    """
    Calculates the Equal Error Rate (EER) and the threshold at which it occurs.

    EER is the point on the ROC curve where the False Positive Rate (FPR)
    equals the False Negative Rate (FNR).

    Args:
        scores (numpy.ndarray): An array of similarity scores for the trials.
        labels (numpy.ndarray): An array of true labels (0 for non-target, 1 for target)
                                for the trials.

    Returns:
        tuple: A tuple containing:
            - float: The Equal Error Rate (EER).
            - float: The threshold value at which EER occurs.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)    # Calculate ROC curve: False Positive Rate (fpr), True Positive Rate (tpr)
    fnr = 1 - tpr   # False Negative Rate (fnr) is 1 - tpr
    abs_diffs = np.abs(fpr - fnr)   # Find the point where fpr is closest to fnr / find the minimum of the absolute difference |fpr - fnr|
    min_index = np.argmin(abs_diffs)
    eer = fpr[min_index]    # EER is the value of fpr or fnr at this point
    eer_threshold = thresholds[min_index]   # The threshold at which this EER occur
    return eer, eer_threshold


def _calculate_accuracy(scores, is_target_trial, eer_threshold):
    """
    Calculates the classification accuracy for a set of trials based on a decision threshold.

    For target trials (i.e., where the speaker identity matches and the sample is bonafide), 
    a trial is considered correct if its score is greater than or equal to the threshold.
    
    For non-target trials (e.g., genuine or deepfake impostors), a trial is correct if 
    its score is below the threshold.

    Args:
        scores (list of float): List of similarity scores for the trials.
        is_target_trial (bool): If True, treats trials as target (same speaker).
                                If False, treats trials as non-target (different speakers or fake).
        eer_threshold (float): The threshold score used for decision-making from EER.

    Returns:
        float: Accuracy as a percentage of correctly classified trials.
    """
    if not scores: return 0.0
    if is_target_trial:
        correct = sum(1 for s in scores if s >= eer_threshold)
    else:
        correct = sum(1 for s in scores if s < eer_threshold)
    return (correct / len(scores)) * 100


def _calculate_precision_recall(scores, labels, threshold):
    """
    Calculate precision and recall based on predicted scores, true labels and a threshold.

    Parameters:
    - scores (list or array-like): Predicted scores or probabilities for the positive class.
    - labels (list or array-like): True binary labels (0 or 1).
    - threshold (float): Threshold to convert scores into binary predictions.

    Returns:
    - precision (float): The ratio of true positives to predicted positives.
    - recall (float): The ratio of true positives to actual positives.
    """
    # Convert scores to binary predictions
    predictions = [1 if s >= threshold else 0 for s in scores]

    # Calculate precision and recall 
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    return precision, recall


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculates the normalised cosine similarity between two embeddings.

    Cosine similarity is computed using PyTorch's F.cosine_similarity,
    which measures the cosine of the angle between the two vectors.
    The result is in the range [-1, 1], where 1 means perfectly similar,
    and -1 means opposite. This function normalises the result to [0, 1]
    for compatibility with probability-based scoring and fusion.

    Args:
        embedding1 (torch.Tensor): A 1D tensor representing the first embedding.
        embedding2 (torch.Tensor): A 1D tensor representing the second embedding.

    Returns:
        float: Normalised cosine similarity score in the range [0, 1].
               Higher values indicate higher similarity.
    """
    cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
    return (cosine_similarity + 1) / 2


def score_fusion(fusion_mode, cosine_similarity, genuineness_prob, weighted_sum_alpha=0.5):
    """
    Computes a fused score from cosine similarity and a classifier-based genuineness probability.

    Args:
        fusion_mode (str): Fusion strategy to use. One of:
            - 'embedding_only': Returns cosine similarity only.
            - 'classifier_only': Returns classifier genuineness probability only.
            - 'multiplication': Returns the product of cosine similarity and genuineness probability.
            - 'weighted_sum': Returns a weighted sum of cosine similarity and genuineness probability.
        cosine_similarity (float): Cosine similarity between two embeddings.
        genuineness_prob (float): Probability output from a classifier indicating how genuine the input is.
        weighted_sum_alpha (float, optional): Weight assigned to cosine similarity in the weighted sum fusion.
                                              Only used when fusion_mode == 'weighted_sum'. Default is 0.5.

    Returns:
        float: The fused score according to the selected fusion mode.

    Raises:
        ValueError: If an invalid fusion_mode is provided.

    Notes:
        - If genuineness_prob and fusion_mode is not 'embedding_only', cosine similarity with a warning is returned.
    """
    if fusion_mode == 'embedding_only':
        return cosine_similarity
    
    if genuineness_prob is None:
        logger.warning("genuineness_prob is None. Returning cosine_similarity!")
        return cosine_similarity
    
    if fusion_mode == 'classifier_only':
        return genuineness_prob
    if fusion_mode == 'multiplication':
        return cosine_similarity * genuineness_prob
    if fusion_mode == 'weighted_sum':
        return weighted_sum_alpha * cosine_similarity + (1 - weighted_sum_alpha) * genuineness_prob
    
    raise ValueError(f"Invalid fusion mode: {fusion_mode}")


def evaluate_speaker_verification_spoofed(model: torch.nn.Module, datasets: List[TrialDataset], device: torch.device, score_fusion_mode: str = 'embedding_only', weighted_sum_alpha: float = 0.5, visualise_save_path=None):
    """
    Evaluates a speaker verification model for deepfake detection by first computing all
    embeddings in batches and then computing pairwise scores for all score fusion modes.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        datasets (List[TrialDataset]): List of datasets containing the trials and unique paths for all audio files.
                                       Trials must be in lines: path1, path2, label where label is "target", "nontarget" or "spoof".
        device (torch.device): The device to run the model on.
        batch_size (int): The batch size for GPU inference. Tune based on VRAM.
        score_fusion_mode (str or List[str]): Fusion method (default: 'embedding_only'). One or multiple of: 
                                              'embedding_only', 'classifier_only', 'multiplication', 'weighted_sum'.
        weighted_sum_alpha (float): Weight for cosine similarity in weighted sum fusion (default: 0.5).
                                    Used only when score_fusion_mode == 'weighted_sum'.
        visualise_save_path (str): Path to the directory to save visualised embeddings (default: None).
                                   If None, embeddings are not visualised.
    
    Returns:
        dict: A dictionary mapping each evaluated score fusion mode to its corresponding
              Equal Error Rate (EER) and threshold.                               
    """
    # Log score fusion modes used
    if isinstance(score_fusion_mode, str):
        score_fusion_mode = [score_fusion_mode]
    logger.info(f"Evaluation with modes: {score_fusion_mode}" +
          (f", weighted_sum_alpha={weighted_sum_alpha}" if 'weighted_sum' in score_fusion_mode else "") +
          f" and {len(datasets)} datasets")

    model.to(device)
    model.eval()

    results = {}

    # Evaluate for each given dataset
    for dataset_idx, dataset in enumerate(datasets):
        logger.info(f"Evalution with trials: {dataset.trial_file}")
        dataloader = dataset.get_dataloader()
        unique_paths = dataset.paths
        trials = dataset.trials

        # Calculate embeddings
        embeddings_dict = {}
        with torch.no_grad():
            progress_bar_inference = tqdm(dataloader, desc="Batch Inference")
            for idx, audio in enumerate(progress_bar_inference):
                audio = audio.to(device)
                emb_batch, gen_score_batch = model(audio)

                embeddings_dict.update(zip(
                    unique_paths[len(embeddings_dict):len(embeddings_dict) + len(emb_batch)],
                    list(zip(emb_batch.cpu(), gen_score_batch.cpu() if gen_score_batch is not None else [None] * len(emb_batch)))
                ))

                if idx % LOGGING_INT == 0:
                    elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar_inference)
                    logger.info(f"Batch Inference: Elapsed: {elapsed_str} | ETA: {remaining_str}")

        # Initialize dictionaries per fusion mode
        mode_scores = {mode: [] for mode in score_fusion_mode}  # overall scores per score fusion mode
        mode_labels = {mode: [] for mode in score_fusion_mode}  # overall labels per score fusion mode 
        mode_trials = {mode: {'target': [], 'nontarget': [], 'spoof': []} for mode in score_fusion_mode}    # scores per label per score fusion mode

        # Save speakers and embeddings for visualising
        speakers = []
        embeddings = []
        genuine_label = []

        # Calculate scores
        progress_bar_trials = tqdm(trials, desc="Scoring Trials")
        for idx, (path1, path2, label) in enumerate(progress_bar_trials):
            embedding1, _ = embeddings_dict[path1]
            embedding2, genuineness_score2 = embeddings_dict[path2]

            spk1 = Path(path1).parts[1]
            spk2 = Path(path2).parts[1]
            speakers.append(spk1)
            speakers.append(spk2 + '_df' if label == 'spoof' else spk2)
            embeddings.append(embedding1)
            embeddings.append(embedding2)
            genuine_label.append(1)
            genuine_label.append(0 if label == 'spoof' else 1)

            cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)
            is_target = 1 if label == 'target' else 0

            for mode in score_fusion_mode:
                # If mode requires genuineness score but it's not available, skip mode
                if genuineness_score2 is None and mode != 'embedding_only':
                    continue

                genuineness_prob = torch.sigmoid(genuineness_score2).item() if genuineness_score2 is not None else None
                final_score = score_fusion(mode, cosine_similarity, genuineness_prob, weighted_sum_alpha)

                mode_scores[mode].append(final_score)
                mode_labels[mode].append(is_target)

                if label == 'target':
                    mode_trials[mode]['target'].append(final_score)
                elif label == 'nontarget':
                    mode_trials[mode]['nontarget'].append(final_score)
                elif label == 'spoof':
                    mode_trials[mode]['spoof'].append(final_score)
            
            """
            if idx % LOGGING_INT == 0:
                elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar_trials)
                logger.info(f"Scoring Trials: Elapsed: {elapsed_str} | ETA: {remaining_str}")
            """

        # Compute and return results
        results[dataset_idx] = {}
        logger.info(f"Computing Metrics...")
        for mode in tqdm(score_fusion_mode, desc="Computing Metrics"):
            if not mode_scores[mode]:
                logger.info(f"\nMode: {mode} — Skipping (no usable trials)")
                continue

            if mode == 'classifier_only':
                cl_scores = mode_trials[mode]['target'] +  mode_trials[mode]['nontarget'] + mode_trials[mode]['spoof']
                cl_labels = [1] * len(mode_trials[mode]['target']) + [1] * len(mode_trials[mode]['nontarget']) + [0] * len(mode_trials[mode]['spoof'])
                eer, eer_threshold = calculate_eer(np.array(cl_scores), np.array(cl_labels))
                
                genuine_scores = mode_trials[mode]['target'] + mode_trials[mode]['nontarget']
                deepfake_scores = mode_trials[mode]['spoof']
                cl_genuine_acc = _calculate_accuracy(genuine_scores, True, eer_threshold)
                cl_deepfake_acc = _calculate_accuracy(deepfake_scores, False, eer_threshold)

                cl_precision, cl_recall = _calculate_precision_recall(cl_scores, cl_labels, eer_threshold)

                logger.info(f"Mode: {mode}")
                logger.info(f"  EER: {eer*100:.4f}% with threshold {eer_threshold:.6f}")
                logger.info(f"  Genuine (target + nontarget) Accuracy: {cl_genuine_acc:.4f}% ({len(np.array(genuine_scores))} trials)")
                logger.info(f"  Spoof Accuracy: {cl_deepfake_acc:.4f}% ({len(np.array(deepfake_scores))} trials)")
                logger.info(f"  Precision: {cl_precision:.4f}")
                logger.info(f"  Recall: {cl_recall:.4f}")

                result_entry = {
                    'eer': eer,
                    'threshold': eer_threshold,
                    'sv_eer': 1.0,
                    'sv_threshold': 0.0,
                    'spf_eer': eer,
                    'spf_threshold': eer_threshold,
                    'genuine_acc': cl_genuine_acc,
                    'nontarget_acc': 0.0,
                    'deepfake_acc': cl_deepfake_acc,
                    'precision': cl_precision,
                    'recall': cl_recall,
                    'sv_precision': 0.0,
                    'sv_recall': 0.0,
                    'spf_precision': cl_precision,
                    'spf_recall': cl_recall
                }

                results[dataset_idx][mode] = result_entry
            
            else:
                sv_scores = mode_trials[mode]['target'] + mode_trials[mode]['nontarget']
                sv_labels = [1] * len(mode_trials[mode]['target']) + [0] * len(mode_trials[mode]['nontarget'])
                spf_scores = mode_trials[mode]['target'] + mode_trials[mode]['spoof']
                spf_labels = [1] * len(mode_trials[mode]['target']) + [0] * len(mode_trials[mode]['spoof'])

                if len(mode_trials[mode]['nontarget']) == 0:
                    sv_eer, sv_eer_threshold = None, None
                    sv_precision, sv_recall = None, None
                else:
                    sv_eer, sv_eer_threshold = calculate_eer(np.array(sv_scores), np.array(sv_labels))
                    sv_precision, sv_recall = _calculate_precision_recall(sv_scores, sv_labels, sv_eer_threshold)
                if len(mode_trials[mode]['spoof']) == 0:
                    spf_eer, spf_eer_threshold = None, None
                    spf_precision, spf_recall = None, None
                else:
                    spf_eer, spf_eer_threshold = calculate_eer(np.array(spf_scores), np.array(spf_labels))
                    spf_precision, spf_recall = _calculate_precision_recall(spf_scores, spf_labels, spf_eer_threshold)
                eer, eer_threshold = calculate_eer(mode_scores[mode], mode_labels[mode])
                precision_all, recall_all = _calculate_precision_recall(mode_scores[mode], mode_labels[mode], eer_threshold)

                gt_acc = _calculate_accuracy(mode_trials[mode]['target'], True, eer_threshold)
                gi_acc = _calculate_accuracy(mode_trials[mode]['nontarget'], False, eer_threshold)
                di_acc = _calculate_accuracy(mode_trials[mode]['spoof'], False, eer_threshold)

                logger.info(f"Mode: {mode}")
                logger.info(f"  EER: {eer*100:.4f}% with threshold {eer_threshold:.6f}")
                logger.info(f"  SV-EER (target vs non-target): {sv_eer*100:.4f}% with threshold: {sv_eer_threshold:.6f}" if sv_eer is not None else "  SV-EER: N/A")
                logger.info(f"  SPF-EER (target vs spoof): {spf_eer*100:.4f}% with threshold: {spf_eer_threshold:.6f}" if spf_eer is not None else "  SPF-EER: N/A")
                logger.info(f"  Target Accuracy: {gt_acc:.4f}% ({len(mode_trials[mode]['target'])} trials)")
                logger.info(f"  Non-Target Accuracy: {gi_acc:.4f}% ({len(mode_trials[mode]['nontarget'])} trials)")
                logger.info(f"  Spoof Accuracy: {di_acc:.4f}% ({len(mode_trials[mode]['spoof'])} trials)")
                logger.info(f"  Precision: {precision_all:.4f}" if precision_all is not None else "  Precision: N/A")
                logger.info(f"  Recall: {recall_all:.4f}" if recall_all is not None else "  Recall: N/A")
                logger.info(f"  SV-Precision (target vs non-target): {sv_precision:.4f}" if sv_precision is not None else "  SV-Precision: N/A")
                logger.info(f"  SV-Recall (target vs non-target): {sv_recall:.4f}" if sv_recall is not None else "  SV-Recall: N/A")
                logger.info(f"  SPF-Precision (target vs spoof): {spf_precision:.4f}" if spf_precision is not None else "  SPF-Precision: N/A")
                logger.info(f"  SPF-Recall (target vs spoof): {spf_recall:.4f}" if spf_recall is not None else "  SPF-Recall: N/A")
                
                # Prepare final result dictionary entry
                if sv_eer is None:
                    sv_eer, sv_eer_threshold = 1.0, 0.0
                if spf_eer is None:
                    spf_eer, spf_eer_threshold = 1.0, 0.0

                result_entry = {
                    'eer': eer,
                    'threshold': eer_threshold,
                    'sv_eer': sv_eer,
                    'sv_threshold': sv_eer_threshold,
                    'spf_eer': spf_eer,
                    'spf_threshold': spf_eer_threshold,
                    'target_acc': gt_acc,
                    'nontarget_acc': gi_acc,
                    'spoof_acc': di_acc,
                    'precision': precision_all,
                    'recall': recall_all,
                    'sv_precision': sv_precision,
                    'sv_recall': sv_recall,
                    'spf_precision': spf_precision,
                    'spf_recall': spf_recall
                }
                if mode == 'weighted_sum':
                    result_entry['alpha'] = weighted_sum_alpha

                results[dataset_idx][mode] = result_entry

        # Visualise
        if visualise_save_path is not None:
            speakers = np.array(speakers)
            embeddings = np.stack(embeddings)
            genuines = np.array(genuine_label)
            save_path_embeddings = Path(visualise_save_path) / f"embeddings_{datasets.index(dataset)}.png"
            visualise_embeddings(embeddings, speakers, genuines, save_path=save_path_embeddings)
            save_path_embedding_lengths = Path(visualise_save_path) / f"embedding_lengths_{datasets.index(dataset)}.png"
            visualize_embedding_lengths(embeddings, speakers, genuines, save_path=save_path_embedding_lengths)
    
    return results


def evaluate_speaker_verification(model, wavloader, trial_file, base_audio_path):
    """
    Evaluate a speaker verification model using trial pairs and cosine similarity.

    Args:
        model (torch.nn.Module): The speaker embedding model. Should output embeddings from input audio features.
        wavloader (Callable[[Path], torch.Tensor]): Function to load audio features from a given file path.
        trial_file (str or Path): Path to a file containing verification trials. Each line should be:
                                  "label path1 path2" where label is 0 (different speakers) or 1 (same speaker).
        base_audio_path (str or Path): Base directory path where audio files are located.

    Returns:
        float: Equal Error Rate (EER)
        float: Threshold at which EER occurs
    """
    model.eval()
    
    # Build embedding cache
    embedding_cache = {}

    with open(trial_file, 'r') as f:
        lines = f.readlines()

    # Collect all unique file paths
    unique_paths = set()
    for line in tqdm(lines, desc="Extract paths"):
        _, path1, path2 = line.strip().split()  # Assumes line format: "label path1 path2"
        unique_paths.add(path1)
        unique_paths.add(path2)

    logger.info(f"Found {len(unique_paths)} unique audio files to process.")

    # Extract embeddings for each unique file path
    progress_bar_inference = tqdm(unique_paths, desc="Extracting embeddings", unit="file")
    for idx, path in enumerate(progress_bar_inference):
        full_path = Path(base_audio_path) / path
        feat = wavloader(full_path)
        with torch.no_grad():
            embedding, _ = model(feat.unsqueeze(0))  # Add batch dimension
        embedding = embedding.squeeze().cpu().numpy()    # Remove batch dimension, move to CPU, and convert to NumPy array
        embedding_cache[path] = torch.tensor(embedding)

        if idx % LOGGING_INT == 0:
            elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar_inference)
            logger.info(f"Extracting embeddings: Elapsed: {elapsed_str} | ETA: {remaining_str}")
    
    # Score trials
    scores = []
    labels = []

    progress_bar_trials = tqdm(lines, desc="Scoring trials", unit="trial")
    for idx, line in enumerate(progress_bar_trials):
        label_str, path1, path2 = line.strip().split()
        label = int(label_str)
        emb1 = embedding_cache[path1]
        emb2 = embedding_cache[path2]
        score = calculate_cosine_similarity(emb1, emb2)
        scores.append(score)
        labels.append(label)

        if idx % LOGGING_INT == 0:
            elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar_trials)
            logger.info(f"Scoring trials: Elapsed: {elapsed_str} | ETA: {remaining_str}")
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(scores, labels)
    logger.info(f"  EER: {eer*100:.4f}%")
    logger.info(f"  Threshold at EER: {eer_threshold:.6f}")


def _get_random_subset(embeddings, labels, genuine_labels, num_speakers=10):
    """
    Filters embeddings, labels, and genuine labels for a random subset of base speakers.

    Args:
        embeddings (np.ndarray): 2D array of shape (num_samples, embedding_dim).
        labels (List or np.ndarray): 1D array for the speaker labels.
        genuine_labels (List or np.ndarray): 1D array for the genuine labels.
        num_speakers (int): Number of speakers to display (default: 10).
                            If -1, then no filtering is applied.

    Returns:
        Tuple of (filtered_embeddings, filtered_labels, filtered_genuine_labels)
    """
    if num_speakers == -1:
        return embeddings, labels, genuine_labels
    
    # Strip "_df" suffix from labels
    base_labels = np.array([re.sub(r'_df$', '', label) for label in labels])

    # Select random base speaker labels
    unique_base_labels = sorted(set(base_labels))
    selected_base_labels = random.sample(unique_base_labels, min(num_speakers, len(unique_base_labels)))

    # Mask to keep only the selected base labels
    mask = np.isin(base_labels, selected_base_labels)

    # Filter data
    return embeddings[mask], np.array(labels)[mask], np.array(genuine_labels)[mask]


def visualise_embeddings(embeddings, labels, genuine_labels, title='Speaker Embeddings by', n_neighbors=15, min_dist=0.1, metric='cosine', num_speakers=10, save_path=None):
    """
    Visualize speaker embeddings using UMAP and optionally save the plot.
    
    Args:
        embeddings (np.ndarray): 2D array of shape (num_samples, embedding_dim).
        labels (List or np.ndarray): 1D array for the speaker labels.
        genuine_labels (List or np.ndarray): 1D array for the genuine labels.
        title (str): Title of the plot (default: 'Speaker Embeddings by').
        n_neighbors (int): UMAP neighborhood size (default: 15).
        min_dist (float): Controls clustering tightness (default: 0.1).
        metric (str): Distance metric (e.g., 'euclidean', 'cosine') (default: 'cosine')
        num_speakers (int): Number of speakers to display (default: 10).
        save_path (str or None): Path to save the image, if desired (default: None).
    """
    logger.info("Visualising embeddings...")

    # Select random subset
    embeddings, labels, genuine_labels = _get_random_subset(embeddings, labels, genuine_labels, num_speakers)

    plt.figure()    # Initialise new figure

    if metric == 'cosine':
        logger.info("Normalizing embeddings for cosine distance...")
        embeddings = normalize(embeddings, norm='l2')
    
    # Reduce embeddings to x and y values
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    reduced_embeddings *= 1.2   # Scale the embeddings for more space in between (zoom out)

    # Extract base speaker names (strip "_df" if present)
    def get_base_label(label):
        return re.sub(r'_df$', '', label)

    # Initialise colors and marker shapes
    base_labels = np.array([get_base_label(label) for label in labels])
    unique_base_labels = sorted(set(base_labels))
    palette = sns.color_palette("hsv", len(unique_base_labels))
    base_color_map = dict(zip(unique_base_labels, palette))
    marker_map = {0: "x", 1: "o"}

    # Plot using base label color and marker type according to genuine label
    for full_label in sorted(set(labels)):
        base_label = get_base_label(full_label)
        color = base_color_map[base_label]
        
        for genuine in sorted(set(genuine_labels)):
            mask = (labels == full_label) & (genuine_labels == genuine)
            if not np.any(mask):
                continue

            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=full_label,
                marker=marker_map[genuine],
                color=color,
                edgecolor='black',
                s=60
            )

    plt.title(f"{title} {metric}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()

    # Add padding to the axes
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    padding_x = (x_max - x_min) * 0.005
    padding_y = (y_max - y_min) * 0.005
    plt.xlim(x_min - padding_x, x_max + padding_x)
    plt.ylim(y_min - padding_y, y_max + padding_y)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embedding visualiation saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_embedding_lengths(embeddings, speaker_labels, genuine_labels, title="Speaker Embedding Length Distribution", save_path=None):
    """
    Visualize the distribution of embedding lengths grouped by speaker class
    and genuine/deepfake labels using strip-plots.

    Args:
        embeddings (np.ndarray): 2D array of shape (num_samples, embedding_dim).
        labels (List or np.ndarray): 1D array for the speaker labels.
        genuine_labels (List or np.ndarray): 1D array for the genuine labels.
        title (str): Title of the plot (default: 'Speaker Embedding Length Distribution').
        save_path (str or None): Path to save the image, if desired (default: None).
    """
    logger.info("Visualising embedding lengths...")

    # Calculate embeddings lenghts    
    lengths = np.linalg.norm(embeddings, axis=1)
    unique_labels = sorted(set(speaker_labels))
    x = np.arange(len(unique_labels))

    plt.figure(figsize=(15, 6)) # Initialise new figure

    # Plot lengths for each speaker label
    for i, label in enumerate(unique_labels):
        idx = [j for j, lab in enumerate(speaker_labels) if lab == label]
        genuine_lengths = lengths[np.array(idx)][np.array(genuine_labels)[idx] == 1]
        deepfake_lengths = lengths[np.array(idx)][np.array(genuine_labels)[idx] == 0]

        plt.scatter(
            np.full_like(genuine_lengths, i) + np.random.uniform(-0.15, 0.15, size=len(genuine_lengths)),
            genuine_lengths,
            color='green',
            alpha=0.5,
            label='Genuine' if i == 0 else "",
            marker='o',
            edgecolor='none',
            s=20,
        )
        plt.scatter(
            np.full_like(deepfake_lengths, i) + np.random.uniform(-0.15, 0.15, size=len(deepfake_lengths)),
            deepfake_lengths,
            color='red',
            alpha=0.5,
            label='Deepfake' if i == 0 else "",
            marker='x',
            s=20,
        )

    plt.xticks(x, unique_labels, rotation=60, ha='right')
    plt.xlabel("Speaker Class")
    plt.ylabel("Embedding Length (L2 norm)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embedding length visualiation saved to {save_path}")
    else:
        plt.show()
    plt.close()
