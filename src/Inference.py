# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from utils.factory import Factory
from train.evaluation import score_fusion, calculate_cosine_similarity


def infer(path1, path2, eer_threshold, wav_loader, model, device, score_fusion_mode, weighted_sum_alpha):
    """
    Performs speaker verification by comparing two audio samples.

    Parameters:
        path1 (str): Path to the first / reference audio file.
        path2 (str): Path to the second audio file.
        eer_threshold (float): Decision threshold for verification based on Equal Error Rate (EER).
        wav_loader (WavLoader): Instantiated WavLoader object for audio preprocessing.
        model (SpeakerEmbedderModel): Instantiated SpeakerEmbedderModel object for computing embeddings.
        device (str): Computation device ('cuda' or 'cpu').
        score_fusion_mode (str or List[str]): Fusion method (default: 'embedding_only'). One or multiple of: 
                                              'embedding_only', 'classifier_only', 'multiplication', 'weighted_sum'.
        weighted_sum_alpha (float): Weight for cosine similarity in weighted sum fusion (default: 0.5).
                                    Used only when score_fusion_mode == 'weighted_sum'.

    Returns:
        bool: `True` if the final verification score is greater than or equal to the EER threshold (i.e., same speaker),
              `False` otherwise.
    """
    input1 = wav_loader(path1).unsqueeze(0).to(device)
    input2 = wav_loader(path2).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding1, _ = model(input1)
        embedding2, genuineness_score = model(input2)
    
    genuineness_prob = torch.sigmoid(genuineness_score).item() if genuineness_score is not None else None

    cosine_similarity = calculate_cosine_similarity(embedding1.squeeze(0), embedding2.squeeze(0))
    final_score = score_fusion(score_fusion_mode, cosine_similarity, genuineness_prob, weighted_sum_alpha)
    print(final_score)

    return final_score >= eer_threshold


def inference_loop_interactive(eer_threshold, wav_loader, model, device, score_fusion_mode, weighted_sum_alpha):
    """
    Interactive inference loop using the provided wav_loader and model. Prompts the user to input two audio paths.
    
    Parameters:
        wav_loader (WavLoader): Instantiated WavLoader object for audio preprocessing.
        model (SpeakerEmbedderModel): Instantiated SpeakerEmbedderModel object for computing embeddings.
        device (str): Computation device ('cuda' or 'cpu').
        score_fusion_mode (str or List[str]): Fusion method (default: 'embedding_only'). One or multiple of: 
                                              'embedding_only', 'classifier_only', 'multiplication', 'weighted_sum'.
        weighted_sum_alpha (float): Weight for cosine similarity in weighted sum fusion (default: 0.5).
                                    Used only when score_fusion_mode == 'weighted_sum'.
    """
    print("Enter two audio file paths (the first must be genuine) separated by a space or type 'exit' to quit:")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == "exit":
            print("Exiting inference loop.")
            break

        try:
            path1, path2 = user_input.split()
        except ValueError:
            print("Invalid input. Please enter exactly two paths separated by a space.")
            continue

        result = infer(path1, path2, eer_threshold, wav_loader, model, device, score_fusion_mode, weighted_sum_alpha)
        print(result)


def inference_loop_file(inference_file, eer_threshold, wav_loader, model, device, score_fusion_mode, weighted_sum_alpha):
    """
    Inference loop using the provided wav_loader and model. Uses audio file path pairs (separated by a space) from the given file.
    Writes the inference results to a file in the same directory as the input file.
    
    Parameters:
        inference_file (str): Path to the file containing pairs of audio file paths to be inferred.
                         The first audio file in each pair must be genuine.
        wav_loader (WavLoader): Instantiated WavLoader object for audio preprocessing.
        model (SpeakerEmbedderModel): Instantiated SpeakerEmbedderModel object for computing embeddings.
        device (str): Computation device ('cuda' or 'cpu').
        score_fusion_mode (str or List[str]): Fusion method (default: 'embedding_only'). One or multiple of: 
                                              'embedding_only', 'classifier_only', 'multiplication', 'weighted_sum'.
        weighted_sum_alpha (float): Weight for cosine similarity in weighted sum fusion (default: 0.5).
                                    Used only when score_fusion_mode == 'weighted_sum'.
    """
    # Determine output path
    dir_path = Path(inference_file).parent
    output_file_path = Path(dir_path) / 'inference_results.txt'

    with open(inference_file, 'r') as f:
        lines = f.readlines()

    with open(output_file_path, 'w') as out_f:
        for line in tqdm(lines, desc="Inferring form file"):
            path1, path2 = line.strip().split()
            result = infer(path1, path2, eer_threshold, wav_loader, model, device, score_fusion_mode, weighted_sum_alpha)
            out_f.write(f"{path1} {path2} {result}\n")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Infer with speaker verification model.")
    parser.add_argument("base_path", type=str, help="Base path to the config directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint file name")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for classification")
    parser.add_argument("--fusion", type=str, required=True, help="Fusion mode (embedding_only, classifier_only, multiplication, weighted_sum)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for weighted sum fusion")
    parser.add_argument("--audio1", type=str, default=None, help="Reference audio file path")
    parser.add_argument("--audio2", type=str, default=None, help="Second audio file path")
    parser.add_argument("--infer_file", type=str, default=None, help="Path to the file with audio pairs")

    # Get paths
    args = parser.parse_args()
    config_path = Path(args.base_path) / "config.yml"
    checkpoint_path = Path(args.base_path) / args.checkpoint
    eer_threshold = args.threshold
    best_fusion_mode = args.fusion
    weighted_sum_alpha = args.alpha
    
    # Load configuration
    factory = Factory(config_path, checkpoint_path)
    wav_loader = factory.create_preprocessor_evaluation()
    model = factory.create_speaker_embedder()
    model.eval()
    device = factory.device

    print(f"Using {best_fusion_mode} with threshold {eer_threshold} for inference" +
          (f" with weighted_sum_alpha={weighted_sum_alpha}" if best_fusion_mode == 'weighted_sum' else ""))

    # Inference
    if args.audio1 is not None and args.audio2 is not None:
        result = infer(Path(args.path1), Path(args.path2), eer_threshold, wav_loader, model, device, best_fusion_mode, weighted_sum_alpha)
        print(result)
    elif args.infer_file is not None:
        inference_loop_file(Path(args.infer_file), eer_threshold, wav_loader, model, device, best_fusion_mode, weighted_sum_alpha)
    else:
        inference_loop_interactive(eer_threshold, wav_loader, model, device, best_fusion_mode, weighted_sum_alpha)
    

if __name__ == "__main__":
    main()