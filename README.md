# ArcFace for Audio Deepfake Detection (ADD)

**Thesis Title:** *ArcFace for Audio Deepfake Detection – Leveraging Additive Angular Margin Loss based Automatic Speaker Verification for the Detection of Audio Deepfakes using Reference Speech Data*  
**Author:** Julius Müther  
**Institution:** Ruhr-University Bochum in cooperation with the Federal Office for Information Security (Bundesamt für Sicherheit in der Informationstechnik)  
**Year:** 2025

---

## Abstract

This thesis investigates reference-based audio deepfake detection (ADD) by adapting automatic speaker verification (ASV) systems. Traditional ADD methods often rely on signal-level artefacts and do not leverage reference speech from the target speaker, limit-ing their ability to assess speaker authenticity and to generalise across diverse speakers, synthesis techniques and real-world acoustic conditions. ASV systems, particularly those trained with ArcFace, produce highly discriminative embeddings, that separate speaker identities effectively, and provide a promising foundation for detecting subtle inconsisten-cies between genuine and synthetic utterances.

The work introduces a modular framework combining discriminative speaker embed-dings, self-supervised learning (SSL) based representations and a modified ArcFace loss for spoof-aware representation learning among other targeted experimental investigations. The proposed loss adds a deepfake-specific term that prevents synthetic utterances from influencing genuine speaker centres during training, while pushing deepfake embeddings away from the cluster of genuine samples of a speaker. This yields embeddings that are identity-discriminative and manipulation-aware, enabling ADD by thresholding the cosine similarity of the suspect and a reference audio utterance.

Extensive experiments on SpoofCeleb, ASVspoof 5, and In-the-Wild (ITW) demonstrate consistent improvements over baselines and state-of-the-art ADD systems, achieving SPF-EERs of 0.95% on SpoofCeleb-Eval and 2.92% on ITW. In particular, SSL frontends, ASV pretraining and a length-based classifier, that utilises embedding magnitude, im-proves generalisation and robustness across domains and real-world conditions.

Overall, these results show that ASV architectures can be effectively adapted for refer-ence-based ADD through targeted architectural and loss modifications. By explicitly lev-eraging speaker identity alongside deepfake artefacts, the proposed framework provides a robust, generalisable and scalable solution for reliable ADD in practical scenarios.


---

## 🧠 Overview
This repository contains the official implementation of the framework developed for the master thesis on **reference-based Audio Deepfake Detection (ADD)** using **Automatic Speaker Verification (ASV)** architectures and a **novel modified ArcFace loss**.

Traditional deepfake detection methods often focus on signal-level artefacts, lacking the ability to leverage reference speech. This work adapts ASV systems, particularly those trained with **ArcFace**, to identify subtle inconsistencies between genuine and synthetic speech. The proposed system integrates **self-supervised learning (SSL)** frontends, **the novel spoof-aware ArcFace loss** and **cosine similarity-based scoring** for robust and generalisable ADD, which leverages **speaker-discriminative and spoof-aware embeddings** of a **suspect and reference audio**.

Key contributions:
- Modified **ArcFace loss** for deepfake-aware speaker representation learning.
- Modular and configurable framework for **reference-based ADD**.
- Integration of **SSL frontends (e.g., WavLM)** via S3PRL.
- Flexible YAML-based experiment recipes for reproducibility and highly modular models and optimisation strategies.
- Empirical validation on **SpoofCeleb**, **ASVspoof 5** and **In-the-Wild (ITW)** datasets.

---

## 📁 Project Structure
```
src
├── Train.py              # Train ADD system
├── Test.py               # Evaluate ADD system
├── Test_SV.py            # Evaluate ASV system
├── Inference.py          # Run inference / ADD detection
├── backends_3dspeaker/   # Backend architectures (3D-Speaker models)
├── backends_wespeaker/   # Backend architectures (WeSpeaker models)
├── classifiers/          # Length-based classifier
├── configs/              # Experiment configurations (YAML recipes)
├── datasets/             # Dataset creation and loaders
├── frontends/            # Feature extraction modules (fbank, WavLM)
├── losses/               # ArcFace variants and schedulers
├── preprocessing/        # Audio loading, preprocessing and data augmentation 
├── speaker_embedder/     # Combined frontend-backend model
├── train/                # Trainer, schedulers, evaluation utilities
├── utils/                # Config loading and factory utilities
└── utils_data/           # Dataset restructuring and trial generation tools
```

---

## ⚙️ Installation

**Python version:** 3.10.12

**Requirements:**
```bash
--extra-index-url https://download.pytorch.org/whl/cu124
numpy
pandas
tqdm
torch==2.4.1
torchaudio==2.4.1
scikit-learn
s3prl
transformers
tinytag
librosa
matplotlib
PyYAML
setuptools
umap-learn
seaborn
```

**Install environment:**
```bash
conda create -n arcface_add python=3.10.12
conda activate arcface_add
pip install -r requirements.txt
```

---

## 🧩 Configuration

All experimental settings are controlled through YAML config files in `src/configs/`.  
Below is an example for **training**, **validation**, **evaluation** and **dataset setup**.
Not all parameters need to be specified, as then default values will be used. Missing parameters are reported in error messages.

### example.yml`
```yaml
model_checkpoint: 'pretrained_model.pt' # Pretrained ADD checkpoint

train_args: # Configuration for training
  num_epochs: 10    # Training epochs
  save_epoch_interval: 1    # Save model every 1 epoch
  log_interval: 500 # Log training process every 500 batches
  accumulation_interval: 4 # Gradient accumulation steps 
  lm_finetune: True # Large-Margin finetuning or finetuning of ASV model for ADD

loss_balancer:  # Used loss balancer to combine losses
  loss_balancer_name: WeightedSum
  loss_balancer_args: {}

train_dataset:  # Training datasets
  dataset_string: SPOOFCELEB_Train  # Names of used datasets, which contain genuine and spoofed audio
  dataset_string_genuine_only: # Names of used datasets, which contain only genuine audio
  speaker_label_encoder: SpkLabelEncoder  # Name of used speaker labels encoder, e.g. SpkLabelEncoder, AlternatingGenuineDeepfakePairEncoder, GroupedGenuineDeepfakePairEncoder, DeepfakeUnifiedEncoder
  filter_out_vocoder: False # Filter vocoded audio to not use it during training
  vocoder_as_genuine: True  # Mark vocoded audio as genuine

train_dataloader:   # Dataloader used for training
  dataloader_args:  # Standard pyTorch dataloader arguments
    num_workers: 8
    pin_memory: True
    prefetch_factor: 4
  sampler_args:   
    sampler: SpeakerGenuineFakeBalancedSampler  # Name of custom data sampler, e.g. SpeakerBalancedBatchSampler, SpeakerGenuineFakeBalancedSampler, GenuineFakeBalancedSampler
    n_speakers_per_batch: 8 # Speakers per batch
    n_genuine_per_speaker: 2    # Genuine utterances per speaker
    n_fake_per_speaker: 2   # Deepfake utterances per speaker

validation_args: # Configuration for validation
  score_fusion_mode: [embedding_only, classifier_only, weighted_sum, multiplication]    # Used core fusion techniques
  weighted_sum_alpha: 0.5

validation_dataset: # Validation datasets
  trial_dataset: SPOOFCELEB_Validation ITW # Names of used datasets
  batch_size: 1 # Unused parameter, always 1
  num_workers: 0  # Unused parameter, always 0
  pin_memory: True
  prefetch_factor: # Unused parameter, always None

test_args:  # Configuration for evaluation
  score_fusion_mode: [embedding_only, classifier_only, weighted_sum, multiplication]    # Used core fusion techniques
weighted_sum_alpha: 0.5

test_dataset:   # Evaluation datasets
  trial_dataset: SPOOFCELEB_Test ITW    # Names of used datasets
  batch_size: 1 # Unused parameter, always 1
  num_workers: 0  # Unused parameter, always 0
  pin_memory: True
  prefetch_factor: # unused parameter, always None

preprocessing:  # Configuration for audio loading and preprocessing
  duration: 2 # Length of audios; for evaluation this is internally -1, so full audio is used during evaluation
  sample_rate: 16000    # Sample rate
  silence_threshold: 30 # Threshold to remove silence
  remove_leading_trailing_silence: True # Remove only leading and trailing silence
  remove_all_silence: False # Remove all silence present in audio
  augmentations: # Data augmentation
    rir:
      prob: 0.0 # Probability of application
      folder: '/path/to/RIRS_NOISES'
    musan:
      prob: 0.0 # Probability of application
      folder: '/path/to/musan'
      noise_types: ["noise", "speech", "music"] # Used noice types
      snr_range:    # Signal to noise ratios for used noise types
        noise: [0, 20]
        speech: [13, 20]
        music: [5, 15]
    rawboost:
      prob: 0.5 # Probability of application
      algorithm:    # Used algorithm index (1-8), or None if random algorithm per application

frontend:   # Configuration for used frontend
  model_name: wavlm_large   # SSL model name from S3PRL
  model_args:
    finetune: False # Train together with backend
    output_layers:  # Output layers used, or None for all layers
    layer_norm: True    # Layer normalisation before weighted sum of layers
    output_dim: 80  # Map SSL features to this dimension
    adapter: ResidualAdapter    # Use this adapter for dimensional mapping

backend:    # Configuration for used backend
  model_name: ECAPA_TDNN_GLOB_c1024 # Name of backend architecture
  model_args: {}    # Model arguments
  checkpoint_path:  # Path to pretrained backend model checkpoint

embedding_projection:   # Configuration for used loss
  projection_name: ArcFace  # Name of embedding classifier / projection; automatically uses modified variant if deepfake utterances are used during training
  projection_args: # Embedding classifier / projection arguments
    margin: 0.3 # Will be overridden if margin scheduler is configured
    scale: 64.0
    easy_margin: False
  loss: # Loss for embeddings
    loss_name: CrossEntropyLoss # Name of used loss
    loss_args:  # Standard pyTorch loss arguments
      reduction: 'none' # No reduction for modified ArcFace

margin_scheduler:   # Configuration for used margin scheduler (optional)
  scheduler_name: ArcFaceMarginScheduler    # Name of used margin scheduler
  scheduler_args:
    start_epoch: 0  # When to start scheduling
    fix_epoch: 2    # When target margin should be reached
    initial_margin: 0.0 # Initial margin
    target_margin: 0.3  # Final margin
    increase_type: "exp" # Interpolation, e.g. exp (exponential), linear

classifier: # Configuration for used classifier (optional)
  model_name: LengthBasedClassifier # Name of used classifier
  model_args: {}    # Classifier arguments
  loss: # Loss for classifier
    loss_name: BCEWithLogitsLoss    # Name of used loss
    loss_args: {}   # Standard pyTorch loss arguments

optimiser:  # Configuration for used optimiser
  optimiser_name: Adam  # Name of optimiser
  optimiser_args:
    backend:    # Arguments for backend optimisation
      lr: 0.001
      weight_decay: 0.0
    ssl:    # Arguments for SSL frontend optimisation
      lr: 0.00001
      weight_decay: 0.0
    other_args:

lr_scheduler:   # Configuration for used learning rate scheduler (optional)
  scheduler_name: ExponentialDecrease # Name of scheduler, e.g. ExponentialDecrease, CosineDecrease
  scheduler_args:
    warmup_epoch: 2 # Increase learning rate from 0 to initial learning rate configured in optimiser; 0 for no warmup
    target_lr: 0.00001  # Final learning rate
```

### Example: `config.yml`
This config file contains dataset paths and the used random seed
```yaml
seed: 42
dataset_base_path: .../Datasets

speaker_verification_evaluation:
  vox1_o_clean:
    trial_file: '.../vox1-O-clean.txt'
    base_audio_path: '.../vox1_test_wav/wav'

deepfake_speaker_verification_evaluation:
  SPOOFCELEB_Test:
    trial_file: '.../spoofceleb/sasv_evaluation_evaluation_protocol.csv'
    base_audio_path: '.../spoofceleb/flac/evaluation/'
  ITW:
    trial_file: '.../release_in_the_wild/trials.txt'
    base_audio_path: '.../release_in_the_wild/'
```

---

## 🧱 Framework Components
- **Frontends:** Traditional (fbank) and SSL-based (WavLM via S3PRL)
- **Backends:** ECAPA-TDNN, ERes2Net, CampPlus, etc.
- **Losses:** ArcFace + spoof-aware variant
- **Schedulers:** Learning rate & margin schedulers
- **Classifier:** Length-based binary deepfake classifier
- **Config System:** YAML-based recipes for reproducibility

---

## 🚀 Usage

### Training
```bash
python src/Train.py src/configs/2_frontend_analysis/1
```
Optionally resume from a checkpoint:
```bash
python src/Train.py src/configs/2_frontend_analysis/1 --checkpoint checkpoint_epoch_1.pt
```

### Testing ADD models
```bash
python src/Test.py src/configs/2_frontend_analysis/1 --checkpoint checkpoint_epoch_10.pt
```

### Testing ASV models
```bash
python src/Test_SV.py src/configs/7_pretrained_analysis/1 --checkpoint checkpoint_epoch_10.pt --test vox1_o_clean
```

### Inference
Interactive mode:
```bash
python src/Inference.py src/configs/2_frontend_analysis/1 --checkpoint checkpoint_epoch_10.pt
```
Batch mode using file pairs:
```bash
python src/Inference.py src/configs/2_frontend_analysis/1 --checkpoint checkpoint_epoch_10.pt --infer_file pairs.txt
```

---

## 📊 Evaluation Results

| Dataset | Fusion | SPF-EER [%] | SPF-Precision [%] | SPF-Recall [%] |
|----------|--------|-------------|-------------------|----------------|
| **SpoofCeleb Validation** | cosine only | 4.60 | 93.36 | 95.40 |
| | classifier only | **2.90** | **98.78** | **97.11** |
| **SpoofCeleb Evaluation** | cosine only | 4.17 | 92.98 | 95.83 |
| | classifier only | **0.95** | **99.68** | **99.08** |
| **ASVspoof 5 Dev (Track 2)** | cosine only | 9.53 | 59.57 | 90.47 |
| | classifier only | **7.25** | **95.28** | **92.75** |
| **ASVspoof 5 Eval (Track 2)** | cosine only | 16.19 | 40.29 | 83.81 |
| | classifier only | **10.08** | **90.81** | **89.92** |
| **In-the-Wild (ITW)** | cosine only | 6.11 | 93.89 | 93.90 |
| | classifier only | **2.92** | **98.52** | **97.08** |

The modified ArcFace loss and SSL frontends (e.g. WavLM) significantly improve spoof detection, achieving up to **0.95% SPF-EER** on SpoofCeleb Evaluation and **2.92% SPF-EER** on ITW, outperforming state-of-the-art baselines.

---

## 📄 License
This project is licensed under the **Apache License 2.0**.  
See [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) for details.

---

## 📚 Citation
If you use this code or framework, please cite:
```bibtex
@mastersthesis{muether2025arcfaceadd,
  title={ArcFace for Audio Deepfake Detection – Leveraging Additive Angular Margin Loss based Automatic Speaker Verification for the Detection of Audio Deepfakes using Reference Speech Data},
  author={Julius Müther},
  school={Ruhr-University Bochum},
  year={2025},
  note={In cooperation with the Federal Office for Information Security (BSI)}
}
```

---

## 🙌 Acknowledgements
This framework draws inspiration from:
- **ESPnet** – end-to-end speech processing toolkit
- **WeSpeaker** – modular ASV system with flexible recipe structure
- **3D-Speaker** – ASV architectures used for baseline comparison
- **S3PRL** – SSL speech representation toolkit

Special thanks to the **Federal Office for Information Security (BSI)** for the cooperation and support in this research.


