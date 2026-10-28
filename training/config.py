"""
training/config.py
Central configuration for the Deep Learning training pipeline.
All paths are relative to the project root (video-summarizer/).
"""

# Reproducibility
SEED = 42

# Dataset paths
KAGGLE_DATASET_ROOT    = None          # set by feature_extraction.py at runtime
PROCESSED_DATASET_PATH = "datasets/processed_dataset.json"
DATA_SPLIT_PATH        = "datasets/data_split.json"
FEATURES_CACHE_DIR     = "datasets/video_features"

# Frame / Feature extraction
FRAME_SAMPLE_FPS = 2           # 2 frames-per-second sampled from each video
FEATURE_DIM      = 512         # ResNet-18 output dimension
CNN_BACKBONE     = "resnet18"
SUMME_FPS        = 2.0
TVSUM_FPS        = 2.0

# Data split (by VIDEO, never by segment)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Model architecture
VISUAL_DIM  = 512
TEXT_DIM    = 384    # all-MiniLM-L6-v2 sentence-transformer
FUSED_DIM   = 256
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
MAX_SEQ_LEN = 200    # max temporal segments per video

# Training hyperparameters
EPOCHS              = 30
BATCH_SIZE          = 4
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
SCHEDULER_PATIENCE  = 5     # ReduceLROnPlateau patience
EARLY_STOP_PATIENCE = 8     # EarlyStopping patience
DEVICE              = "cpu" # overridden to "cuda" if available at runtime

# Output paths
MODEL_SAVE_PATH    = "models/video_importance_model.pth"
CHECKPOINT_DIR     = "models/checkpoints"
EVAL_OUTPUT_PATH   = "evaluation/results.json"
LOG_EVERY_N_EPOCHS = 1
