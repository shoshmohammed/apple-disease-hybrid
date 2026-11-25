"""Default settings used for every result reported in the paper."""

CNN_BACKBONE = "tf_efficientnet_lite0"          # EfficientNet-Lite0 (timm, ImageNet weights)
SWIN_BACKBONE = "swin_tiny_patch4_window7_224"  # Swin-Tiny (timm, ImageNet weights)
FUSED_DIM = 512

IMG_SIZE = 224
SEED = 42
VAL_FRAC = 0.10      # fraction of the training groups held out for checkpoint selection
NUM_EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4            # Adam
AUGMENT = False      # resize + normalisation only, as in the paper
NUM_WORKERS = 2
