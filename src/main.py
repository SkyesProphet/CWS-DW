import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import logging
import sys
import time

# Module Import
from config import LABELS_DEFINITION
from model import UnifiedModel, tokenizer
from Trainer import  TrainerMM
from process_dw import DarkwebDataset, MultimodalCollateFn, FinetuneCollateFn

# --- logging ---
def setup_logging(save_dir):
    """Configuration Log"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_filename = os.path.join(save_dir, f'run_{int(time.time())}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
    )

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Task Training and Testing Script")
    parser.add_argument("--model_type", type=str, default="similarity", choices=["similarity", "head"])
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    """--- 1. Parameters ---"""
    # logging
    SAVE_DIR = './saved_models_mmc'
    setup_logging(SAVE_DIR)
    logging.info(f"Model type: {args.model_type}")

    txt_train_path = r"data_mnt\DATA\...THIS IS TXT TRAIN PATH"
    txt_val_path = r"data_mnt\DATA\...THIS IS TXT VAL PATH"
    txt_test_path = r"data_mnt\DATA\...THIS IS TXT TEST PATH"
    images_dir = r"data_mnt\DATA\...THIS IS ALL IMAGES PATH"
    PRETRAINED_PATH = 'pretraining_model_path'
    model_save_path = 'save_model_path'
    label_list = list(LABELS_DEFINITION.keys())
    num_labels = len(label_list)
    contractive_method = 'adaptive'  # 'jaccard', 'dice', 'harmonic', 'geometric', 'adaptive'
    
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 100
    PATIENCE = 5 # Early stopping
    VALID_STEPS = 50 # 50 step
    THRESHOLD = 0.5

    # --- Configuration ---
    LR_CONFIG = {
        'text_encoder': 2e-5,      
        'image_encoder': 5e-5,     
        'fusion_layers': 1e-4,     
        'classifier': 1e-4,       
    }
    WD_CONFIG = {
        'encoder': 1e-4,
        'classifier': 1e-4,
    }
    LOSS_WEIGHTS = {
        'cls': 1.0,   
        'mlm': 0.5,   
        'con': 0.7,  
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. Loading data
    train_dataset = DarkwebDataset(images_dir, txt_train_path, label_list)
    val_dataset = DarkwebDataset(images_dir, txt_val_path, label_list)
    test_dataset = DarkwebDataset(images_dir, txt_test_path, label_list)

    collate_train = MultimodalCollateFn(tokenizer=tokenizer)
    collate_eval = FinetuneCollateFn(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval, num_workers=NUM_WORKERS, pin_memory=True)
    logging.info(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    # 3. Initialization
    model = UnifiedModel(
        # pretrained_backbone_path=PRETRAINED_PATH,
        model_type=args.model_type,
        label_definitions=LABELS_DEFINITION,
        device=device
    )
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # 4. Trainer
    trainer = TrainerMM(model, train_loader, val_loader,test_loader, label_list,
                        model_save_path, LR_CONFIG, WD_CONFIG, LOSS_WEIGHTS,
                        PATIENCE, args, contractive_method, device)

    # 5. train or test
    if args.train:
        logging.info("Starting training...")
        logging.info(f"Device: {device}")
        logging.info(f"Learning Rates: {LR_CONFIG}")
        logging.info(f"Weight Decays: {WD_CONFIG}")
        logging.info(f"Loss Weights: {LOSS_WEIGHTS}")
        
        trainer.train(EPOCHS, VALID_STEPS)
        trainer.test(THRESHOLD)
    else: 
        trainer.test(THRESHOLD)

if __name__ == "__main__":
    # pip install scikit-learn
    main()
