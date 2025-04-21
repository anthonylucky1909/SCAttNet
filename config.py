from pathlib import Path
import os
import torch
import logging

# Initialize device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
lr = 1e-4
epochs = 10
batch_size = 8
num_workers = 16
high_res = 256
low_res = high_res // 4
num_channels = 3

# Dataset paths (initialized as empty)
data_paths = {
    'train': None,
    'test1': None,
    'test2': None,
    'test3': None,
    'test4': None,
    'test5': None
}

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('super_resolution.log'),
            logging.StreamHandler()
        ]
    )

def setup_paths(downloaded_paths):
    """Update paths after downloading datasets"""
    global data_paths
    try:
        data_paths = {
            'train': os.path.join(downloaded_paths['div2k'], 'DIV2K_train_HR/DIV2K_train_HR'),
            'test1': os.path.join(downloaded_paths['set5_14'], 'Set14/Set14'),
            'test2': os.path.join(downloaded_paths['div2k'], 'DIV2K_valid_HR/DIV2K_valid_HR'),
            'test3': os.path.join(downloaded_paths['urban100'], 'Urban 100/X4 Urban100/X4/HIGH x4 URban100'),
            'test4': os.path.join(downloaded_paths['bsd100'], 'bsd100/bicubic_4x/val/HR'),
            'test5': downloaded_paths['manga109']
        }
        logging.info("Dataset paths configured successfully")
    except KeyError as e:
        logging.error(f"Missing dataset key in downloaded paths: {e}")
        raise
    except Exception as e:
        logging.error(f"Error configuring paths: {e}")
        raise

# Initialize logging when module is imported
setup_logging()