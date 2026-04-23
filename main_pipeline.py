
#Executes the multimodal fusion pipeline sequentially.


import argparse
import logging
import os
import sys
import subprocess
import torch


SCRIPTS = {
    "prepare_data": "data_preprocessing/prepare_bcss.py",    
    "train_swin": "models/swin_training.py",                 
    "extract_vis": "models/visual_embedding.py",             
    "train_clin": "models/clinical_encoder.py",              
    "train_fusion": "models/ds_gca_fusion.py"              
}

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def run_script(stage_name, script_path):
    """Helper function to run a script via subprocess."""
    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name.upper()}")
    logger.info("=" * 60)
    
    if not os.path.exists(script_path):
        logger.error(f" Could not find script at: {script_path}")
        logger.error("Please update the SCRIPTS dictionary in main.py with the correct path.")
        return False
        
    try:
        logger.info(f"Executing {script_path}...")
        
        result = subprocess.run([sys.executable, script_path], check=True)
        logger.info(f" {stage_name} complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f" {stage_name} failed with exit code {e.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="DS-GCA Orchestrator")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', '1', '2', '3', '4', '5'],
                        help='Pipeline stage to run (default: all)')
    
    args = parser.parse_args()
    
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║  DS-GCA: Deep Swin - Gated Clinical Attention Pipeline   ║")
    logger.info("╚" + "=" * 58 + "╝")
    
    device = setup_device()
    success_count = 0
    total_run = 0
    
    if args.stage in ['all', '1']:
        total_run += 1
        if run_script("Data Preparation", SCRIPTS["prepare_data"]): success_count += 1
            
    if args.stage in ['all', '2']:
        total_run += 1
        if run_script("Swin Training", SCRIPTS["train_swin"]): success_count += 1
            
    if args.stage in ['all', '3']:
        total_run += 1
        if run_script("Visual Embedding", SCRIPTS["extract_vis"]): success_count += 1
            
    if args.stage in ['all', '4']:
        total_run += 1
        if run_script("Clinical Encoding", SCRIPTS["train_clin"]): success_count += 1
            
    if args.stage in ['all', '5']:
        total_run += 1
        if run_script("Multimodal Fusion Training", SCRIPTS["train_fusion"]): success_count += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline Complete: {success_count}/{total_run} scripts executed successfully.")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()