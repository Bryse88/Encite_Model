"""
Script to update embeddings in Firestore or vector database.
This is meant to be run as a scheduled job.
"""

import os
import argparse
import yaml
import torch
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_embeddings')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update embeddings in database')
    parser.add_argument('--config', type=str, default='configs/hgt_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='outputs',
                        help='Directory containing trained models')
    parser.add_argument('--export_to', type=str, choices=['firestore', 'pinecone', 'local'],
                        default='firestore', help='Where to export embeddings')
    parser.add_argument('--use_best_model', action='store_true',
                        help='Use best model instead of latest')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of the model')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_model(model_dir, use_best=False):
    """
    Find the latest model in the model directory.
    
    Args:
        model_dir: Directory containing model checkpoints
        use_best: Whether to use the best model instead of the latest
    
    Returns:
        Path to the latest model
    """
    if use_best and os.path.exists(os.path.join(model_dir, 'best_model.pt')):
        return os.path.join(model_dir, 'best_model.pt')
    
    # Find checkpoint with highest epoch number
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        if os.path.exists(os.path.join(model_dir, 'final_model.pt')):
            return os.path.join(model_dir, 'final_model.pt')
        else:
            return None
    
    # Extract epoch numbers and find the highest
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_idx = epochs.index(max(epochs))
    
    return os.path.join(model_dir, checkpoints[latest_idx])


def check_if_retrain_needed(model_path, force_retrain=False):
    """
    Check if retraining is needed.
    
    Args:
        model_path: Path to the latest model
        force_retrain: Whether to force retraining
    
    Returns:
        Boolean indicating whether retraining is needed
    """
    # Force retrain if requested
    if force_retrain:
        logger.info("Forcing retraining as requested")
        return True
    
    # If no model exists, training is needed
    if not model_path or not os.path.exists(model_path):
        logger.info("No existing model found, training needed")
        return True
    
    # Check model age
    model_mtime = os.path.getmtime(model_path)
    model_age_days = (datetime.now().timestamp() - model_mtime) / (60 * 60 * 24)
    
    # Retrain if model is older than 7 days
    if model_age_days > 7:
        logger.info(f"Model is {model_age_days:.1f} days old, retraining needed")
        return True
    
    logger.info(f"Model is {model_age_days:.1f} days old, no retraining needed")
    return False


def train_model(config_path, output_dir):
    """
    Train the model using the training script.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for trained model
    
    Returns:
        Path to the trained model
    """
    logger.info("Starting model training")
    
    # Build command
    cmd = [
        "python", "train.py",
        "--config", config_path,
        "--output_dir", output_dir
    ]
    
    # Run training process
    try:
        process = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Model training completed successfully")
        logger.debug(process.stdout)
        
        # Return path to best model
        return os.path.join(output_dir, 'best_model.pt')
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed: {e}")
        logger.error(e.stderr)
        return None


def export_embeddings(model_path, config_path, export_to):
    """
    Export embeddings from the trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to config file
        export_to: Where to export embeddings (firestore, pinecone, local)
    
    Returns:
        Boolean indicating success
    """
    logger.info(f"Exporting embeddings to {export_to}")
    
    # Build command
    cmd = [
        "python", "export_embeddings.py",
        "--config", config_path,
        "--model_path", model_path,
        "--export_to", export_to
    ]
    
    if export_to == 'local':
        # Add output directory for local export
        output_dir = os.path.join('outputs', 'embeddings', datetime.now().strftime("%Y%m%d-%H%M%S"))
        cmd.extend(["--output_dir", output_dir])
    
    # Run export process
    try:
        process = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Embedding export completed successfully")
        logger.debug(process.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding export failed: {e}")
        logger.error(e.stderr)
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging to file if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Load configuration
    config = load_config(args.config)
    
    # Find latest model
    model_path = find_latest_model(args.model_dir, args.use_best_model)
    logger.info(f"Latest model: {model_path}")
    
    # Check if retraining is needed
    if check_if_retrain_needed(model_path, args.force_retrain):
        # Train model
        model_path = train_model(args.config, args.model_dir)
        
        if not model_path:
            logger.error("Model training failed, exiting")
            return
    
    # Export embeddings
    success = export_embeddings(model_path, args.config, args.export_to)
    
    if success:
        logger.info("Embedding update completed successfully")
    else:
        logger.error("Embedding update failed")


if __name__ == '__main__':
    main()