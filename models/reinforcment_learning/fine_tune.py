"""
Fine-tuning script for RL scheduler using user feedback.

This script loads a pretrained policy and fine-tunes it using
user feedback collected from actual app usage.
"""

import os
import logging
import argparse
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from google.cloud import storage
from firebase_admin import firestore, initialize_app

from models.reinforcement_learning.env.schedule_env import ScheduleEnv
from models.reinforcement_learning.train import PPOTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackCollector:
    """
    Collect and process user feedback for RL fine-tuning.
    
    This class fetches user feedback from Firestore and processes it
    into a format suitable for RL fine-tuning.
    """
    
    def __init__(self, min_feedback_count: int = 50, days_lookback: int = 30):
        """
        Initialize feedback collector.
        
        Args:
            min_feedback_count: Minimum number of feedback samples to collect
            days_lookback: Number of days to look back for feedback
        """
        # Initialize Firebase
        initialize_app()
        self.db = firestore.client()
        self.min_feedback_count = min_feedback_count
        self.days_lookback = days_lookback
        
    def collect_feedback(self) -> List[Dict[str, Any]]:
        """
        Collect user feedback from Firestore.
        
        Returns:
            List of feedback samples
        """
        logger.info("Collecting user feedback from Firestore")
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=self.days_lookback)
        
        # Query feedback collection
        feedback_ref = self.db.collection('feedback')
        query = (
            feedback_ref
            .where('timestamp', '>=', cutoff_date)
            .where('processed_for_training', '==', False)
            .order_by('timestamp', direction=firestore.Query.DESCENDING)
        )
        
        feedback_docs = query.stream()
        feedback_samples = []
        
        for doc in feedback_docs:
            feedback = doc.to_dict()
            feedback['id'] = doc.id
            feedback_samples.append(feedback)
            
        logger.info(f"Collected {len(feedback_samples)} feedback samples")
        
        if len(feedback_samples) < self.min_feedback_count:
            logger.warning(f"Only {len(feedback_samples)} feedback samples found, which is less than the minimum required ({self.min_feedback_count})")
            
        return feedback_samples
    
    def process_feedback(self, feedback_samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process raw feedback into training data.
        
        Args:
            feedback_samples: Raw feedback samples from Firestore
            
        Returns:
            Dictionary mapping feedback types to processed samples
        """
        logger.info("Processing feedback samples")
        
        # Group feedback by type
        processed_feedback = {
            'accept': [],
            'modify': [],
            'reject': []
        }
        
        for feedback in feedback_samples:
            feedback_type = feedback.get('type')
            
            if feedback_type not in processed_feedback:
                continue
                
            # Extract relevant information
            processed_sample = {
                'user_id': feedback.get('user_id'),
                'schedule_id': feedback.get('schedule_id'),
                'feedback_id': feedback.get('id'),
                'original_schedule': feedback.get('original_schedule', []),
                'modified_schedule': feedback.get('modified_schedule', []),
                'reason': feedback.get('reason'),
                'timestamp': feedback.get('timestamp')
            }
            
            processed_feedback[feedback_type].append(processed_sample)
            
        for feedback_type, samples in processed_feedback.items():
            logger.info(f"Processed {len(samples)} {feedback_type} feedback samples")
            
        return processed_feedback
    
    def mark_as_processed(self, feedback_ids: List[str]):
        """
        Mark feedback as processed in Firestore.
        
        Args:
            feedback_ids: List of feedback document IDs
        """
        batch = self.db.batch()
        
        for feedback_id in feedback_ids:
            doc_ref = self.db.collection('feedback').document(feedback_id)
            batch.update(doc_ref, {'processed_for_training': True})
            
        batch.commit()
        logger.info(f"Marked {len(feedback_ids)} feedback samples as processed")


class FeedbackFineTuner:
    """
    Fine-tune RL policy using user feedback.
    
    This class implements methods to fine-tune the policy based on
    different types of user feedback.
    """
    
    def __init__(self, 
                 pretrained_model_path: str,
                 config_path: str,
                 output_dir: str = 'models/reinforcement_learning/saved_models'):
        """
        Initialize fine-tuner.
        
        Args:
            pretrained_model_path: Path to pretrained model
            config_path: Path to config file
            output_dir: Directory to save fine-tuned models
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize trainer
        self.trainer = PPOTrainer(
            policy_type=self.config['model']['policy_type'],
            state_dim=self.config['model']['state_dim'],
            actor_lr=self.config['fine_tuning']['learning_rate'],
            critic_lr=self.config['fine_tuning']['learning_rate'] * 2,
            gamma=self.config['model']['gamma'],
            gae_lambda=self.config['model']['gae_lambda'],
            clip_range=self.config['fine_tuning']['clip_range'],
            batch_size=self.config['fine_tuning']['batch_size'],
            n_epochs=self.config['fine_tuning']['n_epochs'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create test environment
        self.env = self._create_test_env()
        
        # Load pretrained model
        self.trainer.load(pretrained_model_path, self.env)
        
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _create_test_env(self) -> ScheduleEnv:
        """
        Create a test environment for policy evaluation.
        
        Returns:
            Schedule environment
        """
        return ScheduleEnv(
            user_id=self.config['fine_tuning']['test_user_id'],
            date=datetime.strptime(self.config['fine_tuning']['test_date'], '%Y-%m-%d').date(),
            start_time=datetime.strptime(self.config['fine_tuning']['test_start_time'], '%H:%M').time(),
            end_time=datetime.strptime(self.config['fine_tuning']['test_end_time'], '%H:%M').time(),
            location=tuple(self.config['fine_tuning']['test_location']),
            preferences=self.config['fine_tuning']['test_preferences'],
            budget=self.config['fine_tuning'].get('test_budget'),
            transportation_modes=self.config['fine_tuning'].get('test_transportation_modes'),
            config_path=self.config_path
        )
    
    def fine_tune(self, processed_feedback: Dict[str, List[Dict[str, Any]]]):
        """
        Fine-tune policy using processed feedback.
        
        Args:
            processed_feedback: Dictionary mapping feedback types to processed samples
        """
        logger.info("Starting fine-tuning process")
        
        # Get initial policy performance
        initial_rewards = self.trainer.evaluate(self.env, n_episodes=10)
        logger.info(f"Initial policy performance: mean reward = {np.mean(initial_rewards):.2f}")
        
        # Process different feedback types
        self._process_accept_feedback(processed_feedback['accept'])
        self._process_modify_feedback(processed_feedback['modify'])
        self._process_reject_feedback(processed_feedback['reject'])
        
        # Evaluate fine-tuned policy
        final_rewards = self.trainer.evaluate(self.env, n_episodes=10)
        logger.info(f"Fine-tuned policy performance: mean reward = {np.mean(final_rewards):.2f}")
        
        # Save fine-tuned model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trainer.save(os.path.join(self.output_dir, f'fine_tuned_policy_{timestamp}.pt'))
        logger.info(f"Saved fine-tuned model to {self.output_dir}/fine_tuned_policy_{timestamp}.pt")
        
    def _process_accept_feedback(self, accept_samples: List[Dict[str, Any]]):
        """
        Process "accept" feedback to reinforce good schedules.
        
        Args:
            accept_samples: List of accepted schedule samples
        """
        if not accept_samples:
            logger.info("No 'accept' feedback samples to process")
            return
            
        logger.info(f"Processing {len(accept_samples)} 'accept' feedback samples")
        
        # For accepted schedules, we want to increase the probability
        # of generating similar schedules in the future
        
        # This would typically involve collecting these trajectories
        # and training the RL policy with positive rewards
        
        # For brevity, we'll skip the implementation details here
        
    def _process_modify_feedback(self, modify_samples: List[Dict[str, Any]]):
        """
        Process "modify" feedback to learn from user modifications.
        
        Args:
            modify_samples: List of modified schedule samples
        """
        if not modify_samples:
            logger.info("No 'modify' feedback samples to process")
            return
            
        logger.info(f"Processing {len(modify_samples)} 'modify' feedback samples")
        
        # For modified schedules, we want to:
        # 1. Discourage the specific actions that were modified
        # 2. Encourage the actions that users chose as replacements
        
        # This would typically involve:
        # - Recreating the original environment state
        # - Applying negative rewards to removed activities
        # - Applying positive rewards to added activities
        # - Training the policy on these modified trajectories
        
    def _process_reject_feedback(self, reject_samples: List[Dict[str, Any]]):
        """
        Process "reject" feedback to avoid generating rejected schedules.
        
        Args:
            reject_samples: List of rejected schedule samples
        """
        if not reject_samples:
            logger.info("No 'reject' feedback samples to process")
            return
            
        logger.info(f"Processing {len(reject_samples)} 'reject' feedback samples")
        
        # For rejected schedules, we want to decrease the probability
        # of generating similar schedules in the future
        
        # This would typically involve collecting these trajectories
        # and training the RL policy with negative rewards


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune RL policy using user feedback')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--config', type=str, default='configs/rl_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='models/reinforcement_learning/saved_models',
                       help='Directory to save fine-tuned models')
    parser.add_argument('--min-feedback', type=int, default=50,
                       help='Minimum number of feedback samples required')
    parser.add_argument('--days-lookback', type=int, default=30,
                       help='Number of days to look back for feedback')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Collect and process feedback
    collector = FeedbackCollector(
        min_feedback_count=args.min_feedback,
        days_lookback=args.days_lookback
    )
    
    feedback_samples = collector.collect_feedback()
    
    if len(feedback_samples) < args.min_feedback:
        logger.warning("Not enough feedback samples to perform fine-tuning")
        return
        
    processed_feedback = collector.process_feedback(feedback_samples)
    
    # Fine-tune model
    fine_tuner = FeedbackFineTuner(
        pretrained_model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    fine_tuner.fine_tune(processed_feedback)
    
    # Mark feedback as processed
    feedback_ids = [f['id'] for f in feedback_samples]
    collector.mark_as_processed(feedback_ids)


if __name__ == '__main__':
    main()