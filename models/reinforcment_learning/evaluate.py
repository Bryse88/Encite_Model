"""
Evaluation script for RL scheduler.

This script evaluates the performance of a trained policy on a set of test cases.
"""

import os
import logging
import argparse
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, date, time as dt_time
from google.cloud import storage
from firebase_admin import firestore, initialize_app
import matplotlib.pyplot as plt
import pandas as pd

from models.reinforcment_learning.env.schedule_env import ScheduleEnv
from models.reinforcment_learning.train import PPOTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyEvaluator:
    """
    Evaluate RL policies for schedule optimization.
    
    This class implements methods to evaluate policies on various metrics
    and compare different policies.
    """
    
    def __init__(self, 
                 config_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to config file
            device: Device to use for evaluation
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.device = device
        self.config_path = config_path
        
        # Initialize Firebase
        initialize_app()
        self.db = firestore.client()
        
    def load_model(self, model_path: str) -> PPOTrainer:
        """
        Load a trained policy.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded trainer with policy
        """
        trainer = PPOTrainer(device=self.device)
        
        # Create test environment
        env = self._create_test_env()
        
        # Load model
        trainer.load(model_path, env)
        
        return trainer, env
    
    def _create_test_env(self) -> ScheduleEnv:
        """
        Create a test environment for policy evaluation.
        
        Returns:
            Schedule environment
        """
        return ScheduleEnv(
            user_id=self.config['evaluation']['test_user_id'],
            date=datetime.strptime(self.config['evaluation']['test_date'], '%Y-%m-%d').date(),
            start_time=datetime.strptime(self.config['evaluation']['test_start_time'], '%H:%M').time(),
            end_time=datetime.strptime(self.config['evaluation']['test_end_time'], '%H:%M').time(),
            location=tuple(self.config['evaluation']['test_location']),
            preferences=self.config['evaluation']['test_preferences'],
            budget=self.config['evaluation'].get('test_budget'),
            transportation_modes=self.config['evaluation'].get('test_transportation_modes'),
            config_path=self.config_path
        )
    
    def evaluate_model(self, 
                       trainer: PPOTrainer, 
                       env: ScheduleEnv, 
                       n_episodes: int = 10,
                       deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate a model on multiple metrics.
        
        Args:
            trainer: Trainer with loaded policy
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model for {n_episodes} episodes")
        
        # Metrics to track
        metrics = {
            'episode_rewards': [],
            'schedule_lengths': [],
            'total_travel_times': [],
            'budget_utilization': [],
            'time_utilization': [],
            'preference_match_scores': [],
            'category_variety': []
        }
        
        # Run evaluation episodes
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Track additional metrics
            schedule = []
            total_travel_time = 0
            
            while not done:
                # Get action from policy
                tensor_state = trainer._normalize_state(state)
                action, _ = trainer.policy.act(tensor_state, deterministic=deterministic)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Track action details if not "end schedule" action
                if action < len(env.candidate_places):
                    schedule.append(env.candidate_places[action])
                    
                    # Track travel time if available in info
                    if 'travel_time_minutes' in info:
                        total_travel_time += info['travel_time_minutes']
                
                # Move to next state
                state = next_state
            
            # Record episode metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['schedule_lengths'].append(len(schedule))
            metrics['total_travel_times'].append(total_travel_time)
            
            # Calculate budget utilization
            initial_budget = env.budget
            remaining_budget = env.remaining_budget
            budget_utilization = (initial_budget - remaining_budget) / initial_budget if initial_budget > 0 else 0
            metrics['budget_utilization'].append(budget_utilization)
            
            # Calculate time utilization
            start_dt = datetime.combine(env.date, env.start_time)
            end_dt = datetime.combine(env.date, env.end_time)
            current_dt = datetime.combine(env.date, env.current_time)
            total_available_minutes = (end_dt - start_dt).total_seconds() / 60
            used_minutes = (current_dt - start_dt).total_seconds() / 60
            time_utilization = used_minutes / total_available_minutes
            metrics['time_utilization'].append(time_utilization)
            
            # Calculate preference match score
            preference_match = 0
            categories = set()
            for place in schedule:
                for category, weight in env.preferences.items():
                    if category in place.get('categories', []):
                        preference_match += weight
                categories.update(place.get('categories', []))
                        
            metrics['preference_match_scores'].append(preference_match)
            
            # Calculate category variety
            category_variety = len(categories) / max(len(env.preferences), 1)
            metrics['category_variety'].append(category_variety)
            
            logger.info(f"Episode {episode+1}: reward = {episode_reward:.2f}, "
                      f"length = {len(schedule)}, time utilization = {time_utilization:.2f}")
        
        # Calculate summary statistics
        summary = {}
        for key, values in metrics.items():
            summary[f'mean_{key}'] = np.mean(values)
            summary[f'std_{key}'] = np.std(values)
            summary[f'min_{key}'] = np.min(values)
            summary[f'max_{key}'] = np.max(values)
            
        return summary
    
    def compare_models(self, 
                       model_paths: List[str], 
                       model_names: Optional[List[str]] = None,
                       n_episodes: int = 10) -> pd.DataFrame:
        """
        Compare multiple models on evaluation metrics.
        
        Args:
            model_paths: List of paths to model files
            model_names: Optional list of model names
            n_episodes: Number of episodes to evaluate each model
            
        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(model_paths))]
            
        if len(model_names) != len(model_paths):
            raise ValueError("model_names and model_paths must have the same length")
            
        results = []
        
        for name, path in zip(model_names, model_paths):
            logger.info(f"Evaluating {name} from {path}")
            
            trainer, env = self.load_model(path)
            metrics = self.evaluate_model(trainer, env, n_episodes=n_episodes)
            
            # Add model name to metrics
            metrics['model_name'] = name
            results.append(metrics)
            
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def visualize_results(self, results_df: pd.DataFrame, output_dir: str = 'results'):
        """
        Visualize comparison results.
        
        Args:
            results_df: DataFrame with comparison results
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot key metrics
        metrics_to_plot = [
            'mean_episode_rewards',
            'mean_schedule_lengths',
            'mean_time_utilization',
            'mean_budget_utilization',
            'mean_preference_match_scores',
            'mean_category_variety'
        ]
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart
            ax = results_df.plot.bar(x='model_name', y=metric, legend=False)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points'
                )
            
            # Format plot
            plt.title(f"Comparison of {metric.replace('mean_', '').replace('_', ' ').title()}")
            plt.xlabel("Model")
            plt.ylabel(metric.replace('mean_', '').replace('_', ' ').title())
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{metric}.png"))
            plt.close()
            
        # Create radar chart for comparing models across all metrics
        metrics_for_radar = [
            'mean_episode_rewards',
            'mean_schedule_lengths',
            'mean_time_utilization',
            'mean_budget_utilization',
            'mean_preference_match_scores',
            'mean_category_variety'
        ]
        
        # Normalize metrics for radar chart
        norm_df = results_df.copy()
        for metric in metrics_for_radar:
            min_val = norm_df[metric].min()
            max_val = norm_df[metric].max()
            if max_val > min_val:
                norm_df[metric] = (norm_df[metric] - min_val) / (max_val - min_val)
            else:
                norm_df[metric] = 0.5  # If all values are the same
                
        # Plot radar chart
        N = len(metrics_for_radar)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for _, row in norm_df.iterrows():
            values = row[metrics_for_radar].values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=row['model_name'])
            ax.fill(angles, values, alpha=0.1)
            
        # Set labels
        labels = [m.replace('mean_', '').replace('_', ' ').title() for m in metrics_for_radar]
        labels += labels[:1]  # Close the loop
        plt.xticks(angles, labels)
        
        plt.title("Model Comparison Across Metrics")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_radar.png"))
        plt.close()
        
        logger.info(f"Saved visualizations to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RL policies for schedule optimization')
    
    parser.add_argument('--config', type=str, default='configs/rl_config.yaml',
                       help='Path to config file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--compare', type=str, nargs='+', default=[],
                       help='Paths to additional models to compare')
    parser.add_argument('--names', type=str, nargs='+', default=[],
                       help='Names for models being compared')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create evaluator
    evaluator = PolicyEvaluator(config_path=args.config)
    
    # Load and evaluate main model
    trainer, env = evaluator.load_model(args.model_path)
    metrics = evaluator.evaluate_model(trainer, env, n_episodes=args.episodes)
    
    logger.info("Evaluation results:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    # Compare with other models if specified
    if args.compare:
        all_models = [args.model_path] + args.compare
        
        if not args.names:
            model_names = [os.path.basename(path).split('.')[0] for path in all_models]
        else:
            if len(args.names) != len(all_models):
                logger.warning("Number of names does not match number of models, using default names")
                model_names = [os.path.basename(path).split('.')[0] for path in all_models]
            else:
                model_names = args.names
                
        results_df = evaluator.compare_models(
            model_paths=all_models,
            model_names=model_names,
            n_episodes=args.episodes
        )
        
        # Save results to CSV
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "model_comparison.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved comparison results to {results_path}")
        
        # Visualize results
        evaluator.visualize_results(results_df, output_dir=args.output_dir)


if __name__ == '__main__':
    main()