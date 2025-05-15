"""
Training script for the RL scheduler.

This script implements PPO (Proximal Policy Optimization) algorithm
to train the policy for optimizing schedules.
"""

import os
import time
import logging
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, date, time as dt_time
from google.cloud import storage
from firebase_admin import firestore, initialize_app
import gym

from models.reinforcement_learning.env.schedule_env import ScheduleEnv
from models.reinforcement_learning.policies.mlp_policy import MLPPolicy
from models.reinforcement_learning.policies.transformer_policy import TransformerPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for schedule optimization.
    
    This class implements the PPO algorithm to train policies for the
    scheduling environment, with support for both MLP and Transformer policies.
    """
    
    def __init__(self,
                 policy_type: str = 'transformer',
                 state_dim: int = 128,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the PPO trainer.
        
        Args:
            policy_type: Type of policy to use ('mlp' or 'transformer')
            state_dim: Dimension of the policy's hidden state
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            device: Device to use for training
        """
        self.policy_type = policy_type
        self.state_dim = state_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
        # Initialize Firebase
        initialize_app()
        self.db = firestore.client()
        
        # Policy is initialized with environment parameters during training
        self.policy = None
        self.optimizer = None
        
    def _init_policy(self, env: ScheduleEnv):
        """
        Initialize policy based on environment.
        
        Args:
            env: Schedule environment
        """
        action_dim = env.action_space.n
        
        if self.policy_type == 'mlp':
            self.policy = MLPPolicy(
                state_dim=self.state_dim,
                action_dim=action_dim,
                hidden_dims=[256, 256]
            ).to(self.device)
        elif self.policy_type == 'transformer':
            self.policy = TransformerPolicy(
                state_dim=self.state_dim,
                action_dim=action_dim,
                embedding_dim=64,
                nhead=4,
                num_layers=2,
                dropout=0.1
            ).to(self.device)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.actor_lr
        )
        
    def _normalize_state(self, state: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy state to PyTorch tensors and add batch dimension if needed.
        
        Args:
            state: Dictionary of state components
            
        Returns:
            Dictionary of state components as PyTorch tensors
        """
        tensor_state = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                # Add batch dimension if not present
                if len(value.shape) == 1:
                    value = np.expand_dims(value, 0)
                tensor_state[key] = torch.FloatTensor(value).to(self.device)
            else:
                tensor_state[key] = value
        return tensor_state
    
    def collect_rollouts(self, 
                         env: ScheduleEnv, 
                         n_steps: int = 1024,
                         n_episodes: int = 16) -> List[Dict[str, Any]]:
        """
        Collect experience using current policy.
        
        Args:
            env: Schedule environment
            n_steps: Minimum number of steps to collect
            n_episodes: Minimum number of episodes to collect
            
        Returns:
            List of trajectories (episodes)
        """
        trajectories = []
        total_steps = 0
        episodes = 0
        
        while total_steps < n_steps or episodes < n_episodes:
            # Reset the environment
            state = env.reset()
            done = False
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            }
            
            # Run one episode
            episode_steps = 0
            while not done:
                tensor_state = self._normalize_state(state)
                
                # Get action from policy
                with torch.no_grad():
                    policy, value = self.policy(tensor_state)
                    dist = torch.distributions.Categorical(policy)
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor([action]).to(self.device)).item()
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['values'].append(value.item())
                trajectory['log_probs'].append(log_prob)
                trajectory['dones'].append(done)
                
                # Move to next state
                state = next_state
                episode_steps += 1
                total_steps += 1
            
            episodes += 1
            trajectories.append(trajectory)
            logger.info(f"Collected episode {episodes} with {episode_steps} steps and reward {sum(trajectory['rewards'])}")
        
        logger.info(f"Collected {len(trajectories)} episodes with {total_steps} steps total")
        return trajectories
    
    def compute_advantages_and_returns(self, trajectory: Dict[str, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using GAE and returns.
        
        Args:
            trajectory: Dictionary containing episode data
            
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = np.array(trajectory['rewards'])
        values = np.array(trajectory['values'])
        dones = np.array(trajectory['dones'])
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_value = 0  # Terminal state has value 0
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]
            
            returns[t] = rewards[t] + self.gamma * next_non_terminal * (returns[t + 1] if t < len(rewards) - 1 else 0)
        
        return advantages, returns
    
    def update(self, trajectories: List[Dict[str, List]]) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            trajectories: List of trajectories (episodes)
            
        Returns:
            Dictionary of training metrics
        """
        # Process all trajectories
        states_batch = []
        actions_batch = []
        old_log_probs_batch = []
        advantages_batch = []
        returns_batch = []
        
        for trajectory in trajectories:
            advantages, returns = self.compute_advantages_and_returns(trajectory)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            states_batch.extend(trajectory['states'])
            actions_batch.extend(trajectory['actions'])
            old_log_probs_batch.extend(trajectory['log_probs'])
            advantages_batch.extend(advantages)
            returns_batch.extend(returns)
        
        # Convert to tensors
        batch_size = len(states_batch)
        indices = np.arange(batch_size)
        
        # Training loop
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'total_loss': 0,
            'approx_kl': 0,
            'clip_fraction': 0
        }
        
        for _ in range(self.n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, batch_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, batch_size)
                mbatch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = [states_batch[i] for i in mbatch_indices]
                mb_actions = torch.LongTensor([actions_batch[i] for i in mbatch_indices]).to(self.device)
                mb_old_log_probs = torch.FloatTensor([old_log_probs_batch[i] for i in mbatch_indices]).to(self.device)
                mb_advantages = torch.FloatTensor([advantages_batch[i] for i in mbatch_indices]).to(self.device)
                mb_returns = torch.FloatTensor([returns_batch[i] for i in mbatch_indices]).to(self.device)
                
                # Forward pass
                mb_tensor_states = []
                for state in mb_states:
                    mb_tensor_states.append(self._normalize_state(state))
                
                # Combine batch dimensions for each state component
                batch_state = {}
                for key in mb_tensor_states[0].keys():
                    batch_state[key] = torch.cat([s[key] for s in mb_tensor_states], dim=0)
                
                # Get policy and value predictions
                policy, values = self.policy(batch_state)
                
                # Get log probs for actions
                dist = torch.distributions.Categorical(policy)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                
                # Compute losses
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * clipped_ratio
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['total_loss'] += loss.item()
                metrics['approx_kl'] += ((ratio - 1) - torch.log(ratio)).mean().item()
                metrics['clip_fraction'] += ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
        
        # Average metrics
        n_updates = self.n_epochs * (batch_size // self.batch_size + 1)
        for key in metrics:
            metrics[key] /= n_updates
            
        return metrics
    
    def train(self, 
              env: ScheduleEnv,
              n_iterations: int,
              n_steps_per_iteration: int = 1024,
              n_episodes_per_iteration: int = 16,
              save_freq: int = 10,
              model_dir: str = 'models/reinforcement_learning/saved_models',
              eval_freq: int = 5,
              log_freq: int = 1):
        """
        Train the policy.
        
        Args:
            env: Schedule environment
            n_iterations: Number of training iterations
            n_steps_per_iteration: Steps per iteration
            n_episodes_per_iteration: Episodes per iteration
            save_freq: Frequency of saving models
            model_dir: Directory to save models
            eval_freq: Frequency of evaluation
            log_freq: Frequency of logging
        """
        # Initialize policy if not already done
        if self.policy is None:
            self._init_policy(env)
            
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Training loop
        start_time = time.time()
        best_mean_reward = -float('inf')
        
        for iteration in range(1, n_iterations + 1):
            logger.info(f"Starting iteration {iteration}/{n_iterations}")
            
            # Collect experience
            trajectories = self.collect_rollouts(
                env, 
                n_steps=n_steps_per_iteration,
                n_episodes=n_episodes_per_iteration
            )
            
            # Compute rewards
            episode_rewards = [sum(traj['rewards']) for traj in trajectories]
            mean_reward = np.mean(episode_rewards)
            
            # Update policy
            metrics = self.update(trajectories)
            
            # Log progress
            if iteration % log_freq == 0:
                end_time = time.time()
                logger.info(f"Iteration {iteration}: mean reward = {mean_reward:.2f}, time = {end_time - start_time:.2f}s")
                logger.info(f"Metrics: {metrics}")
                start_time = end_time
                
                # Log to Firestore
                self.db.collection('training_logs').document(f'{self.policy_type}_iteration_{iteration}').set({
                    'iteration': iteration,
                    'mean_reward': float(mean_reward),
                    'metrics': {k: float(v) for k, v in metrics.items()},
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
            
            # Evaluate policy
            if iteration % eval_freq == 0:
                eval_rewards = self.evaluate(env, n_episodes=10)
                logger.info(f"Evaluation: mean reward = {np.mean(eval_rewards):.2f}")
                
                # Save best model
                if np.mean(eval_rewards) > best_mean_reward:
                    best_mean_reward = np.mean(eval_rewards)
                    self.save(os.path.join(model_dir, f'best_{self.policy_type}_policy.pt'))
                    logger.info(f"Saved best model with mean reward {best_mean_reward:.2f}")
            
            # Save model periodically
            if iteration % save_freq == 0:
                self.save(os.path.join(model_dir, f'{self.policy_type}_policy_iter_{iteration}.pt'))
                logger.info(f"Saved model at iteration {iteration}")
    
    def evaluate(self, env: ScheduleEnv, n_episodes: int = 10) -> List[float]:
        """
        Evaluate the policy.
        
        Args:
            env: Schedule environment
            n_episodes: Number of episodes to evaluate
            
        Returns:
            List of episode rewards
        """
        rewards = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from policy (deterministic)
                tensor_state = self._normalize_state(state)
                action, _ = self.policy.act(tensor_state, deterministic=True)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # Move to next state
                state = next_state
            
            rewards.append(episode_reward)
            
        return rewards
    
    def save(self, path: str):
        """
        Save policy to disk.
        
        Args:
            path: Path to save policy
        """
        torch.save({
            'policy_type': self.policy_type,
            'state_dict': self.policy.state_dict(),
            'state_dim': self.state_dim,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda
        }, path)
        
    def load(self, path: str, env: Optional[ScheduleEnv] = None):
        """
        Load policy from disk.
        
        Args:
            path: Path to load policy from
            env: Optional environment to initialize policy
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_type = checkpoint['policy_type']
        self.state_dim = checkpoint['state_dim']
        self.gamma = checkpoint.get('gamma', 0.99)
        self.gae_lambda = checkpoint.get('gae_lambda', 0.95)
        
        # Initialize policy
        if env is not None and self.policy is None:
            self._init_policy(env)
            
        # Load state dict
        self.policy.load_state_dict(checkpoint['state_dict'])


    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Train RL policy for schedule optimization')
    
        parser.add_argument('--policy-type', type=str, default='transformer', choices=['mlp', 'transformer'],
                       help='Type of policy to use')
        parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations')
        parser.add_argument('--steps-per-iteration', type=int, default=1024,
                       help='Number of steps per iteration')
        parser.add_argument('--episodes-per-iteration', type=int, default=16,
                       help='Number of episodes per iteration')
        parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
        parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
        parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
        parser.add_argument('--save-freq', type=int, default=10,
                       help='Frequency of saving models')
        parser.add_argument('--model-dir', type=str, default='models/reinforcement_learning/saved_models',
                       help='Directory to save models')
        parser.add_argument('--config', type=str, default='configs/rl_config.yaml',
                       help='Path to config file')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
        return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create environment
    env = ScheduleEnv(
        user_id=config['training']['user_id'],
        date=datetime.strptime(config['training']['date'], '%Y-%m-%d').date(),
        start_time=datetime.strptime(config['training']['start_time'], '%H:%M').time(),
        end_time=datetime.strptime(config['training']['end_time'], '%H:%M').time(),
        location=tuple(config['training']['location']),
        preferences=config['training']['preferences'],
        budget=config['training'].get('budget'),
        transportation_modes=config['training'].get('transportation_modes'),
        config_path=args.config
    )
    
    # Create trainer
    trainer = PPOTrainer(
        policy_type=args.policy_type,
        state_dim=config['model']['state_dim'],
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate * 2,
        gamma=args.gamma,
        gae_lambda=config['model'].get('gae_lambda', 0.95),
        clip_range=config['model'].get('clip_range', 0.2),
        value_loss_coef=config['model'].get('value_loss_coef', 0.5),
        entropy_coef=config['model'].get('entropy_coef', 0.01),
        max_grad_norm=config['model'].get('max_grad_norm', 0.5),
        batch_size=args.batch_size,
        n_epochs=config['model'].get('n_epochs', 10),
        device=args.device
    )
    
    # Train
    trainer.train(
        env=env,
        n_iterations=args.iterations,
        n_steps_per_iteration=args.steps_per_iteration,
        n_episodes_per_iteration=args.episodes_per_iteration,
        save_freq=args.save_freq,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()