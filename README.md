# Encite AI Scheduling System

## Overview

Encite is an intelligent AI-powered scheduling system that creates personalized itineraries and recommendations based on user preferences, context, and real-time data. The system uses a hybrid approach combining Graph Neural Networks, Two-Tower Recommendation Models, and Reinforcement Learning to deliver optimal schedules.

## Key Features

- **Personalized Recommendations**: Tailored suggestions based on user history, preferences, and social connections
- **Dynamic Scheduling**: Optimizes itineraries considering time, budget, weather, and travel constraints
- **Continuous Learning**: Improves based on user feedback and interaction patterns
- **Real-time Adaptability**: Adjusts to changing conditions like weather and venue availability

## System Architecture

### 1. Graph Neural Network Transformer

A heterogeneous graph neural network that captures complex relationships between users, places, and events.

- **Framework**: PyTorch Geometric (PyG)
- **Model**: Heterogeneous Graph Transformer (HGT)
- **Node Types**: Users, Places, Events
- **Edge Types**: Visits, Friendships, Interests, etc.

### 2. Two-Tower Matching Model

Fast retrieval system for generating candidate recommendations.

- **Framework**: TensorFlow Recommenders
- **Architecture**: Dual encoder (User Tower and Place Tower)
- **Output**: Embeddings for similarity scoring

### 3. RL-based Schedule Generation

Uses reinforcement learning to optimize schedule creation with multiple constraints.

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: Custom gym environment with realistic state/action spaces
- **Reward Function**: Multi-objective optimization for user satisfaction

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- TensorFlow 2.6+
- Firebase Admin SDK
- Stable-Baselines3
- FastAPI
- Flutter (for mobile app)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/encite-ai-scheduler.git
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Firebase credentials:
   ```
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
   ```

4. Set up API keys in config.py:
   ```python
   GOOGLE_MAPS_API_KEY = "your-api-key"
   WEATHER_API_KEY = "your-api-key"
   TICKETMASTER_API_KEY = "your-api-key"
   ```

### Project Structure

```
encite-ai-platform/
│
├── README.md
├── .env.example
├── pyproject.toml             # Modern Python dependency management
├── setup.py                   # For installing as a package
├── docker-compose.yml
│
├── .github/
│   └── workflows/
│       ├── deploy.yml         # CI/CD for deploying services
│       ├── test.yml           # Automated test runs
│       └── model_train.yml    # Scheduled model retraining
│
├── apps/
│   ├── firebase_functions/
│   │   ├── functions/
│   │   │   ├── schedule/
│   │   │   ├── feedback/
│   │   │   └── auth/
│   │   └── firestore.rules
│   │
│   ├── two_tower_api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── app/
│   │   │   ├── api/
│   │   │   ├── core/
│   │   │   ├── models/
│   │   │   └── services/
│   │   ├── tests/
│   │   └── requirements.txt
│   │
│   ├── rl_scheduler_api/     # Similar structure as two_tower_api
│   └── feedback_logger/      # Service for handling user feedback
│
├── models/
│   ├── graph_transformer/
│   │   ├── train.py          # Training loop 
│   │   ├── model.py          # HGT definition
│   │   ├── dataset_loader.py # Loads graph from Firestore
│   │   ├── evaluation.py     # Model evaluation metrics
│   │   ├── configs/          # Configuration files
│   │   └── utils.py
│   │
│   ├── two_tower/
│   │   ├── train.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── evaluation.py
│   │   ├── configs/
│   │   └── export_embeddings.py
│   │
│   └── reinforcement_learning/
│       ├── env/
│       │   ├── schedule_env.py       # ScheduleEnv gym environment
│       │   └── reward_functions.py   # Custom reward calculations
│       ├── train.py                  # PPO training loop
│       ├── policies/
│       │   ├── mlp_policy.py
│       │   └── transformer_policy.py # Alternative architecture
│       ├── fine_tune.py              # Feedback-based fine-tuning
│       ├── configs/
│       └── evaluate.py               # Evaluate policy performance
│
├── services/
│   ├── api_clients/
│   │   ├── google/
│   │   │   ├── places.py
│   │   │   ├── distance_matrix.py
│   │   │   └── maps.py
│   │   ├── ticketmaster.py
│   │   ├── openweather.py
│   │   └── uber.py
│   │
│   ├── schedule_pipeline/
│   │   ├── generate_schedule.py    # Orchestrates entire flow
│   │   ├── preprocess.py           # Clean + normalize inputs
│   │   ├── rank_candidates.py      # Two-Tower matching
│   │   ├── build_schedule.py       # RL-based sequence generation
│   │   └── postprocess.py          # Format schedule for response
│   │
│   ├── embedding_updater/
│   │   ├── export_firestore_graph.py
│   │   ├── update_embeddings.py    # Triggers model & writes back to DB
│   │   └── cron_config.yaml        # GCP Cloud Scheduler config
│   │
│   └── data_collectors/
│       ├── places_updater.py     # Daily update of place data
│       ├── events_collector.py   # Pulls from Ticketmaster
│       └── weather_updater.py    # Pre-fetches weather forecasts
│
├── database/
│   ├── firestore/
│   │   ├── schema/
│   │   │   ├── users.json
│   │   │   ├── places.json
│   │   │   └── events.json
│   │   └── seed_data/
│   │       ├── sample_users.json
│   │       ├── sample_places.json
│   │       └── sample_events.json
│   │
│   ├── pinecone/
│   │   └── index_creation.py    # Vector DB setup
│   │
│   └── migrations/
│       └── add_embeddings_field.py
│
├── configs/
│   ├── rl_config.yaml
│   ├── hgt_config.yaml
│   ├── api_config.yaml
│   ├── firebase_config.json
│   ├── logging_config.yaml
│   ├── data_source_config.yaml   # Location-Aware Data Source Routing
│   └── environment/
│       ├── development.env
│       ├── staging.env
│       └── production.env
│
├── notebooks/
│   ├── model_exploration/
│   │   ├── graph_visualization.ipynb
│   │   ├── embedding_exploration.ipynb
│   │   └── reward_function_tuning.ipynb
│   └── data_analysis/
│       ├── user_behavior.ipynb
│       ├── place_popularity.ipynb
│       └── schedule_quality.ipynb
│
└── tests/
    ├── unit/
    │   ├── models/
    │   │   ├── test_graph.py
    │   │   ├── test_two_tower.py
    │   │   └── test_policy.py
    │   ├── services/
    │   │   ├── test_api_clients.py
    │   │   └── test_schedule_pipeline.py
    │   └── env/
    │       └── test_schedule_env.py
    │
    ├── integration/
    │   ├── test_generate_schedule_flow.py
    │   ├── test_user_feedback_loop.py
    │   └── test_embedding_updates.py
    │
    └── e2e/
        └── test_full_system.py
```

## Running Services Locally

### 1. Start the GNN Service

```bash
cd graph-service
uvicorn service:app --reload --port 8000
```

### 2. Start the Two-Tower Service

```bash
cd two-tower-service
uvicorn service:app --reload --port 8001
```

### 3. Start the RL Scheduler Service

```bash
cd rl-scheduler-service
uvicorn service:app --reload --port 8002
```

### 4. Deploy Firebase Functions

```bash
cd firebase
firebase deploy --only functions
```

### 5. Run the Flutter App

```bash
cd flutter-app
flutter run
```

## Deployment

### Cloud Run Deployment

1. Build and deploy the GNN service:
   ```bash
   gcloud builds submit --tag gcr.io/your-project/graph-service graph-service/
   gcloud run deploy graph-service --image gcr.io/your-project/graph-service
   ```

2. Repeat for the two-tower and RL scheduler services.

### Continuous Integration

The repository includes GitHub Actions workflows for CI/CD in `.github/workflows/deploy.yml`.

## Monitoring and Fine-tuning

### Scheduled Jobs

- **Daily**: Update place data from external APIs
- **Weekly**: Export feedback for offline training 
- **Weekly**: Fine-tune RL model based on user feedback

### Evaluation Metrics

Run the evaluation script to measure system performance:

```bash
python scripts/evaluation.py --output-file metrics.json
```

Key metrics include:
- Average activities per schedule
- Travel time ratio
- Budget utilization
- Diversity score
- Weather compatibility
- User feedback score

## Extending the System

### Adding New Data Sources

1. Create an API client in `scripts/data_processing.py`
2. Update the graph schema in `graph-service/model.py`
3. Create a data importer in Firebase Functions

### Customizing the RL Environment

1. Modify reward functions in `rl-scheduler-service/environment.py`
2. Adjust state and action spaces as needed
3. Retrain the model with `python rl-scheduler-service/training.py`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [TensorFlow Recommenders](https://github.com/tensorflow/recommenders)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Flutter](https://flutter.dev/)
