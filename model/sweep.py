#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the dual encoder search model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os
import shutil
import tempfile
from pathlib import Path
import time

from models import TrainingHyperparameters, PooledTwoTowerModelHyperparameters, PooledTwoTowerModel, RNNTwoTowerModel, RNNTowerModelHyperparameters
from trainer import ModelTrainer
from common import select_device

PROJECT_NAME = "week2-two-towers"

class SweepManager:
    """
    Manages temporary model storage and tracks the best model across sweep runs.
    Now tracks best per model_type (e.g., 'pooled', 'rnn').
    """
    
    def __init__(self, sweep_id: str, temp_dir: str | None = None):
        self.sweep_id = sweep_id
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix=f"sweep_{sweep_id}_")
        self.best_model_path = {}      # model_type -> path
        self.best_metric_value = {}    # model_type -> metric
        self.best_run_id = {}          # model_type -> run_id
        self.metric_name = "final_validation_reciprical_rank"
        
        print(f"📁 Sweep manager initialized with temp directory: {self.temp_dir}")
    
    def get_model_path(self, run_id: str) -> str:
        """Get the path where a specific run's model should be saved."""
        return os.path.join(self.temp_dir, f"run_{run_id}.pt")
    
    def save_run_model(self, model, run_id: str, metric_value: float, model_type: str):
        """Save a run's model and update best model for the given type if necessary."""
        model_path = self.get_model_path(run_id)
        default_model_path = os.path.join(os.path.dirname(__file__), "data", f"{model.model_name}.pt")
        
        if os.path.exists(default_model_path):
            shutil.copy2(default_model_path, model_path)
            print(f"📁 Model copied to temp directory: {model_path}")
            if (model_type not in self.best_metric_value) or (metric_value > self.best_metric_value[model_type]):
                self.best_metric_value[model_type] = metric_value
                self.best_model_path[model_type] = model_path
                self.best_run_id[model_type] = run_id
                print(f"🏆 New best {model_type} model! Run {run_id} with metric {metric_value:.4f}")
        else:
            print(f"⚠️  Model file not found at expected location: {default_model_path}")
    
    def finalize_sweep(self):
        """Copy the best models for each type to the main data directory."""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        for model_type, best_path in self.best_model_path.items():
            final_model_path = os.path.join(data_dir, f"best-sweep-model-{model_type}.pt")
            shutil.copy2(best_path, final_model_path)
            print(f"✅ Best {model_type} model copied to {final_model_path}")
            print(f"🏆 Best run: {self.best_run_id[model_type]} with metric: {self.best_metric_value[model_type]:.4f}")
        if not self.best_model_path:
            print("⚠️  No models were successfully completed in this sweep")
            return None
        return [os.path.join(data_dir, f"best-sweep-model-{mt}.pt") for mt in self.best_model_path]
    
    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")

# Global sweep manager instance
sweep_manager = None

# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
SWEEP_CONFIG = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'final_validation_reciprical_rank',
        'goal': 'maximize'
    },
    'parameters': {
        'model_type': {
            'values': ['pooled', 'rnn']
        },
        'batch_size': {
            'values': [128, 256]
        },
        'tokenizer': {
            'values': ["week1-word2vec", "pretrained:sentence-transformers/all-MiniLM-L6-v2"]
        },
        "embeddings": {
            'values': ["default-frozen", "default-unfrozen", "learned"]
        },
        "token_boosts": {
            'values': ["none", "learned", "sqrt-inverse-frequency"]
        },
        'include_hidden_layer': {
            'values': [True, False]
        },
        'embedding_size': {
            'values': [32, 64, 128]
        },
        'learning_rate': {
            'min': 0.001,
            'max': 0.03,
            'distribution': 'log_uniform_values'
        },
        'dropout': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'margin': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'epochs': {
            'values': [5]
        }
    }
}

def train_sweep_run():
    """
    Single training run for wandb sweep.
    This function is called by the sweep agent for each hyperparameter combination.
    """
    global sweep_manager
    
    # Initialize wandb run
    wandb.init()
    
    try:
        config = wandb.config
        run_id = f"run_{int(time.time() * 1000)}"
        model_type = config.model_type
        
        print(f"\n🚀 Starting sweep run {run_id}")
        device = select_device()

        match config.embeddings:
            case "default-frozen":
                initial_token_embeddings_kind = "default"
                freeze_embeddings = True
            case "default-unfrozen":
                initial_token_embeddings_kind = "default"
                freeze_embeddings = False
            case "learned":
                initial_token_embeddings_kind = "random"
                freeze_embeddings = False
            case _:
                raise ValueError(f"Unknown embeddings type: {config.embeddings}")
            
        match config.token_boosts:
            case "none":
                initial_token_embeddings_boost_kind = "ones"
                freeze_embedding_boosts = True
            case "learned":
                initial_token_embeddings_boost_kind = "ones"
                freeze_embedding_boosts = False # Learn boosts
            case "sqrt-inverse-frequency":
                if config.tokenizer == "week1-word2vec":
                    initial_token_embeddings_boost_kind = "sqrt-inverse-frequency"
                else:
                    initial_token_embeddings_boost_kind = "ones"
                freeze_embedding_boosts = True
            case _:
                raise ValueError(f"Unknown token boosts type: {config.token_boosts}")

        training_parameters = TrainingHyperparameters(
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            margin=config.margin,
            initial_token_embeddings_kind=initial_token_embeddings_kind,
            freeze_embeddings=freeze_embeddings,
            initial_token_embeddings_boost_kind=initial_token_embeddings_boost_kind,
            freeze_embedding_boosts=freeze_embedding_boosts,
        )

        # Determine hidden layer dimensions based on model type and configuration
        if config.model_type == 'pooled':
            hidden_dimensions = [] if not config.include_hidden_layer else [config.embedding_size * 2]
            model_parameters = PooledTwoTowerModelHyperparameters(
                tokenizer=config.tokenizer,
                comparison_embedding_size=config.embedding_size,
                query_tower_hidden_dimensions=hidden_dimensions,
                doc_tower_hidden_dimensions=hidden_dimensions,
                include_layer_norms=True,
            )

            model = PooledTwoTowerModel(
                model_name=f"run_{run_id}",
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        elif config.model_type == 'rnn':
            # For RNN models, always include hidden layers for the RNN processing
            hidden_dimensions = [config.embedding_size * 2, config.embedding_size]
            model_parameters = RNNTowerModelHyperparameters(
                tokenizer=config.tokenizer,
                comparison_embedding_size=config.embedding_size,
                query_tower_hidden_dimensions=hidden_dimensions,
                doc_tower_hidden_dimensions=hidden_dimensions,
                include_layer_norms=True,
            )

            model = RNNTwoTowerModel(
                model_name=f"run_{run_id}",
                training_parameters=training_parameters,
                model_parameters=model_parameters,
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        trainer = ModelTrainer(model=model.to(device))
        results = trainer.train()
        
        # Get the final metric value
        final_metric_value = results['validation']["reciprical_rank"]
        
        # Save the model using sweep manager if available
        if sweep_manager is not None:
            sweep_manager.save_run_model(model, run_id, final_metric_value, model_type)
        
        # Log final metrics (wandb.log is also called within train_model)
        wandb.log({
            "final_train_loss": results['last_epoch']['average_loss'],
            "total_epochs": results['total_epochs'],
            "final_validation_reciprical_rank": results['validation']["reciprical_rank"],
            "final_validation_any_relevant_result": results['validation']["any_relevant_result"],
            "final_validation_average_relevance": results['validation']["average_relevance"],
        })
        
        print(f"✅ Sweep run {run_id} completed! Reciprical Rank: {final_metric_value:.4f}")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Ensure wandb run is properly finished
        wandb.finish()


def create_and_run_sweep(config, project_name, count=10, final_model_name="best-sweep-model"):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
        final_model_name: Name for the best model file
    """
    global sweep_manager
    
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Initialize sweep manager
    sweep_manager = SweepManager(sweep_id)
    
    try:
        # Run the sweep
        print(f"🏃 Starting sweep agent with {count} runs...")
        wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
        
        # Finalize the sweep by copying the best models
        final_model_paths = sweep_manager.finalize_sweep()
        
        print(f"🎉 Sweep completed!")
        if final_model_paths:
            for model_type, final_model_path in zip(sweep_manager.best_model_path, final_model_paths):
                print(f"🏆 Best {model_type} model saved as: {final_model_path}")
        
        return sweep_id
        
    finally:
        # Clean up temporary files
        if sweep_manager:
            sweep_manager.cleanup()
            sweep_manager = None


def run_existing_sweep(sweep_id, project_name, count=10, final_model_name="best-sweep-model"):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
        final_model_name: Name for the best model file
    """
    global sweep_manager
    
    print(f"🔄 Joining existing sweep: {sweep_id} against {project_name}")
    
    # Initialize sweep manager for existing sweep
    sweep_manager = SweepManager(sweep_id)
    
    try:
        wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
        
        # Finalize the sweep by copying the best models
        final_model_paths = sweep_manager.finalize_sweep()
        
        print(f"🎉 Sweep runs completed!")
        if final_model_paths:
            for model_type, final_model_path in zip(sweep_manager.best_model_path, final_model_paths):
                print(f"🏆 Best {model_type} model saved as: {final_model_path}")
            
    finally:
        # Clean up temporary files
        if sweep_manager:
            sweep_manager.cleanup()
            sweep_manager = None


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps')
    parser.add_argument('--project', default=PROJECT_NAME,
                        help=f'W&B project name (default: {PROJECT_NAME})')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-id', type=str,
                        help='Join existing sweep by ID instead of creating new one')
    parser.add_argument('--final-model-name', type=str, default='best-sweep-model',
                        help='Name for the best model file (default: best-sweep-model)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show the configuration without running')
    
    args = parser.parse_args()
    
    # Select configuration
    config = SWEEP_CONFIG
    print("📋 Using default Bayesian optimization configuration")
    
    if args.dry_run:
        print("\n🔍 Sweep configuration:")
        import json
        print(json.dumps(config, indent=2))
        return
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Run sweep
    if args.sweep_id:
        run_existing_sweep(args.sweep_id, args.project, args.count, args.final_model_name)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count, args.final_model_name)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
