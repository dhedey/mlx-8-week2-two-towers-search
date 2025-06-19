#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the HackerNews score prediction model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os

from models import TrainingHyperparameters, PooledTwoTowerModelHyperparameters, PooledTwoTowerModel
from trainer import ModelTrainer
from common import select_device

PROJECT_NAME = "week2-two-towers"

# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
SWEEP_CONFIG = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'final_validation_reciprical_rank',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {
            'values': [128, 256]
        },
        'tokenizer': {
            'values': ["week1-word2vec", "pretrained:sentence-transformers/all-MiniLM-L6-v2", "pretrained:gpt2"]
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
    # Initialize wandb run
    wandb.init()
    
    try:
        config = wandb.config
        
        print(f"\n🚀 Starting sweep run")
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
        model_parameters = PooledTwoTowerModelHyperparameters(
            tokenizer=config.tokenizer,
            comparison_embedding_size=config.embedding_size,
            query_tower_hidden_dimensions=[] if not config.include_hidden_layer else [config.embedding_size * 2],
            doc_tower_hidden_dimensions=[] if not config.include_hidden_layer else [config.embedding_size * 2],
            include_layer_norms=True,
        )

        model = PooledTwoTowerModel(
            model_name="sweep-run",
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        trainer = ModelTrainer(model=model.to(device))
        results = trainer.train()
        
        # Log final metrics (wandb.log is also called within train_model)
        wandb.log({
            "final_train_loss": results['last_epoch']['average_loss'],
            "total_epochs": results['total_epochs'],
            "final_validation_reciprical_rank": results['validation']["reciprical_rank"],
            "final_validation_any_relevant_result": results['validation']["any_relevant_result"],
            "final_validation_average_relevance": results['validation']["average_relevance"],
        })
        
        print(f"✅ Sweep run completed! Reciprical Rank: {results['validation']['reciprical_rank']:.4f}")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Ensure wandb run is properly finished
        wandb.finish()


def create_and_run_sweep(config, project_name, count=10):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
    """
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run the sweep
    print(f"🏃 Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
    
    print(f"🎉 Sweep completed!")
    return sweep_id


def run_existing_sweep(sweep_id, project_name, count=10):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
    """
    print(f"🔄 Joining existing sweep: {sweep_id} against {project_name}")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)


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
        run_existing_sweep(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
