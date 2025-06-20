# Run as uv run -m model.continue_train
import argparse
from .models import DualEncoderModel
from .trainer import ModelTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a dual encoder model for text search')
    parser.add_argument(
        '--model',
        type=str,
        default="fixed-boosted-word2vec-pooled",
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--immediate-validation',
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    model, training_state = DualEncoderModel.load_for_training(
        model_name=args.model
    )

    trainer = ModelTrainer(
        model=model,
        continuation=training_state,
        override_to_epoch=args.end_epoch,
        immediate_validation=args.immediate_validation,
    )
    trainer.train()



        
