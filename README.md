# Machine Learning Institute - Week 2 - Search ranking

This week, we are using the [MS Marco](https://huggingface.co/datasets/microsoft/ms_marco) dataset to predict documents for search queries.

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

# Model Training

Run model training with various options:

## Pooled Models (Fast, Memory Efficient)

Pooled models use `EmbeddingBag` to aggregate token embeddings, making them faster and more memory efficient.

### Fixed Boosted Word2Vec (Pooled)
```bash
uv run ./model/start_train.py --model fixed-boosted-word2vec-pooled
```

### Learned Boosted Word2Vec (Pooled)
```bash
uv run ./model/start_train.py --model learned-boosted-word2vec-pooled
```

### Learned Boosted MiniLM (Pooled)
```bash
uv run ./model/start_train.py --model learned-boosted-mini-lm-pooled
```

## RNN Models (Sequential Processing)

RNN models use recurrent neural networks to process token sequences, capturing sequential information better.

### Fixed Boosted Word2Vec (RNN)
```bash
uv run ./model/start_train.py --model fixed-boosted-word2vec-rnn
```

### Learned Boosted Word2Vec (RNN)
```bash
uv run ./model/start_train.py --model learned-boosted-word2vec-rnn
```

### Learned Boosted MiniLM (RNN)
```bash
uv run ./model/start_train.py --model learned-boosted-mini-lm-rnn
```

## Model Architecture Comparison

| Model Type | Token Processing | Memory Usage | Speed | Sequential Info |
|------------|------------------|--------------|-------|-----------------|
| Pooled | EmbeddingBag aggregation | Low | Fast | Limited |
| RNN | Recurrent neural network | High | Slower | Better |

## Training Parameters

All models use the same training configuration:
- **Batch size**: 128
- **Epochs**: 20
- **Learning rate**: 0.002
- **Dropout**: 0.3
- **Margin**: 0.4 (for triplet loss)
- **Output embedding size**: 64

## Validation

Models are validated every epoch using top-k recall metrics on the validation set.
