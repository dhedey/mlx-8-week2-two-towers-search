# Machine Learning Institute - Week 2 - Search ranking

This week, we are using the [MS Marco](https://huggingface.co/datasets/microsoft/ms_marco) dataset to predict documents for search queries.

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

Run model training with:
```bash
uv run ./model/train.py
```
