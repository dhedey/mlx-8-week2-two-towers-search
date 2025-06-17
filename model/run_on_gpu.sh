# ssh -p [PORT] -i ~/.ssh/mli_computa root@[IP]

apt-get update
apt-get install git
apt-get install git-lfs
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

cd ~
git clone https://github.com/dhedey/mlx-8-week2-two-towers-search
cd mlx-8-week2-two-towers-search

# uv run ./model/train.py --batch-size 1024
# uv run ./model/sweep.py --count 20