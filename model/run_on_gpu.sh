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

# Change if you're someone else!
git config --global user.email "mli@david-edey.com"
git config --global user.name "David Edey"

# uv run ./model/start_train.py