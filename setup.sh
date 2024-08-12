cd $HOME
mkdir workspace
cd workspace
git clone https://github.com/fixie-ai/ultravox.git -b main
git clone https://github.com/fixie-ai/evals.git -b main
cd ultravox
mkdir -p ~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
just install
poetry run pip install -e ../evals
bash ./scripts/vscode_tunnel.sh
