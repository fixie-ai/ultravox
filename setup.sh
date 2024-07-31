cd $HOME
mkdir workspace
cd workspace
git clone git@github.com:fixie-ai/ultravox.git -b main

cd ultravox
mkdir -p ~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
just install
bash ./scripts/vscode_tunnel.sh
