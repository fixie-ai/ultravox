cd $HOME
mkdir workspace
cd workspace
git clone https://github.com/fixie-ai/ultravox-omni.git -b main
cd ultravox-omni
mkdir -p ~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
just install
bash ./scripts/vscode_tunnel.sh
