#!/bin/bash

# Copied from https://docs.mosaicml.com/projects/mcli/en/latest/training/interactive.html
# This script is used to create a tunnel to the MosaicML training environment
# It allows you to use the VSCode interface to interact with the training environment
# Very useful for debugging and monitoring training runs

trap '/tmp/code tunnel unregister' EXIT
cd /tmp && curl -A "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
/tmp/code tunnel --accept-server-license-terms --no-sleep --name mml-dev-01