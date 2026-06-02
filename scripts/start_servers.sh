#!/bin/bash
# Start MCP proxy (port 7111) + typesense-server for training rollouts.
# Uses the PRIVATE servers.yaml (AI.ThinkingBox.Data) because both
# train_list_origin and train_list_in_scenario depend on private tools.
#
# Usage:
#   source .venv/bin/activate
#   ./scripts/start_servers.sh
# Stop with Ctrl+C (delegates to background_tasks.sh which handles cleanup).

set -euo pipefail

export THINKINGBOX_DATA="${THINKINGBOX_DATA:-/home/azureuser/zhuochun/AI.ThinkingBox.Data}"
export TB_MCP_START_SERVERS_FILE="$THINKINGBOX_DATA/servers/servers.yaml"
export TYPESENSE_API_KEY="${TYPESENSE_API_KEY:-Fake}"

# typesense-server binary lives in the sibling thinkingbox venv; add to PATH.
export PATH="/home/azureuser/zhuochun/thinkingbox/.venv/bin:$PATH"

# Sanity checks
if [[ ! -r "$TB_MCP_START_SERVERS_FILE" ]]; then
  echo "ERROR: TB_MCP_START_SERVERS_FILE not readable: $TB_MCP_START_SERVERS_FILE" >&2
  exit 1
fi
if ! command -v tb >/dev/null 2>&1; then
  echo "ERROR: 'tb' not on PATH. Did you 'source .venv/bin/activate'?" >&2
  exit 1
fi
if ! command -v typesense-server >/dev/null 2>&1; then
  echo "ERROR: 'typesense-server' not on PATH (expected at /home/azureuser/zhuochun/thinkingbox/.venv/bin)" >&2
  exit 1
fi

echo "MCP servers file : $TB_MCP_START_SERVERS_FILE"
echo "THINKINGBOX_DATA : $THINKINGBOX_DATA"
echo "Typesense API key: $TYPESENSE_API_KEY"
echo

exec /home/azureuser/zhuochun/thinkingbox/scripts/background_tasks.sh
