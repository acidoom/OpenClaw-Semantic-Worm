#!/usr/bin/env bash
# farm-start.sh — Launch all farm services in a tmux session.
#
# Usage: ./farm-start.sh [--no-vllm]
#
# Creates a tmux session "farm" with panes:
#   [0] vLLM server (unless --no-vllm)
#   [1] Farm Controller :9000
#   [2] MiniMolt Feed :8080
#   [3] TSM Logger
#   [4] Monitoring (nvidia-smi + htop)

set -euo pipefail

# Ensure openclaw and venv Python are in PATH
export PATH="$HOME/.npm-global/bin:$HOME/semantic-worm/.venv/bin:$PATH"

FARM_DIR="${FARM_BASE_DIR:-$HOME/semantic-worm}"
PYTHON="$FARM_DIR/.venv/bin/python"
DAEMONS_DIR="$FARM_DIR/daemons"
SESSION="farm"
VLLM_MODEL="qwen2.5-32b"
VLLM_PORT=8000
NO_VLLM=false

# Parse args
for arg in "$@"; do
    case $arg in
        --no-vllm) NO_VLLM=true ;;
    esac
done

# Kill existing session if any
set +e
tmux kill-session -t "$SESSION" 2>/dev/null
set -e
sleep 1

# Create new session
tmux new-session -d -s "$SESSION" -n "farm"

# Pane 0: vLLM
if [ "$NO_VLLM" = false ]; then
    tmux send-keys -t "$SESSION:0" \
        "cd $FARM_DIR && $PYTHON -m vllm.entrypoints.openai.api_server \
        --model $FARM_DIR/models/$VLLM_MODEL \
        --host 0.0.0.0 --port $VLLM_PORT \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --tensor-parallel-size 1" C-m
else
    tmux send-keys -t "$SESSION:0" "echo 'vLLM skipped (--no-vllm)'; sleep infinity" C-m
fi

# Pane 1: MiniMolt Feed
tmux split-window -t "$SESSION:0" -h
tmux send-keys -t "$SESSION:0.1" \
    "cd $DAEMONS_DIR && $PYTHON minimolt.py --port 8080 --db $FARM_DIR/data/feed.db" C-m

# Pane 2: Controller
tmux split-window -t "$SESSION:0.0" -v
tmux send-keys -t "$SESSION:0.2" \
    "sleep 3 && cd $DAEMONS_DIR && export PATH=$HOME/.npm-global/bin:\$PATH && $PYTHON controller.py --port 9000" C-m

# Pane 3: TSM Logger
tmux split-window -t "$SESSION:0.1" -v
tmux send-keys -t "$SESSION:0.3" \
    "sleep 5 && cd $DAEMONS_DIR && $PYTHON tsm_logger.py" C-m

# Pane 4: Monitoring
tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:1" \
    "watch -n 2 'echo \"=== GPU ===\"; nvidia-smi; echo; echo \"=== Farm ===\"; curl -s http://localhost:9000/health 2>/dev/null || echo \"controller not ready\"'" C-m

# Select first pane
tmux select-window -t "$SESSION:0"
tmux select-pane -t 0

echo "Farm started in tmux session '$SESSION'"
echo "  Attach: tmux attach -t $SESSION"
echo ""
echo "Services:"
echo "  vLLM:       http://localhost:$VLLM_PORT  (pane 0)"
echo "  MiniMolt:   http://localhost:8080         (pane 1)"
echo "  Controller: http://localhost:9000         (pane 2)"
echo "  TSM Logger:                               (pane 3)"
echo "  Monitor:                                  (window 1)"
