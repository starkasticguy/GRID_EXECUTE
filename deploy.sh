#!/usr/bin/env bash
# deploy.sh — Push code to GitHub and deploy to production server
# Usage:  ./deploy.sh [--no-tests] [--restart-only]
#
# What it does:
#   1. Optionally runs tests locally (abort if any fail)
#   2. Commits any uncommitted changes with an auto message
#   3. Pushes to GitHub
#   4. SSHes to the server and runs the remote deploy steps

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────
SERVER="root@64.227.132.182"
REPO_DIR="~/GRID_EXECUTE"
SERVICE="gridbot"
BRANCH="main"
COIN="ETH"                    # coin the service is trading
LEVERAGE="3"                  # must match config.py
CAPITAL=""                    # leave empty = use full wallet balance

# ─── Flags ───────────────────────────────────────────────────────
RUN_TESTS=true
RESTART_ONLY=false

for arg in "$@"; do
  case $arg in
    --no-tests)     RUN_TESTS=false ;;
    --restart-only) RESTART_ONLY=true ;;
  esac
done

# ─── Colours ─────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()    { echo -e "${YELLOW}[deploy]${NC} $*"; }
error()   { echo -e "${RED}[deploy]${NC} $*"; exit 1; }

# ─── Step 1: Local tests ─────────────────────────────────────────
if [ "$RUN_TESTS" = true ] && [ "$RESTART_ONLY" = false ]; then
  info "Running test suite..."
  if ! python3 -m pytest tests/ -q --tb=short 2>&1; then
    error "Tests failed — aborting deploy. Use --no-tests to skip."
  fi
  info "Tests passed."
fi

# ─── Step 2: Commit + push ───────────────────────────────────────
if [ "$RESTART_ONLY" = false ]; then
  cd "$(git rev-parse --show-toplevel)"

  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Uncommitted changes detected — committing automatically."
    git add -A
    git commit -m "deploy: $(date -u '+%Y-%m-%d %H:%M UTC')"
  fi

  info "Pushing to GitHub ($BRANCH)..."
  git push origin "$BRANCH"
  info "Push complete."
fi

# ─── Step 3: Remote deploy ───────────────────────────────────────
CAPITAL_ARG=""
if [ -n "$CAPITAL" ]; then
  CAPITAL_ARG="--capital $CAPITAL"
fi

info "Deploying to $SERVER..."

ssh "$SERVER" bash << REMOTE
set -euo pipefail

echo "[server] Stopping service..."
systemctl stop $SERVICE || true

echo "[server] Pulling latest code..."
cd $REPO_DIR
git fetch origin
git reset --hard origin/$BRANCH

echo "[server] Installing dependencies into venv..."
python3 -m venv venv
venv/bin/pip install -q -r requirements.txt

echo "[server] Snapshotting live state (backup)..."
SNAP_DIR="data/live_state/snapshots_deploy"
mkdir -p "\$SNAP_DIR"
cp -r data/live_state/*.json "\$SNAP_DIR/" 2>/dev/null && \\
  echo "[server] State backed up to \$SNAP_DIR" || \\
  echo "[server] No state files to back up."

echo "[server] Generating systemd service file dynamically..."
cat << EOF_SVC > /etc/systemd/system/gridbot.service
[Unit]
Description=GRID Trading Bot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=root
WorkingDirectory=/root/GRID_EXECUTE
EnvironmentFile=/root/GRID_EXECUTE/.env
ExecStart=/root/GRID_EXECUTE/venv/bin/python3 live_trade.py --coin ${COIN} --leverage ${LEVERAGE} --resume ${CAPITAL_ARG}
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gridbot

[Install]
WantedBy=multi-user.target
EOF_SVC


systemctl daemon-reload

echo "[server] Starting service..."
systemctl start $SERVICE

sleep 3
if systemctl is-active --quiet $SERVICE; then
  echo "[server] Service is running."
  systemctl status $SERVICE --no-pager -l | tail -20
else
  echo "[server] ERROR: Service failed to start."
  journalctl -u $SERVICE --no-pager -n 30
  exit 1
fi
REMOTE

info "Deploy complete. Tailing logs for 15s..."
ssh "$SERVER" "journalctl -u $SERVICE -f --no-pager" &
LOG_PID=$!
sleep 15
kill $LOG_PID 2>/dev/null
echo ""
info "Done. Run 'ssh $SERVER journalctl -u $SERVICE -f' to keep watching."
