"""Default configuration for farmlib."""

import os
from pathlib import Path

# Service URLs
CONTROLLER_URL = "http://localhost:9000"
FEED_URL = "http://localhost:8080"

# Paths — override with FARM_BASE_DIR env var
BASE_DIR = Path(os.environ.get("FARM_BASE_DIR", Path.home() / "semantic-worm"))
AGENTS_DIR = BASE_DIR / "agents"
RUNS_DIR = BASE_DIR / "runs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODELS_DIR = BASE_DIR / "models"

# Defaults
DEFAULT_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"
DEFAULT_FLEET_SIZE = 30
DEFAULT_CYCLES = 100
DEFAULT_RATE_LIMIT = 2.0
