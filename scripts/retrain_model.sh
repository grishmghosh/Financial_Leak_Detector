#!/bin/bash
# LeakWatch Model Retraining Script
# Schedule via cron: 0 3 * * * /path/to/project/scripts/retrain_model.sh

set -e

echo "[$(date)] Starting LeakWatch model retraining..."

python -m app.ml.train_model

echo "[$(date)] Model retraining completed."
