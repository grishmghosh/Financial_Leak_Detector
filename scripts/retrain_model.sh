#!/bin/bash
# LeakWatch Model Retraining Script
# Schedule via cron: 0 3 * * * /path/to/project/scripts/retrain_model.sh

echo "Starting LeakWatch model retraining..."

python -m app.ml.train_model

echo "Model retraining completed."
