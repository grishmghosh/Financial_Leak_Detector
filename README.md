# Financial_Leak_Detector
Financial Leak Detector is an end-to-end expense intelligence system that applies anomaly detection on large-scale vendor payment data to uncover duplicate payments, vendor irregularities, cost creep, and abnormal spending patterns. Built using Python, ML models, Power BI, Docker, and workflow automation.

## Automatic Model Retraining

The anomaly detection model can be retrained automatically using the provided script:

```bash
./scripts/retrain_model.sh
```

To schedule daily retraining at 3:00 AM via cron:

```
0 3 * * * /path/to/project/scripts/retrain_model.sh
```

This retrains the Isolation Forest anomaly detection model using the latest transaction data from the database.
