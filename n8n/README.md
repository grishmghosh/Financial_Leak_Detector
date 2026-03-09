# n8n Automation — LeakWatch ML Pipeline

## Overview

This directory contains the n8n workflow that automates the LeakWatch financial anomaly detection pipeline. The workflow triggers ML analysis on a schedule, polls for completion, evaluates results, and sends alerts when high-risk transactions are detected.

## Workflow Diagram

```
┌──────────────────┐
│  Schedule Trigger │   (Daily 06:00 UTC)
│  (Cron)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Run Analysis    │   POST /api/v1/ml/run-analysis
│  (HTTP POST)     │   → returns { run_id, status: "running" }
└────────┬─────────┘
         │
         ▼
┌──────────────────┐ ◄──────────────────────────────────────┐
│  Wait Before     │                                         │
│  Poll (30s)      │   Pause between each status check       │
└────────┬─────────┘                                         │
         │                                                   │
         ▼                                                   │
┌──────────────────┐                                         │
│  Check Run       │   GET /api/v1/ml/run-status/{run_id}    │
│  Status (HTTP)   │                                         │
└────────┬─────────┘                                         │
         │                                                   │
         ▼                                                   │
┌──────────────────┐                                         │
│  Track Poll      │   Merges status data + increments       │
│  Count (Code)    │   iteration counter (poll_count)        │
└────────┬─────────┘                                         │
         │                                                   │
         ▼                                                   │
┌──────────────────┐                                         │
│  Is Analysis     │                                         │
│  Complete? (IF)  │                                         │
└──┬───────────┬───┘                                         │
   │ YES       │ NO                                          │
   │           ▼                                             │
   │  ┌──────────────────┐                                   │
   │  │  Max Retries     │                                   │
   │  │  Reached? (IF)   │   poll_count >= 20?               │
   │  └──┬───────────┬───┘                                   │
   │     │ YES       │ NO ───────────────────────────────────┘
   │     ▼              (loop back)
   │  ┌──────────────────┐
   │  │  Send Timeout    │   Webhook: "analysis timed out"
   │  │  Alert           │
   │  └──────────────────┘
   │
   ▼
┌──────────────────┐
│  Has Anomalies?  │   high_risk_count > 0?
│  (IF)            │
└──┬───────────┬───┘
   │ YES       │ NO
   │           ▼
   │  ┌──────────────────┐
   │  │  No Anomalies    │   Workflow ends cleanly
   │  │  Detected (NoOp) │
   │  └──────────────────┘
   │
   ▼
┌──────────────────┐
│  Fetch High Risk │   GET /api/v1/dashboard/high-risk-transactions
│  Transactions    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Format Alert    │   Builds human-readable alert with
│  Message (Code)  │   top 10 high-risk transactions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Send Alert      │   POST to webhook (Slack/Teams/custom)
│  Notification    │
└──────────────────┘
```

## Node Reference

| # | Node | Type | Purpose |
|---|------|------|---------|
| 1 | Schedule Trigger | Cron | Fires daily at 06:00 UTC |
| 2 | Run Analysis | HTTP POST | Triggers ML scoring pipeline |
| 3 | Wait Before Poll | Wait (30s) | Pauses between status checks |
| 4 | Check Run Status | HTTP GET | Polls `/ml/run-status/{run_id}` |
| 5 | Track Poll Count | Code | Increments counter, merges data |
| 6 | Is Analysis Complete? | IF | `status == "completed"` |
| 7 | Max Retries Reached? | IF | `poll_count >= 20` (10 min max) |
| 8 | Has Anomalies? | IF | `high_risk_count > 0` |
| 9 | Fetch High Risk Transactions | HTTP GET | Gets detailed transaction data |
| 10 | Format Alert Message | Code | Builds alert with top 10 risky txns |
| 11 | Send Alert Notification | HTTP POST | Sends to webhook URL |
| 12 | No Anomalies Detected | NoOp | Clean exit path |
| 13 | Send Timeout Alert | HTTP POST | Notifies ops of timeout |

## Alert Logic

An alert is triggered when **all** of the following are true:

1. The ML analysis run completes successfully (`status == "completed"`)
2. At least one high-risk transaction is detected (`high_risk_count > 0`)

The alert message includes:
- Run ID and completion timestamp
- Count of transactions scored, high-risk transactions, and duplicates
- Top 10 high-risk transactions with voucher numbers, leak probabilities, risk factors, and duplicate flags

## Loop Safety

The polling loop has a hard ceiling of **20 iterations × 30 seconds = 10 minutes**. If the analysis has not completed by then, the workflow exits the loop and sends a timeout alert instead of running indefinitely.

---

## Setup Instructions

### Prerequisites

- n8n instance running (Docker, npm, or cloud)
- LeakWatch FastAPI backend running and accessible from n8n
- A Supabase JWT for an **admin** user

### 1. Configure Environment Variables

Set these environment variables in your n8n instance:

| Variable | Description | Example |
|----------|-------------|---------|
| `LEAKWATCH_JWT_TOKEN` | Supabase JWT for an admin-role user | `eyJhbGciOi...` |
| `ALERT_WEBHOOK_URL` | Webhook endpoint for alerts | `https://hooks.slack.com/services/T.../B.../xxx` |

**Docker (docker-compose.yml):**
```yaml
services:
  n8n:
    image: n8nio/n8n
    environment:
      - LEAKWATCH_JWT_TOKEN=your_admin_jwt_here
      - ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**n8n Cloud:** Settings → Environment Variables

### 2. Import the Workflow

1. Open your n8n instance (default: `http://localhost:5678`)
2. Click **"..."** (menu) → **Import from File**
3. Select `n8n/workflows/leakwatch_ml_automation.json`
4. The workflow will appear in your workflow list

### 3. Verify Connectivity

Before activating, run a manual test:

1. Open the imported workflow
2. Click **"Test Workflow"** (play button)
3. Check that the **Run Analysis** node returns `{ run_id, status: "running" }`
4. If you get a `401` error → check `LEAKWATCH_JWT_TOKEN`
5. If you get a connection error → verify the backend is reachable at `http://host.docker.internal:8000`

### 4. Activate

Toggle the **Active** switch in the top-right corner of the workflow editor.

---

## Docker Networking

The workflow uses `http://host.docker.internal:8000` to reach the FastAPI backend. This is the Docker-standard way to access the host machine from within a container.

| Scenario | Base URL |
|----------|----------|
| n8n in Docker, backend on host | `http://host.docker.internal:8000` |
| Both in same Docker network | `http://leakwatch-api:8000` (use service name) |
| n8n on host (npm install) | `http://localhost:8000` |

If using a shared Docker network, update the URLs in nodes 2, 4, and 9.

## Customization

### Change Schedule
Edit the **Schedule Trigger** node's cron expression:
- Hourly: `0 * * * *`
- Every 6 hours: `0 */6 * * *`
- Weekdays 9 AM: `0 9 * * 1-5`

### Replace Notification Channel
Swap the **Send Alert Notification** node for:
- **Slack** node (native n8n integration)
- **Microsoft Teams** node
- **Email Send** node (requires SMTP credentials)
- **Telegram** node

### Filter Analysis Scope
Modify the **Run Analysis** node's JSON body:
```json
{
  "start_date": "2026-01-01",
  "end_date": "2026-03-07",
  "department": "procurement"
}
```

### Adjust Polling
- **Wait duration:** Edit the `amount` parameter in the **Wait Before Poll** node
- **Max retries:** Edit the `rightValue` in the **Max Retries Reached?** node (default: 20)
