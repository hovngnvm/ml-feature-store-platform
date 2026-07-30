# Real-Time ML Feature Store & Fraud Detection Platform

## Project Overview

An enterprise-grade **Real-Time Machine Learning Feature Store & Low-Latency Model Serving Platform** built for high-throughput credit card fraud detection. The platform unifies **Dual-Path Feature Processing** (PyFlink streaming window aggregations & DuckDB batch historical transforms), manages feature lifecycle via **Feast** (Redis online store for sub-1ms low-latency retrieval & MinIO S3 offline store for point-in-time joins), enforces schema data quality through **Pandera**, monitors feature drift via **Evidently AI**, trains an **XGBoost + LightGBM Model Ensemble** with SHAP explainability, serves online inference via **FastAPI**, orchestrates automated continuous training (CT) through **Prefect 3**, and provides operational observability using **Prometheus**, **Grafana**, and **Streamlit**.

**Business Goal:** Detect anomalous and fraudulent financial transactions in real-time under a strict **< 50ms P95 API Latency SLA** (with sub-1ms Redis feature lookup), preventing financial loss while minimizing false positives through dynamic on-demand feature transformations and automated drift-triggered continuous retraining.

---

## Architecture & Tech Stack

```mermaid
flowchart TD
    subgraph Sources [Data Sources & Streaming Broker]
        producer["Transaction Producer<br/>(Redpanda / Kafka Broker)"]:::source
        raw_parquet["Historical Raw Logs<br/>(Batch Parquet Files)"]:::source
    end

    subgraph DualPath [Dual-Path Feature Engine]
        flink["PyFlink Stream Engine<br/>(1m / 5m / 1h / 24h Windows)"]:::flink
        duckdb["DuckDB Batch Engine<br/>(7d / 30d Historical Aggs)"]:::duckdb
    end

    subgraph FeatureStore [Feast Feature Store]
        redis["Online Store: Redis 7.2<br/>(< 1ms Low-Latency Feature Lookup)"]:::redis
        minio["Offline Store: MinIO S3<br/>(Hive-Partitioned Lakehouse Parquet)"]:::minio
        feast_registry["Feast Registry & On-Demand Engine<br/>(Ratio, Z-Score & Velocity UDFs)"]:::feast
    end

    subgraph Quality [Data Quality & Drift Governance]
        pandera["Pandera DQ Gate<br/>(Schema Validation Assertions)"]:::quality
        evidently["Evidently AI<br/>(Data Drift & Performance Reports)"]:::quality
        dlq["Dead Letter Queue (DLQ)<br/>(Quarantine Store Parquet)"]:::quality
    end

    subgraph ML [Machine Learning & Continuous Training]
        ensemble["Model Ensemble<br/>(XGBoost + LightGBM Blending)"]:::ml
        shap["SHAP Explainability<br/>(Feature Importance & Summary Plots)"]:::ml
        prefect["Prefect 3 Orchestrator<br/>(Hybrid CT & Materialization Flow)"]:::orchestrator
    end

    subgraph Serving [Serving & Analytics Layer]
        fastapi["FastAPI Model Serving REST API<br/>(/predict Endpoint < 50ms P95 SLA)"]:::serving
        streamlit["Streamlit Analytics Dashboard<br/>(Real-Time Monitoring UI)"]:::viz
        grafana["Grafana & Prometheus<br/>(System & Redis Metrics)"]:::viz
    end

    %% Flow lines
    producer -->|Stream Ingest| flink
    raw_parquet -->|Batch Processing| duckdb

    flink -->|Push Stream Features| redis
    flink -.->|Quarantine Errors| dlq

    duckdb -->|Pandera DQ Assertions| pandera
    pandera -->|Write Lakehouse Partitions| minio

    minio -->|Feast Materialize Sync| redis
    redis -->|Online Feature Lookup| feast_registry
    feast_registry -->|Serve Feature Vector| ensemble

    ensemble -->|Model Artifacts| fastapi
    ensemble -->|Generate SHAP Plots| shap

    prefect -->|Orchestrate 6-Step Pipeline| duckdb
    prefect -->|Trigger Retrain on Drift| evidently
    prefect -->|Materialize Batch Features| redis

    fastapi -->|Inference Scores & Metrics| streamlit
    redis -.->|Export Telemetry| grafana

    %% Style Classes
    classDef source fill:#E5E7EB,stroke:#9CA3AF,color:#1F2937,stroke-width:2px;
    classDef flink fill:#FFEDD5,stroke:#FB923C,color:#7C2D12,stroke-width:2px;
    classDef duckdb fill:#FFFBEB,stroke:#F59E0B,color:#92400E,stroke-width:2px;
    classDef redis fill:#FEE2E2,stroke:#EF4444,color:#7F1D1D,stroke-width:2px;
    classDef minio fill:#E0F2FE,stroke:#0284C7,color:#075985,stroke-width:2px;
    classDef feast fill:#F3E8FF,stroke:#A855F7,color:#581C87,stroke-width:2px;
    classDef quality fill:#FEF3C7,stroke:#F59E0B,color:#78350F,stroke-width:2px;
    classDef ml fill:#D1FAE5,stroke:#10B981,color:#065F46,stroke-width:2px;
    classDef orchestrator fill:#E0E7FF,stroke:#6366F1,color:#3730A3,stroke-width:2px;
    classDef serving fill:#CCFBF1,stroke:#14B8A6,color:#115E59,stroke-width:2px;
    classDef viz fill:#FAE8FF,stroke:#D946EF,color:#701A75,stroke-width:2px;
```

* **Workflow Orchestration:** ![Prefect](https://img.shields.io/badge/Prefect-3.8.1-070219?style=flat&logo=prefect&logoColor=white)
* **Stream Processing:** ![Apache Flink](https://img.shields.io/badge/Apache%20Flink-1.18.1-E6522C?style=flat&logo=apacheflink&logoColor=white) (`PyFlink`) + ![Redpanda](https://img.shields.io/badge/Redpanda-Kafka-EC1C24?style=flat&logo=redpanda&logoColor=white)
* **Batch Processing:** ![DuckDB](https://img.shields.io/badge/DuckDB-0.10.1-FFF000?style=flat&logo=duckdb&logoColor=black)
* **Feature Store Engine:** ![Feast](https://img.shields.io/badge/Feast-0.38.0-3776AB?style=flat&logo=feast&logoColor=white)
* **Storage Layer:** ![Redis](https://img.shields.io/badge/Redis-7.2-DC382D?style=flat&logo=redis&logoColor=white) (Online Store) + ![MinIO](https://img.shields.io/badge/MinIO-S3-C72C48?style=flat&logo=minio&logoColor=white) (Offline Lakehouse)
* **Data Quality & Governance:** ![Pandera](https://img.shields.io/badge/Pandera-0.32.1-10B981?style=flat) + ![Evidently AI](https://img.shields.io/badge/Evidently%20AI-0.4.0-6366F1?style=flat)
* **Machine Learning & Serving:** ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-F7931E?style=flat&logo=scikitlearn&logoColor=white) + ![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=flat&logo=fastapi&logoColor=white)
* **Dashboard & Observability:** ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white) + ![Grafana](https://img.shields.io/badge/Grafana-10.3.0-F46800?style=flat&logo=grafana&logoColor=white) + ![Prometheus](https://img.shields.io/badge/Prometheus-2.50.0-E6522C?style=flat&logo=prometheus&logoColor=white)
* **Containerization:** ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## Feature Store Schema & Feature Matrix

The platform models feature definitions using Feast feature views across streaming, batch, and on-demand transformation layers:

| Feature View | Source Engine / Store | Feature Name | Data Type | Transformation Logic & Description | Target Consumer |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `card_stream_features` | **PyFlink** $\rightarrow$ Redis Online Store | `trans_count_1m`<br>`trans_count_5m`<br>`trans_count_1h`<br>`avg_amount_24h`<br>`std_amount_24h` | `Int64`<br>`Int64`<br>`Int64`<br>`Float32`<br>`Float32` | Real-time sliding window aggregations over Redpanda stream to track burst velocities and recent spending baseline | FastAPI Online Model Serving (`/predict`) |
| `card_batch_features` | **DuckDB** $\rightarrow$ MinIO Offline Lakehouse | `trans_count_7d`<br>`trans_count_30d`<br>`avg_amount_30d`<br>`max_amount_30d` | `Int64`<br>`Int64`<br>`Float32`<br>`Float32` | Historical long-term customer behavioral profiles computed over raw transaction lakehouse partitions | Offline Point-in-Time Joins & Redis Materialization |
| `on_demand_feature_view` | **Feast UDF** (In-Memory Request Time) | `amount_ratio_24h`<br>`amount_ratio_30d`<br>`amount_zscore_24h`<br>`is_amount_gt_30d_max`<br>`is_high_velocity_5m` | `Float32`<br>`Float32`<br>`Float32`<br>`Bool`<br>`Bool` | Computed dynamically during inference:<br>• `amount / (avg_amount_24h + 1e-5)`<br>• `amount / (avg_amount_30d + 1e-5)`<br>• `(amount - avg_24h) / (std_24h + 1e-5)`<br>• `amount > max_amount_30d`<br>• `trans_count_5m >= 3` | Blended Model Ensemble Classifier |

---

## Data Quality, Governance & Drift Architecture

```mermaid
flowchart LR
    subgraph StreamGov [Stream Governance]
        stream_in["Raw Kafka Stream"] --> flink_val{"PyFlink Validator"}
        flink_val -->|Valid Payload| redis_sync["Redis Online Push"]
        flink_val -->|Corrupted / Negative| dlq["DLQ Parquet Quarantine"]
    end

    subgraph BatchGov [Batch Quality Gate]
        duckdb_out["DuckDB Aggregates"] --> pandera_gate{"Pandera Schema Gate"}
        pandera_gate -->|Pass 100% Constraints| minio_store["MinIO Lakehouse S3"]
        pandera_gate -->|Constraint Violation| halt["Halt Materialization"]
    end

    subgraph DriftRetrain [Drift-Driven CT]
        minio_store --> evidently_drift["Evidently AI Drift Test"]
        evidently_drift -->|K-S Score > 0.10| retrain["Prefect CT Flow Trigger"]
        evidently_drift -->|No Drift Flagged| keep["Preserve Model Artifact"]
    end
```

### 1. Pandera Schema Quality Gate (`src/quality/data_assert.py`)
* Enforces non-null primary entity keys (`card_id`), positive bounds on transaction amounts (`amount > 0`), non-negative velocity counts, and finite standard deviations.
* Blocks invalid feature partitions from persisting to MinIO S3 Lakehouse.

### 2. Dead Letter Queue Quarantine (`data/lakehouse/dlq/`)
* Streaming records with invalid JSON formatting, negative amounts, or missing entity keys are side-outputted to `stream_errors.parquet` without blocking the PyFlink stream ingestion pipeline.

### 3. Drift-Driven Continuous Training (`src/quality/feature_monitoring.py`)
* Computes two-sample Kolmogorov-Smirnov (K-S) distribution divergence tests comparing production feature values against the baseline training distribution.
* Generates an interactive audit report (`dashboards/feature_drift_report.html`) and triggers automated retraining when drift exceeds tolerance thresholds.

---

## Inference REST API Data Contract (`/predict`)

The serving layer exposes a low-latency endpoint with predictable JSON payload contracts:

### Request Contract
```json
{
  "card_id": "11556",
  "current_amount": 750.0
}
```

### Response Contract (< 50ms P95 SLA)
```json
{
  "card_id": "11556",
  "is_fraud": true,
  "fraud_score": 0.942,
  "decision": "BLOCKED",
  "latency_ms": 3.42,
  "top_features": {
    "amount_zscore_24h": 3.85,
    "is_high_velocity_5m": true,
    "amount_ratio_30d": 6.25
  }
}
```

---

## Pipeline Workflow

### 1. Dual-Path Feature Ingestion Engine
* **Real-Time Stream Processing (`PyFlink` / `DirectStreamProcessor`):** Consumes live transaction streams from Redpanda/Kafka (`raw_transactions`). Calculates sliding window metrics (1m, 5m, 1h velocity counts, 24h average/stddev transaction amounts) and pushes stream features directly into the Redis online feature store (`card_stream_features`).
* **DLQ Quarantine Side Output:** Malformed JSON records, negative transaction amounts, or missing entity keys are automatically routed to a Dead Letter Queue (DLQ) side output stream and quarantined into partitioned Parquet Lakehouse (`data/lakehouse/dlq/`).
* **Historical Batch Lakehouse (`DuckDB`):** Executes high-performance analytical queries over historical transaction logs, computing 7-day and 30-day windowed metrics (`trans_count_7d`, `trans_count_30d`, `avg_amount_30d`, `max_amount_30d`). Exports partitioned Parquet files to the MinIO S3 Lakehouse (`feature-store-offline`).

### 2. Feast Feature Store & On-Demand Derivations
* **Offline Point-in-Time Joins:** Feast joins historical batch features with target labels without data leakage, producing clean training datasets.
* **Online Materialization:** Synchronizes batch features from MinIO S3 Parquet into Redis online store using `store.materialize()`.
* **On-Demand Feature Calculation:** Derives dynamic real-time features at request time (`amount_ratio_24h`, `amount_ratio_30d`, `amount_zscore_24h`, `is_amount_gt_30d_max`, `is_high_velocity_5m`) via Feast `@on_demand_feature_view` UDFs.

### 3. Data Quality Assertions & Drift Monitoring
* **Pandera Data Quality Gate:** Enforces structural integrity constraints before committing feature partitions to the Lakehouse.
* **Evidently AI Feature Drift Monitoring:** Compares distribution stats of current inference features against baseline reference data and generates an interactive HTML report (`dashboards/feature_drift_report.html`).

### 4. Continuous Training (CT) & Model Ensemble
* **Model Ensemble Blending:** Trains and evaluates an ensemble combining **XGBoost** and **LightGBM** classifiers. Blends prediction probabilities to maximize Precision-Recall AUC (PR-AUC) and ROC-AUC.
* **SHAP Explainability:** Calculates SHAP values to output global feature importance rankings and beeswarm distribution plots (`dashboards/shap_summary_beeswarm.png`).
* **Event-Driven Retraining:** Automatically triggers model retraining if Evidently AI flags significant dataset drift or upon scheduled cron execution.

### 5. Low-Latency REST API Serving (< 50ms P95 SLA)
* **FastAPI Server (`src/api/main.py`):** Exposes `/predict` and `/health` endpoints. On transaction request, fetches pre-calculated online features from Redis (< 1ms), evaluates on-demand derived features, runs model ensemble inference, and returns decision metrics under **50 milliseconds (P95)**.

### 6. Orchestration & Observability
* **Prefect 3 Flow (`execute.py`):** Coordinates end-to-end steps: Batch execution → Stream DLQ audit → Feast materialization → End-to-end verification → Evidently drift analysis → Hybrid CT retraining decision.
* **Dashboards:**
  * **Streamlit App (`dashboards/app.py`):** Visualizes live transactions, fraud alerts, model performance metrics, SHAP charts, and drift status.
  * **Grafana & Prometheus:** Monitors Redis memory footprint, request rate, exporter metrics, and system CPU/RAM usage.

---

## Key Engineering Highlights

* **Dual-Path Feature Store Architecture:** Seamlessly unifies low-latency real-time streaming features (PyFlink + PushSource) and scalable historical batch features (DuckDB + FileSource).
* **Sub-50ms End-to-End Inference SLA:** Direct Redis online feature caching (< 1ms) and warm-loaded model artifacts enable ultra-low-latency REST API prediction responses.
* **Strict DLQ Quarantine Policy:** Non-blocking streaming pipeline with dedicated Dead Letter Queue (DLQ) side outputs to isolate corrupt payloads without halting ingestion.
* **Pandera DQ Assertions & Governance:** Integrated data validation layer preventing schema drift or invalid values from corrupting downstream models.
* **Automated Hybrid Continuous Training (CT):** Prefect 3 flow dynamically evaluates Evidently AI drift scores to automate retraining decisions, preventing model degradation.
* **Model Explainability & Transparency:** Integrates SHAP summary plots and feature contribution weights directly into prediction telemetry and analytics dashboards.

---

## Project Structure

```
ml-feature-store-platform/
│
├── feature_repository/                # Feast Feature Store Definitions
│   ├── entities.py                    # Card entity schema definition (card_id primary key)
│   ├── feature_store.yaml             # Feast provider, registry & online/offline store config
│   └── features.py                    # Batch, Push/Stream & On-Demand Feature Views definitions
│
├── src/                               # System Core Source Code
│   ├── api/                           # Low-Latency Model Serving REST API
│   │   └── main.py                    # FastAPI server (< 50ms P95 SLA) fetching online features & scoring fraud
│   │
│   ├── config/                        # System Configurations
│   │   ├── __init__.py
│   │   └── settings.py                # Pydantic Settings loading environment variables & system paths
│   │
│   ├── ml/                            # Machine Learning Engine & Model Training
│   │   ├── ensemble.py                # Blended Model Ensemble wrapper (XGBoost + LightGBM)
│   │   ├── evaluate.py                # Model evaluation (PR-AUC, ROC-AUC, Confusion Matrix)
│   │   ├── explain.py                 # SHAP explainability analysis & summary plot generation
│   │   ├── prepare_dataset.py         # Offline feature joiner & train/test split dataset builder
│   │   └── train.py                   # Model ensemble training pipeline execution script
│   │
│   ├── offline/                       # Batch Feature Engine
│   │   ├── batch_feature_job.py       # DuckDB analytical job: computes 7d/30d historical aggregates
│   │   └── feast_retrieval_demo.py    # Feast point-in-time historical feature retrieval validator demo
│   │
│   ├── orchestration/                 # Prefect Workflow Engine
│   │   └── execute.py                 # 6-Step Prefect 3 Flow (Lakehouse, Stream, Materialize, Audit, Drift, CT)
│   │
│   ├── producer/                      # Event Simulation Stream Producer
│   │   └── producer.py                # High-frequency Redpanda/Kafka producer (normal transactions & fraud bursts)
│   │
│   ├── quality/                       # Data Quality & Drift Detection Governance
│   │   ├── data_assert.py             # Pandera DataFrame Schema assertions & quality constraints
│   │   └── feature_monitoring.py      # Evidently AI feature drift analysis & HTML report generator
│   │
│   ├── streaming/                     # Real-Time Processing Engine
│   │   └── flink_feature_job.py       # PyFlink job: tumbling/sliding window aggregations & DLQ side output
│   │
│   └── utils/                         # Infrastructure Client Utilities
│       ├── logger.py                  # Standardized ISO-8601 logging utility
│       ├── minio_client.py            # MinIO S3 object storage client & bucket helper
│       └── redis_client.py            # Redis client wrapper with connection pooling & healthchecks
│
├── dashboards/                        # UI & Monitoring Configurations
│   ├── app.py                         # Interactive Streamlit BI & Fraud Analytics Dashboard
│   └── prometheus.yml                 # Prometheus scrapers configuration (Redis exporter & system metrics)
│
├── data/                              # Local Storage & Lakehouse Partitions
│   ├── batch_features.parquet         # Historical batch feature store output
│   └── lakehouse/                     # Partitioned Parquet Lakehouse & DLQ quarantine files
│
├── models/                            # Serialized Machine Learning Artifacts
│   └── ensemble_fraud_model.joblib    # Trained model ensemble binary (XGBoost + LightGBM)
│
├── tests/                             # Comprehensive Modular Test Suite (pytest)
│   ├── test_api_serving.py            # FastAPI REST API serving & inference tests
│   ├── test_config_logger.py          # Settings, ISO Logger & Ensemble verification tests
│   ├── test_ml_training.py            # Cost Matrix & Decision Threshold tests
│   ├── test_quality_gate.py           # Pandera Schema Gate & Quarantine tests
│   ├── test_serving_latency_benchmark.py # Sub-50ms Warm-Path Serving SLA Latency Benchmark
│   ├── test_streaming_sink.py         # DualPathRedisFeatureSink & DLQ Isolation tests
│   └── test_utils_clients.py          # MinIO & Redis Client connection & health tests
│
├── conftest.py                        # Root Pytest Configuration & Mock Fixtures
├── docker-compose.yml                 # Full stack container configuration (Redpanda, Redis, MinIO, Prometheus, Grafana)
├── requirements.txt                   # Dependencies manifest
└── .env.example                       # Environment variables template
```

---

## How to Run

### 1. Clone and Configure Environment

```bash
git clone <your-repo-url>
cd ml-feature-store-platform
cp .env.example .env
# Edit .env to adjust port configurations if needed
```

### 2. Launch Infrastructure Services (Docker)

Ensure Docker Desktop is running, then boot up Redpanda, Redis, MinIO, Prometheus, and Grafana:

```bash
docker compose up -d
```

Verify service status:
* **Redpanda Console:** `http://localhost:8080`
* **MinIO Console:** `http://localhost:9001` (User: `minioadmin`, Password: `minioadminpassword`)
* **Prometheus:** `http://localhost:9090`
* **Grafana:** `http://localhost:3000` (User: `admin`, Password: `admin`)

### 3. Setup Python Virtual Environment

```bash
# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 4. Run Test Suite Verification

Run the full pytest suite to verify system logic and module integrations:

```bash
PYTHONPATH=. pytest
```

### 5. Execute Pipeline Orchestration (Prefect Flow)

Execute the 6-step Prefect orchestration pipeline locally to populate batch feature stores, perform data quality assertions, materialize online features into Redis, run drift analysis, and train the initial Model Ensemble:

```bash
PYTHONPATH=. python src/orchestration/execute.py
```

*To run as a scheduled daemon with Prefect deployment:*
```bash
PYTHONPATH=. python src/orchestration/execute.py --serve --cron "0 0 * * 0"
```

### 6. Start Real-Time Ingestion Producer & Stream Job

Launch the transaction event producer to simulate live transaction traffic (run in separate terminal sessions):

```bash
# Terminal 1: Start High-Frequency Transaction Producer
PYTHONPATH=. python src/producer/producer.py
```

### 7. Launch Model Serving REST API (< 50ms P95 SLA)

Start the FastAPI low-latency inference server:

```bash
PYTHONPATH=. python src/api/main.py
```

Access API Swagger Interactive Documentation:
* Open **`http://localhost:8000/docs`** to test `/predict` and `/health` endpoints interactively.

*Sample `/predict` cURL Request:*
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"card_id": "11556", "current_amount": 750.0}'
```

### 8. Launch Interactive Streamlit BI Dashboard

Launch the Streamlit dashboard for real-time fraud monitoring, SHAP charts, and drift reports:

```bash
streamlit run dashboards/app.py
```
Open **`http://localhost:8501`** in your browser.
