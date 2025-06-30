# BallsAI 🏀

**An end-to-end data engineering and machine learning platform for NBA analytics, featuring real-time data, game predictions, and comprehensive player/team statistics through both web and mobile applications.**

BallsAI is a full-stack, end-to-end data platform built for comprehensive NBA analytics. It ingests real-time data, processes it through a modern data stack, serves it via a robust API, and presents insights through both web and mobile applications. The project includes machine learning models for predicting game outcomes and player performance.

---

## ✨ Features

- **Real-time Data Ingestion**: Daily data pipeline using **Airflow** to fetch game scores, boxscores, and schedules from the `nba-api`.
- **Advanced Analytics**: **dbt** models for transforming raw data into analytics-ready tables (e.g., season stats, head-to-head records).
- **Spark-powered Metrics**: **PySpark** jobs for complex calculations like Player Impact Rating (PIR), which are integrated back into the dbt models.
- **Machine Learning Predictions**:
  - **Game Predictor**: Predicts game winners and final scores using a PyTorch neural network.
  - **Player Projections**: Forecasts individual player statistics for upcoming games.
- **Cross-Platform Frontend**: A `pnpm` monorepo using **Turborepo** for high-performance builds.
  - **💻 Web App**: A feature-rich **Next.js** application for deep-diving into data.
  - **📱 Mobile App**: A cross-platform **React Native (Expo)** app for on-the-go access.
- **Robust Backend**: A **FastAPI** backend serving data and predictions with clear schemas, rate limiting, and automated documentation.
- **Containerized & Deployable**: Fully containerized with **Docker** and ready for **Kubernetes** deployment.

## 🏛️ Architecture & Tech Stack

BallsAI is built on a modern data stack, separating concerns from data ingestion to final presentation.

| Layer                       | Technology                                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Data Ingestion**          | **Apache Airflow** for orchestration, Python `nba_api` client.                                         |
| **Data Storage**            | **PostgreSQL** (Data Warehouse), **MinIO** (Raw Data Object Storage), **Redis** (Caching).             |
| **Data Transformation**     | **dbt** for SQL-based transformations, **PySpark** for advanced analytics.                             |
| **Backend & ML Serving**    | **FastAPI** (Python), **SQLAlchemy**.                                                                  |
| **Machine Learning**        | **PyTorch**, **Scikit-learn**.                                                                         |
| **Frontend (Web)**          | **Next.js**, **React**, **TypeScript**, **Tailwind CSS**, **TanStack Query**, **Shadcn/ui**.           |
| **Frontend (Mobile)**       | **Expo**, **React Native**, **TypeScript**, **NativeWind**, **Expo Router**, **TanStack Query**.         |
| **Authentication**          | **Better-Auth** for unified session management across web and native.                                  |
| **Infrastructure**          | **Docker**, **Docker Compose**, **Kubernetes**.                                                        |
| **Monorepo Tooling**        | **pnpm Workspaces**, **Turborepo**.                                                                    |

## 📂 Project Structure

The project is organized as a monorepo, with the backend/data components at the root and a dedicated `frontend` workspace.

```
.
├── airflow/                # Airflow DAGs and configurations
├── dbt/                    # dbt models for data transformation
├── docker/                 # Dockerfiles for custom service images
├── frontend/               # pnpm monorepo for all frontend apps & packages
│   ├── apps/
│   │   ├── native/         # Expo (React Native) mobile app
│   │   └── web/            # Next.js web app
│   └── packages/           # Shared packages (UI, API client, auth, etc.)
├── k8s/                    # Kubernetes manifests for deployment
├── scripts/                # Utility scripts (DB init, model training, deployment)
├── src/                    # Python source code
│   ├── analytics/          # PySpark jobs
│   ├── api/                # FastAPI application
│   ├── models/             # SQLAlchemy models & ML model definitions
│   └── utils/              # Shared Python utilities (data loader, client)
├── tests/                  # Backend tests
├── docker-compose.yml      # Local development environment setup
├── pyproject.toml          # Python project dependencies
└── README.md
```

## 🚀 Getting Started

Follow these instructions to get the entire platform running on your local machine.

### Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/products/docker-desktop/) & [Docker Compose](https://docs.docker.com/compose/)
- [pnpm](https://pnpm.io/installation) (`npm install -g pnpm`)
- Python 3.11

### 1. Clone the Repository

```bash
git clone https://github.com/harshsbajwa/BallsAI.git
cd BallsAI
```

### 2. Set Up Environment Variables

The project uses `.env` files for configuration. Start by copying the examples:

```bash
# Backend/Data .env (at the root)
# No .env file needed for local Docker setup, as variables are passed in docker-compose.yml

# Frontend Web App .env
cp frontend/apps/web/.env.example frontend/apps/web/.env.local

# Frontend Native App .env
cp frontend/apps/native/.env.example frontend/apps/native/.env
```

The default values in the `.env` files are configured for the local Docker Compose setup and should work out of the box.

### 3. Launch Backend & Data Services

Use Docker Compose to build and run all the backend services (Postgres, MinIO, Redis, API, Airflow).

```bash
docker-compose up -d --build
```

This will start all services in the background. You can monitor their logs with `docker-compose logs -f`.

### 4. Initialize Database & Train Models

The first time you run the project, you need to populate the database with historical data and train the machine learning models.

```bash
# (Optional but recommended) Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev] # Install project dependencies

# Run the initialization scripts
python scripts/init_database.py
python scripts/train_models.py
```

### 5. Run the Frontend

The frontend is a `pnpm` monorepo. The following commands will install dependencies and start the development servers for both the web and native apps.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
pnpm install

# Start both web and native dev servers
pnpm dev
```

### 6. Accessing the Services

Once everything is running, you can access the different parts of the platform:

- **🌐 Web App**: [http://localhost:3001](http://localhost:3001)
- **📱 Mobile App**: Scan the QR code from the `pnpm dev` output with the **Expo Go** app on your phone.
- **🐍 FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
- **📖 API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **💨 Airflow UI**: [http://localhost:8080](http://localhost:8080) (user: `airflow`, pass: `airflow`)
- **🗃️ MinIO Console**: [http://localhost:9001](http://localhost:9001) (user: `minioadmin`, pass: `minioadmin`)

## 🚢 Deployment

This project is configured for deployment on a Kubernetes cluster. The `/k8s` directory contains the necessary manifests for all services.

The `scripts/deploy.sh` script provides an automated way to:
1. Build the required Docker images.
2. Apply the Kubernetes manifests.
3. Wait for all pods to become ready.

**Note**: The script is configured for a **K3s** environment and may require adjustments for other Kubernetes distributions (e.g., changing `hostPath` for volumes).

To run the deployment:
```bash
./scripts/deploy.sh
```

## 📚 API Documentation

The FastAPI backend provides automatic, interactive API documentation. Once the API is running, you can access it at:

- **Swagger UI**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ReDoc**: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

## ✅ Testing

The backend includes a suite of tests using `pytest`. To run the tests:

```bash
# Ensure you have installed dev dependencies (pip install -e .[dev])
pytest
```
