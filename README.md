# Predictive Maintenance

**Course project for Big Data Analysis and Business Intelligence**

This project builds a data pipeline for predictive maintenance using the NASA Bearing Dataset. The pipeline streams sensor data, processes it, stores the processed output, and supports business intelligence reporting through Power BI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Power BI Dashboard Showcase](#power-bi-dashboard-showcase)
- [Dataset Setup](#dataset-setup)
- [How to Run](#how-to-run)
- [Important Notes](#important-notes)
- [How to Stop](#how-to-stop)

---

## Project Overview

Predictive maintenance is used to monitor machine condition and identify early signs of equipment failure. In this project, bearing sensor data is processed through a big data pipeline and prepared for analysis and visualization.

The main goals of this project are to:

- Ingest bearing sensor data into a streaming pipeline.
- Process sensor readings for predictive maintenance analysis.
- Store processed data in PostgreSQL.
- Visualize machine condition and maintenance insights in Power BI.

---

## Architecture

The project uses Docker Compose to run the main services together.

```text
NASA Bearing Dataset
        |
        v
Producer
        |
        v
Kafka + Zookeeper
        |
        v
Spark Processing
        |
        v
PostgreSQL
        |
        v
Power BI Dashboard
```

---

## Tech Stack

- **Python** - data producer and processing logic
- **Apache Kafka** - streaming sensor data
- **Zookeeper** - Kafka coordination
- **Apache Spark** - big data processing
- **PostgreSQL** - processed data storage
- **Docker / Docker Compose** - containerized deployment
- **Power BI** - dashboard and business intelligence visualization

---

## Repository Structure

```text
predictive_maintainance/
├── Producer/
├── Processing/
├── postgresql_init/
├── NASA_Bearing_Data/          # Add this folder manually after downloading the dataset
├── assets/
│   └── powerbi/
│       ├── dashboard_overview.jpg
│       ├── bearing_health_summary.jpg
│       ├── sensor_trend_analysis.jpg
│       └── maintenance_insights.jpg
├── docker-compose.yml
└── README.md
```

> Note: `NASA_Bearing_Data/` is not included in the repository. You need to download the dataset manually.

---

## Power BI Dashboard Showcase

Add your exported Power BI dashboard screenshots as `.jpg` files inside:

```text
assets/powerbi/
```

Recommended file names:

```text
dashboard_overview.jpg
bearing_health_summary.jpg
sensor_trend_analysis.jpg
maintenance_insights.jpg
```

### Dashboard Overview

<p align="center">
  <img src="Dashboard (Power BI)/Overview.jpg" alt="Power BI Dashboard Overview" width="850">
</p>

### Sensor Trend Analysis

<p align="center">
  <img src="assets/powerbi/Diagnostics (Trend).jpg" alt="Sensor Trend Analysis Dashboard" width="850">
</p>

### Maintenance Insights

<p align="center">
  <img src="assets/powerbi/Diagnostics (Risk-Reward).jpg" alt="Maintenance Insights Dashboard" width="850">
</p>

---

## Dataset Setup

First, download the **NASA Bearing Dataset** from Kaggle.

After downloading, place the dataset folder in the same directory as this repository.

Example:

```text
predictive_maintainance/
├── NASA_Bearing_Data/
├── Producer/
├── Processing/
├── docker-compose.yml
└── README.md
```

Your dataset folder should match the volume mapping in `docker-compose.yml`:

```yaml
producer:
  build: ./Producer
  depends_on:
    - kafka
  volumes:
    - ./NASA_Bearing_Data:/app/NASA_Bearing_Data
```

If your dataset folder has a different name or location, update the volume path in `docker-compose.yml`.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/chaunguye/predictive_maintainance.git
cd predictive_maintainance
```

### 2. Add the dataset

Place the downloaded dataset folder in the project root and name it:

```text
NASA_Bearing_Data
```

### 3. Check line endings for the Spark start script

Before running the project, make sure `Processing/start.sh` uses **LF** line endings instead of **CRLF**.

In VS Code:

1. Open `Processing/start.sh`.
2. Look at the bottom-right corner of the window.
3. If it says `CRLF`, click it.
4. Choose `LF`.
5. Save the file.

### 4. Build and start the services

```bash
docker-compose up --build
```

Docker Compose will start the required services, including Kafka, Zookeeper, Spark, PostgreSQL, and the producer.

---

## Important Notes

- Make sure Docker Desktop is running before starting the project.
- Make sure the dataset path in `docker-compose.yml` is correct.
- If `Processing/start.sh` uses `CRLF`, the Spark container may fail to start.
- The Power BI screenshots must be committed to the repository for them to appear in the README.
- If an image does not appear on GitHub, check that the file path and filename match exactly.

---

## How to Stop

To stop and remove the running containers and volumes, run:

```bash
docker-compose down -v
```

---

## Suggested Git Commands for Adding Power BI Images

```bash
mkdir -p assets/powerbi

# Copy your exported .jpg Power BI screenshots into assets/powerbi/
# Example filenames:
# dashboard_overview.jpg
# bearing_health_summary.jpg
# sensor_trend_analysis.jpg
# maintenance_insights.jpg

git add README.md assets/powerbi/*.jpg
git commit -m "Update README with Power BI dashboard showcase"
git push
```
