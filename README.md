# Air Quality Forecasting System

<p align="center">
  <b>A Machine Learning Project for AQI Forecasting and Air Pollution Analysis Across Indian Cities</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>

## Overview

Air pollution is a major environmental challenge that affects public health and quality of life. Analyzing historical air quality data can help identify pollution patterns and estimate future pollution levels.

In this project, I developed a machine learning pipeline to analyze historical air quality data from Indian cities and forecast the Air Quality Index (AQI) and selected pollutant concentrations.

The project covers data preprocessing, exploratory data analysis, feature engineering, time-series forecasting, model evaluation, and interactive visualization.

## Objectives

- Predict future Air Quality Index (AQI) values.
- Forecast major pollutants such as PM2.5 and PM10.
- Analyze seasonal and temporal air pollution patterns.
- Compare statistical and machine learning forecasting models.
- Identify important features associated with AQI.
- Visualize historical trends and future predictions.
- Highlight predicted AQI values that cross configured safety thresholds.

## Dataset

**Source:** [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The dataset contains historical air quality measurements collected across multiple Indian cities and monitoring stations.

### Dataset Files

| File | Description |
|---|---|
| `city_day.csv` | Daily air quality measurements by city |
| `city_hour.csv` | Hourly air quality measurements by city |
| `station_day.csv` | Daily measurements by monitoring station |
| `station_hour.csv` | Hourly measurements by monitoring station |
| `stations.csv` | Monitoring station metadata |

The primary dataset used for daily forecasting is `city_day.csv`.

### Important Features

- `City`: Name of the city.
- `Date`: Date of observation.
- `PM2.5`: Fine particulate matter concentration.
- `PM10`: Coarse particulate matter concentration.
- `NO`: Nitric oxide concentration.
- `NO2`: Nitrogen dioxide concentration.
- `NOx`: Nitrogen oxides concentration.
- `NH3`: Ammonia concentration.
- `CO`: Carbon monoxide concentration.
- `SO2`: Sulfur dioxide concentration.
- `O3`: Ozone concentration.
- `AQI`: Air Quality Index.
- `AQI_Bucket`: AQI category.

## Project Structure

```text
air-quality-prediction/
│
├── data/
│   ├── raw/
│   │   ├── city_day.csv
│   │   ├── city_hour.csv
│   │   ├── station_day.csv
│   │   ├── station_hour.csv
│   │   └── stations.csv
│   │
│   └── processed/
│       ├── preprocessed_city_day.csv
│       └── features_city_day.csv
│
├── scripts/
│   ├── utils/
│   │   └── hourly_to_daily.py
│   ├── 01_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_features.py
│   └── 04_modeling.py
│
├── models/
│
├── run_all.py
├── app.py
├── pyproject.toml
└── README.md