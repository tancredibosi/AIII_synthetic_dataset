# Synthetic Data Generation for HR Dataset

This project focuses on generating synthetic HR data while maintaining statistical properties and business constraints from the original dataset. It uses the Synthetic Data Vault (SDV) library to create realistic synthetic data for testing and analysis purposes.

## Features

- **Data Preprocessing**: Cleans and organizes raw HR data
- **Constraints**: Implements constraints to have data validity
- **Synthetic Data Generation**: Uses Gaussian Copula synthesizer
- **Data Polarization**: Generates data with specific distribution requirements
- **Quality Evaluation**: Compares synthetic data quality against original
- **Visualization**: Provides comparison plots between different synthesizers

## Key Components

### 1. Data Preprocessing (`data_preprocess_utils.py`)
- Cleans and organizes raw data
- Handles missing values and inconsistencies
- Clusters similar job roles and tags
- Filters out invalid age-experience combinations

### 2. Custom Constraints (`custom_constraint_years.py`)
Implements business rules:
- Hired candidates must have `Year of insertion` ≤ `Year of Recruitment`
- Non-hired candidates must have `Year of Recruitment` as NaN

### 3. Data Generation (`main.py`)
- Uses Gaussian Copula synthesizer
- Applies custom constraints
- Generates polarized data with specific distributions

### 4. Quality Evaluation (`utils.py`)
- Compares synthetic data against original
- Provides diagnostic and quality scores
- Visualizes comparison between different synthesizers
