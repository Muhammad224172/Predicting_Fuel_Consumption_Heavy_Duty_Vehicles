# Graduation Project 2024  
**Predicting Fuel Consumption in Heavy-Duty Vehicles**

## Problem Statement
Heavy-duty trucks account for a significant share of fuel consumption in the transportation industry. This high usage leads to increased operational costs and greater environmental impact. Efficient modeling of fuel consumption is therefore essential for optimizing truck route planning and reducing costs.  

This project develops accurate machine-learning models to predict fuel consumption in heavy-duty trucks, with a particular focus on **feature subset selection**.  

Previous studies have explored various machine-learning models and feature-selection techniques for fuel-consumption prediction. However, many of these models lack generalization due to issues such as inadequate data preprocessing, overfitting, or sub-optimal feature selection.  
Our goal is to build a **robust and reliable predictive model** that addresses these issues and delivers consistent performance.

## Dataset Features
The dataset includes the following key features:

- `EngSpeed`
- `ActualEngPercentTorque`
- `EngTotalFuelUsed`
- `TachographVehicleSpeed`
- `AccelPedalPos1`
- `BrakePedalPos`
- `EngOilTemp1`
- `EngCoolantTemp`
- `GrossCombinationVehicleWeight`
- `HghRslutionTotalVehicleDistance`
- `DStgy_dmRdcAgAct`
- `PCCM_Slope`

## Methodology
Two machine-learning models were compared:

- **XGBoost**
- **Random Forest**

A **10-fold cross-validation** strategy was applied to evaluate model performance and ensure robust, unbiased results.
