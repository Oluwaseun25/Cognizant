# Predictive Stock Level Management for Gala Groceries

## Background

Gala Groceries is a technology-driven grocery store chain based in the USA, leveraging IoT for competitive advantage. They pride themselves on providing fresh, locally sourced produce but face challenges with inventory management of perishable items. Overstocking leads to waste and storage costs, while understocking results in lost sales and dissatisfied customers.

## Problem Statement

“Can we accurately predict the stock levels of products based on sales data and sensor data on an hourly basis to more intelligently procure products from our suppliers?”

## Data Sources

1. **Sales Data (`sales`)**
   - Columns: `product_id`, `timestamp`, `quantity_sold`

2. **Sensor Storage Temperature Data (`sensor_storage_temperature`)**
   - Columns: `storage_id`, `timestamp`, `temperature`

3. **Sensor Stock Levels Data (`sensor_stock_levels`)**
   - Columns: `product_id`, `timestamp`, `stock_level`

## Strategic Plan

### Data Collection
- Gather historical sales, temperature, and stock level data.

### Data Preprocessing
- Clean and preprocess data.
- Handle missing values.
- Align timestamps across datasets for accurate merging.

### Feature Engineering
- Create new features such as average hourly temperature, cumulative sales, and stock depletion rates.
- Generate lag features to capture trends over time.

### Modeling
- Select appropriate machine learning models (e.g., Time Series models, Regression models).
- Train models using historical data to predict future stock levels.

### Model Evaluation
- Evaluate model performance using metrics like Mean Absolute Error and Root Mean Squared Error.
- Fine-tune models for improved accuracy.

### Implementation
- Develop a system to make real-time predictions using the trained model.
- Integrate predictions into the procurement system to dynamically adjust stock levels.

## Recommendations

- **Acquire More Data:** More extensive data sets are needed to capture long-term trends.
- **Gather Additional Data Types:**
  - **Inventory Data:** Current and historical inventory levels, turnover rates.
  - **Supplier Data:** Lead times, reliability, and supply chain disruptions.
  - **External Data:** Seasonal trends, weather patterns, holidays, and local events.

## Tools and Libraries Used

- **Python** for data analysis and modeling.
- **Pandas** for data manipulation.
- **Plotly** for interactive visualizations.
- **Scikit-learn** for machine learning models.
- **ARIMA** for time series analysis.

## Project Steps

1. **Data Modeling:** Understand the data relationships and how to merge tables using common columns.
2. **Strategic Planning:** Develop a plan for data collection, preprocessing, feature engineering, modeling, and implementation.
3. **Communication:** Summarize the plan and findings in a business-friendly manner.