## Gala Groceries Inventory Management - SUMMARY

### Overview
This README provides a comprehensive overview of the tasks undertaken for Gala Groceries, a technology-driven grocery store chain based in the USA. The project aims to improve inventory management for perishable items using machine learning. The tasks are segmented into five key phases: data exploration and analysis, predictive modeling, modeling and analysis, production code development, and quality control.

---

### Task 1: Gala Groceries Inventory Management Analysis

**Background:**
Gala Groceries leverages IoT technology to maintain a competitive edge. They face challenges in balancing overstocking and understocking of perishable items. 

**Objective:**
To analyze sales data and provide recommendations on better stock management of perishable items.

**Problem Formulation:**
- **Key Questions:**
  - Historical sales patterns and seasonal trends
  - Impact of external factors like holidays and weather
  - Current inventory levels and fluctuations
- **Data Sources Needed:**
  - Sales Data
  - Inventory Data
  - Supplier Data
  - External Data (weather, holidays, etc.)

**Findings:**
- **Sales Trend Analysis:**
  - Limited long-term trend visibility due to small dataset
- **Category Insights:**
  - Fruits and vegetables have the highest sales and purchase frequencies
- **Recommendations:**
  - Acquire more extensive data
  - Include additional data types like inventory and supplier information

**Tools and Libraries:**
- Python, Pandas, Plotly, Jupyter Notebook, Matplotlib, Seaborn

**Steps to Complete Analysis:**
1. Prepare: Load and review the dataset.
2. Exploration: Conduct exploratory data analysis.
3. Communication: Summarize findings and recommendations.

---

### Task 2: Predictive Stock Level Management

**Background:**
Gala Groceries aims to leverage sales and sensor data to predict stock levels and optimize procurement.

**Problem Statement:**
"Can we accurately predict the stock levels of products based on sales and sensor data on an hourly basis?"

**Data Sources:**
- **Sales Data:** product_id, timestamp, quantity_sold
- **Sensor Data:** storage_id, timestamp, temperature, stock_level

**Strategic Plan:**
1. **Data Collection:**
   - Gather historical sales, temperature, and stock level data.
2. **Data Preprocessing:**
   - Clean and align data.
3. **Feature Engineering:**
   - Create features like average hourly temperature and stock depletion rates.
4. **Modeling:**
   - Train models (e.g., Time Series, Regression).
5. **Evaluation:**
   - Assess models using MAE and RMSE.
6. **Implementation:**
   - Develop a real-time prediction system integrated into procurement.

**Recommendations:**
- Acquire more extensive and diverse data sets.

**Tools and Libraries:**
- Python, Pandas, Plotly, Scikit-learn, ARIMA

---

### Task 3: Modeling and Analysis

**Introduction:**
This task focuses on developing a predictive model to assist in stock level management.

**Libraries Used:**
- Pandas, Numpy, Scikit-learn, Matplotlib, Warnings

**Findings:**
- **Models Used:** Linear Regression, Gradient Boosting Regressor, Random Forest Regressor
- **Best Model:** Gradient Boosting Regressor
  - **MAE:** 0.1634
  - **R-squared:** 0.2759

**Feature Importance Analysis:**
- Stock Depletion Rate: 52.63%
- Quantity: 27.27%
- Other features: Unit Price, Hourly Sales, Timestamp, Temperature

**Conclusion:**
- Gradient Boosting Regressor is promising but requires further improvement.
- Emphasis on real-time monitoring and historical trends.

**Deliverables:**
- A single PowerPoint slide summarizing results for business stakeholders.

---

### Task 4: Production Code

**Introduction:**
Development of a Python module (`module_train_model.py`) to train and evaluate the model for production deployment.

**Steps to Completion:**
1. **Plan:**
   - Structure the module, incorporating functions and constants.
2. **Write:**
   - Follow best practices and document extensively.

**Module Overview:**
- **Importing Packages:**
  - Pandas, Scikit-learn, Joblib
- **Defining Constants:**
  - DATA_PATH, TARGET, SPLIT, SEED
- **Algorithm Code:**
  - Functions for loading data, creating target and predictors, scaling features, training and evaluating models, cross-validation.
- **Main Function:**
  - Orchestrates the execution: data loading, preprocessing, splitting, scaling, model training, evaluation, and saving.

**Deliverables:**
- Submit `module_train_model.py` for ML engineering team review.

---

### Task 5: Quality Control

**Introduction:**
Quality control phase involving addressing potential problems with the ML solution and proposing improvements.

**Key Questions Addressed:**
- Potential issues with the current model
- Strategies for model improvement
- Ensuring robustness and scalability of the model
- Addressing data quality issues
- Enhancing feature engineering
- Monitoring and maintaining the model in production

**Recommendations:**
- Continuous monitoring and updating of the model
- Incorporate feedback loops for model retraining
- Regularly evaluate and address data quality
- Enhance feature engineering based on new insights

---

### Conclusion

This project has covered the end-to-end process of developing a machine learning model for inventory management at Gala Groceries. From initial data analysis to model development and production readiness, each task has been meticulously executed to ensure a robust and scalable solution. Continuous improvement and quality control are essential for maintaining the model's performance and adapting to evolving business needs.

---
