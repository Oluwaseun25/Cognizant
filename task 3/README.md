# Modeling and Analysis README

## Introduction
This README provides a comprehensive overview of the machine learning modeling process conducted to address the business problem statement. The task involved combining, transforming, and modeling three datasets provided by the client. The ultimate goal was to develop a predictive model to assist the business in managing stock levels effectively.

## Task Overview
The task involved implementing a strategic plan previously outlined for completing the modeling work. Two resources were provided to aid in this task: 
- `modeling.ipynb`: A Python notebook for those comfortable with independent modeling.

Both resources are intended for use either in Google Colab or within an individual's development environment.

## Libraries Used
The following libraries were utilized for data manipulation, modeling, and evaluation:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

## Findings
### Overview
- Three regression models were employed: Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor.
- Gradient Boosting Regressor exhibited the best performance among the three models.

### Final Model Performance
**Model**: Gradient Boosting Regressor
- **Mean Absolute Error (MAE)**: 0.1634
- **R-squared (R²)**: 0.2759

#### Interpretation
- The model's average prediction error is approximately 16.34%.
- It explains about 27.59% of the variance in stock levels, indicating it captures some patterns but has room for improvement.

### Feature Importance Analysis
The top 10 most important features influencing stock level predictions:
1. Stock Depletion Rate: 52.63%
2. Quantity: 27.27%
3. Unit Price: 3.54%
4. Hourly Sales: 2.63%
5. Timestamp Hour: 2.25%
6. Temperature Rolling Mean (3 hours): 2.22%
7. Temperature: 2.04%
8. Temperature Lag (1 hour): 1.61%
9. Timestamp Day of Week: 1.17%
10. Timestamp Day of Month: 1.11%

#### Analysis
- Stock Depletion Rate is the most significant predictor, accounting for over half of the model's decisions.
- Quantity, pricing, and sales data are also critical, indicating effective utilization of sales and inventory data.
- Temporal features provide valuable context for predicting stock levels.
- Temperature-related features suggest environmental conditions influence stock levels moderately.

### Conclusion
- The Gradient Boosting Regressor demonstrates promise in predicting stock levels with reasonable accuracy.
- Emphasis on stock depletion rate and sales data underscores the importance of real-time monitoring and historical sales trends in inventory management.
- Continuous improvement through feature engineering and model tuning can enhance predictive performance.

## Deliverables
The main deliverable for this task is a single PowerPoint slide summarizing the results in a business-friendly manner. The slide should explain whether the developed model can effectively address the problem statement using non-technical language and understandable metrics.
