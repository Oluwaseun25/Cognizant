# PRODUCTIONN CODE

## Introduction
This README provides detailed guidance for the creation of a Python module, **module_train_model.py**, aimed at training machine learning models and outputting performance metrics. The module is intended to facilitate the transition of the machine learning algorithm from a Python notebook to a production-ready format. 

## Task Overview
Gala Groceries seeks to integrate the promising machine learning model into their operational workflow. To achieve this, a Python module is required to encapsulate the model training process and provide comprehensive performance evaluation metrics. The module should adhere to best practices, ensuring clarity, consistency, and documentation.

## Steps to Completion

### Step 1: Plan
Before writing the module, meticulous planning is essential to ensure a clear and uniform structure. Consideration should be given to the organization of functions, constants, and documentation. The provided starter file, **module_starter.py**, offers insights into possible structuring approaches. Additionally, **module_helper.py** contains functions that may aid in this task.

### Step 2: Write
With a clear plan in place, proceed to write the Python module. Ensure adherence to a consistent and well-documented structure, incorporating best practices outlined in additional resources. Extensive comments and documentation are crucial for comprehension by the ML engineering team, who may not be familiar with the codebase. 

## Module Overview
The **module_train_model.py** file consists of several key components:

### 1. Importing Packages
Import statements are included to import necessary libraries for data manipulation, model training, and evaluation. Packages such as pandas, scikit-learn, and joblib are imported to facilitate these tasks.

### 2. Defining Global Constants
Global constants, including the path to the dataset, target variable, data split ratio, and random seed, are defined for consistent usage throughout the module.

### 3. Algorithm Code
The algorithm code encompasses functions responsible for loading data, creating target and predictor variables, scaling features, training and evaluating models, and performing cross-validation. Each function is documented extensively to clarify its purpose and usage.

### 4. Main Function
The main function orchestrates the execution of the module by sequentially calling the defined functions. It loads the dataset, preprocesses the data, splits it into training and testing sets, scales features, initializes machine learning models, trains and evaluates each model, selects the best-performing model based on mean absolute error (MAE), and saves the model and scaler objects for future use.

## Deliverables
Upon completion, the **module_train_model.py** file should be submitted for review by the ML engineering team. The file should adhere to best practices, be thoroughly documented, and demonstrate clear and consistent code organization.

## Additional Notes
- The provided code assumes that the CSV file containing data for model training has the same structure as the dataset used in the previous task.
