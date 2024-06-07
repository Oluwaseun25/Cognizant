# Gala Groceries Inventory Management Analysis

## Background Information

Gala Groceries is a technology-led grocery store chain based in the USA, leveraging technologies such as IoT to gain a competitive edge. They emphasize providing high-quality, fresh produce from locally sourced suppliers, which presents challenges in maintaining consistent delivery year-round. Cognizant has been approached to help address their supply chain issue, focusing on the balance between overstocking and understocking of perishable items.

## Objective

To explore the data, formulate questions, and provide recommendations to Gala Groceries on how to better stock their perishable items, ensuring an optimal balance that avoids both waste from overstocking and lost sales from understocking.

## Problem Formulation

### Key Questions

1. **Demand Forecasting:**
   - What are the historical sales patterns for different items?
   - How do seasonal trends impact sales?
   - Are there any external factors (e.g., holidays, weather) that significantly affect demand?

2. **Supply Chain Efficiency:**
   - What is the lead time from suppliers for each product?
   - How often do supply chain disruptions occur, and what are their causes?

3. **Stock Optimization:**
   - What are the current inventory levels, and how do they fluctuate?
   - What are the storage capacities and costs?

## Data Sources Needed

To answer these questions, we need the following data sources:

- **Sales Data**: Historical sales data, segmented by product, time, and location.
- **Inventory Data**: Current and historical inventory levels, turnover rates.
- **Supplier Data**: Lead times, reliability, and supply chain disruptions.
- **External Data**: Seasonal trends, weather patterns, holidays, and local events.

## Findings

From the initial analysis of a small subset of data, we have gathered the following insights:

- **Sales Trend Analysis**:
  - The current sample includes data from only one store and one week, which limits the ability to observe long-term trends.

- **Frequency of Sales by Category**:
  - Fruits and vegetables are the most purchased categories, with over 800 vegetables and approximately 1000 fruits sold in a week.

- **Total Spending and Frequency of Purchase by Customer Type**:
  - Non-members have the highest spending score and purchase frequency based on the one-week data.

- **Daily Quantities Sold for Top 10 Categories**:
  - Fruits and vegetables are the highest quantities sold daily on average.

- **Frequency of Maximum Quantity Purchased**:
  - Fruits and vegetables have the highest frequency of maximum quantity purchased.

## Recommendations

To better address Gala Groceries' inventory management problem, we recommend the following actions:

- **Acquire More Data**: Collect more rows of data to get a clearer picture of sales trends over time.
- **Gather Additional Data Types**:
  - **Inventory Data**: Include current and historical inventory levels, and turnover rates.
  - **Supplier Data**: Gather information on lead times, reliability, and supply chain disruptions.
  - **External Data**: Collect data on seasonal trends, weather patterns, holidays, and local events to understand external impacts on demand.

## Tools and Libraries Used

- **Python**: Programming language used for data analysis.
- **Pandas**: Library used for data manipulation and analysis.
- **Plotly**: Library used for creating interactive visualizations.
- **jupyter notebook**: Platform used for running Jupyter notebooks in the cloud.
- **matplotlib**: Platform used for running Jupyter notebooks in the cloud.
- **seaborn**: Platform used for running Jupyter notebooks in the cloud.

## Steps to Complete Analysis

1. **Prepare:**
   - Download the provided CSV file (`sample_sales_data.csv`).
   - Use the provided notebooks (`eda.ipynb` and `eda_walkthrough.ipynb`) to start your analysis. 

2. **Exploration:**
   - Conduct exploratory data analysis using the notebooks. 
   - Gain a solid understanding of the statistical properties of the dataset.

3. **Communication:**
   - Summarize your findings and recommendations in a business-friendly email to the Data Science team leader for review.















