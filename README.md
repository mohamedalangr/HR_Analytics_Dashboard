# HR Analytics: Employee Attrition Analysis

## Project Structure

The project is organized into a clear directory structure to facilitate navigation and reproducibility.

```
├── HR Analytics.csv
├── HR_Analytics_Clean.csv
├── DA_Project_PRE.ipynb
├── app.py
├── README.md

```

* **`HR Analytics.csv`**: The original, raw dataset.
* **`HR_Analytics_Clean.csv`**: The cleaned and preprocessed dataset.
* **`DA_Project_PRE.ipynb`**: A Jupyter Notebook containing the full data analysis workflow, from cleaning to modeling.
* **`app.py`**: A Streamlit script that runs the interactive web dashboard.
* **`README.md`**: This file, providing an overview of the project.

## Project Overview

This project provides a comprehensive analysis of an HR dataset to identify and understand the key factors contributing to employee attrition. The goal is to deliver actionable insights that can help improve employee retention and satisfaction. The project is built around a data analysis workflow and an interactive web dashboard.

**Authors:** Mohamed Fathy, Mohamed Haitham

## Methodology

1.  **Data Preprocessing:** The raw dataset was cleaned by handling missing values, standardizing column names, and performing feature engineering.
2.  **Exploratory Data Analysis (EDA):** Visualizations were used to uncover relationships between employee attributes and attrition rates.
3.  **Predictive Modeling:** A Logistic Regression model was trained to predict the likelihood of an employee leaving the company.

## Key Findings

* **Overtime:** Employees who work overtime have a significantly higher attrition rate.
* **Job Satisfaction:** Low job and environment satisfaction scores are strongly correlated with a higher risk of attrition.
* **Compensation:** There is a clear inverse relationship between monthly income and attrition.
* **Departmental Turnover:** The Sales department exhibits a higher attrition rate compared to others.

## How to Run the Project

To run the interactive dashboard locally, you will need to have Python and the required libraries installed.

1.  Clone this repository to your local machine.
2.  Install the necessary libraries (e.g., `streamlit`, `pandas`, `scikit-learn`, `plotly`):
    ```bash
    pip install streamlit pandas scikit-learn plotly
    ```
3.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
4.  The dashboard will automatically open in your web browser.
5.  Try it Live [https://hrdashboardd.streamlit.app/]

## Data Source

The dataset used in this project was sourced from Kaggle.
