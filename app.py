import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("HR_Analytics_Clean.csv")

# Clean column names by removing spaces
df.columns = df.columns.str.replace(" ", "")

st.set_page_config(page_title="HR Analytics Dashboard",
                   layout="wide",
                   page_icon="📊")

st.title("📊 HR Analytics Dashboard")
st.markdown("### Explore employee attrition & workforce factors with interactive filters")

# Sidebar filters
st.sidebar.header("Filter Data")

departments = df['Department'].unique().tolist()
selected_depts = st.sidebar.multiselect("Select Department(s)", options=departments, default=departments)

genders = df['Gender'].unique().tolist()
selected_genders = st.sidebar.multiselect("Select Gender(s)", options=genders, default=genders)

jobroles = df['JobRole'].unique().tolist()
selected_roles = st.sidebar.multiselect("Select Job Role(s)", options=jobroles, default=jobroles)

min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
selected_age = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

attrition_options = ['Yes', 'No']
selected_attrition = st.sidebar.multiselect("Attrition Status", options=attrition_options, default=attrition_options)

filtered_df = df[
    (df['Department'].isin(selected_depts)) &
    (df['Gender'].isin(selected_genders)) &
    (df['JobRole'].isin(selected_roles)) &
    (df['Age'] >= selected_age[0]) &
    (df['Age'] <= selected_age[1]) &
    (df['Attrition'].isin(selected_attrition))
].copy()

# Show filtered rows count in sidebar
st.sidebar.markdown(f"**Filtered dataset rows:** {filtered_df.shape[0]}")

# General stats
st.markdown("---")
st.markdown("## 📌 General Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Employees", filtered_df.shape[0])
with col2:
    st.metric("Attrition Rate (%)", round(filtered_df['Attrition'].value_counts(normalize=True).get("Yes", 0) * 100, 2))
with col3:
    st.metric("Average Age", round(filtered_df['Age'].mean(), 1))
with col4:
    st.metric("Average Monthly Income", round(filtered_df['MonthlyIncome'].mean(), 2))

# How is attrition dependent on Age?
st.markdown("---")
st.markdown("### How is attrition dependent on Age?")
age_att = filtered_df.groupby(['Age', 'Attrition']).size().reset_index(name='Counts')
fig_age_att = px.line(age_att, x='Age', y='Counts', color='Attrition', title='Agewise Counts of People in an Organization')
st.plotly_chart(fig_age_att, use_container_width=True)

st.markdown(
  """
    "<p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>"
    "Observation: As seen in the chart above, the attrition is maximum between the age groups 28-32. "
    "The attrition rate keeps on falling with increasing age, as people look after stability in their jobs at these point of times. "
    "Also at a very younger age, i.e. from 18-20, the chances of an employee leaving the organization is far more—since they are exploring at that point of time. "
    "It reaches a break even point at the age of 21."
    "</p>"
  """ , 
    unsafe_allow_html=True
)



# Is income the main factor towards employee attrition?
st.markdown("---")
st.markdown("### Is income the main factor towards employee attrition?")
filtered_df['MonthlyIncomeRounded'] = filtered_df['MonthlyIncome'].round(-3)

rate_att = filtered_df.groupby(['MonthlyIncomeRounded', 'Attrition']).size().reset_index(name='Counts')

fig_income = px.line(rate_att, x='MonthlyIncomeRounded', y='Counts', color='Attrition',
                     title='Monthly Income basis counts of People in an Organization')

st.plotly_chart(fig_income, use_container_width=True)

st.markdown(
    "<p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>"
    "Observation: As seen in the above chart, the attrition rate is evidently high at very low income levels - less than 5k monthly. "
    "This decreases further - but a minor spike is noticed around 10k - indicating the middle class livelihood. "
    "They tend to shift towards a better standard of living, and hence move to a different job. "
    "When the monthly income is pretty decent, the chances of an employee leaving the organization is low - as seen by the flat line."
    "</p>",
    unsafe_allow_html=True
)

# Does the Department of work impact attrition?
st.markdown("---")
st.markdown("### Does the Department of work impact attrition?")
dept_att = filtered_df.groupby(['Department', 'Attrition']).size().reset_index(name='Counts')

fig_dept_att = px.bar(dept_att, x='Department', y='Counts', color='Attrition',
                      title='Department wise Counts of People in an Organization')

st.plotly_chart(fig_dept_att, use_container_width=True)

st.markdown(
    "<p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>"
    "Observations: This data comprises of only 3 major departments — among which Sales department has the highest attrition rates (25.84%), "
    "followed by the Human Resource Department (19.05%). Research and Development has the least attrition rates, "
    "which suggests the stability and content of the department as can be seen from the chart above (13.83%)."
    "</p>",
    unsafe_allow_html=True
)

# How does the environment satisfaction impact attrition?
st.markdown("---")
st.markdown("### How does the environment satisfaction impact attrition?")
fig_env_sat_normalized = px.bar(
    filtered_df,
    x='EnvironmentSatisfaction',
    color='Attrition',
    title='Attrition Rate by Environment Satisfaction',
    barmode='group', # Or use barnorm='percent' for a 100% stacked bar
    text_auto=True,
    labels={'x': 'Environment Satisfaction Level'},
    category_orders={"EnvironmentSatisfaction": ["Low", "Medium", "High", "Very High"]} # Ensures correct order
)

# To calculate and show the percentage explicitly
fig_env_sat_percent = px.histogram(filtered_df,
    x='EnvironmentSatisfaction',
    color='Attrition',
    barnorm='percent', # This is the key change!
    text_auto='.2f',   # Format text to 2 decimal places
    title='Attrition Rate by Environment Satisfaction',
    labels={'x': 'Environment Satisfaction Level'},
    category_orders={"EnvironmentSatisfaction": ["Low", "Medium", "High", "Very High"]}
)


st.plotly_chart(fig_env_sat_percent, use_container_width=True) # Display the new chart

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The <b>attrition rate</b> is highest for employees with 'Low' Environment Satisfaction (approximately 25%). 
    As satisfaction improves, the attrition rate steadily decreases, dropping to around 15% for employees reporting 'High' 
    or 'Very High' satisfaction. This demonstrates that a poor work environment is a significant factor contributing 
    to an employee's decision to leave.
    </p>
    """,
    unsafe_allow_html=True
)

# How does self Job Satisfaction impact the Attrition?
st.markdown("---")
st.markdown("### How does Job Satisfaction impact Attrition?")

# The new code block for Job Satisfaction vs. Attrition
jsats_att = filtered_df.groupby(['JobSatisfaction', 'Attrition']).size().reset_index(name='Counts')
fig_jsats_att = px.area(
    jsats_att,
    x='JobSatisfaction',
    y='Counts',
    color='Attrition',
    title='Job Satisfaction level Counts of People in an Organization',
    category_orders={"JobSatisfaction": ["Low", "Medium", "High", "Very High"]}
)
st.plotly_chart(fig_jsats_att, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The chart shows a clear relationship between job satisfaction and attrition.
    The number of employees who have left (Attrition='Yes') is highest among those with **low job satisfaction**.
    As job satisfaction increases from 'Low' to 'Very High', the number of employees who have left steadily decreases,
    while the number of employees who have stayed increases. This suggests that a higher level of job satisfaction
    is a significant factor in employee retention.
    </p>
    """,
    unsafe_allow_html=True
)

# Does company stocks for employees impact attrition?
st.markdown("---")
st.markdown("### How does Stock Option Level impact Attrition?")

stock_att = filtered_df.groupby(['StockOptionLevel', 'Attrition']).size().reset_index(name='Counts')
fig_stock_att_percent = px.histogram(
    filtered_df,
    x='StockOptionLevel',
    color='Attrition',
    barnorm='percent',  # This shows the percentage of each group
    text_auto='.2f',    # Format text to 2 decimal places
    title='Attrition Rate by Stock Option Level',
    labels={'x': 'Stock Option Level'}
)
st.plotly_chart(fig_stock_att_percent, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The chart indicates that the highest attrition rate is for employees with **Stock Option Level 0** (approximately 25%). As the stock option level increases, the attrition rate significantly decreases. This suggests that providing employees with stock options is a strong incentive for retention, aligning their long-term financial interests with the company's success.
    </p>
    """,
    unsafe_allow_html=True
)

# How does Work Life Balance impact the overall attrition rates?
st.markdown("---")
st.markdown("### How does Work Life Balance impact Attrition?")

wlb_counts = filtered_df.groupby(['WorkLifeBalance', 'Attrition']).size().reset_index(name='Counts')
wlb_totals = filtered_df.groupby('WorkLifeBalance').size().reset_index(name='TotalCounts')

wlb_att_percent = pd.merge(wlb_counts, wlb_totals, on='WorkLifeBalance')
wlb_att_percent['Percentage'] = (wlb_att_percent['Counts'] / wlb_att_percent['TotalCounts']) * 100
fig_wlb_percent_grouped = px.bar(
    wlb_att_percent,
    x='WorkLifeBalance',
    y='Percentage',
    color='Attrition',
    barmode='group',
    title='Work Life Balance vs. Attrition (Percentage)',
    labels={'WorkLifeBalance': 'Work Life Balance Level', 'Percentage': 'Percentage of Employees'},
    category_orders={"WorkLifeBalance": ["Low", "Medium", "High", "Very High"]},
    text_auto='.2f' # Show percentages on the bars
)
st.plotly_chart(fig_wlb_percent_grouped, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: This grouped bar chart shows the attrition rate as a percentage for each work-life balance level. It is clear that employees with a **Low work-life balance** have the highest attrition rate. As the work-life balance improves, the percentage of employees who leave the company decreases significantly. This strongly suggests that a positive work-life balance is a critical factor for employee retention.
    </p>
    """,
    unsafe_allow_html=True
)

# How does work experience affect attrition?
st.markdown("---")
st.markdown("### How does Work Experience impact Attrition?")

ncwrd_att = filtered_df.groupby(['NumCompaniesWorked', 'Attrition']).size().reset_index(name='Counts')
fig_ncwrd_att = px.area(
    ncwrd_att,
    x='NumCompaniesWorked',
    y='Counts',
    color='Attrition',
    title='Work Experience level Counts of People in an Organization',
    labels={'NumCompaniesWorked': 'Number of Companies Worked'}
)
st.plotly_chart(fig_ncwrd_att, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The area chart illustrates the relationship between an employee's work experience (number of companies worked) and attrition. The highest number of employees who have left the company is among those who have worked at **one company** prior to the current one. The attrition rate is also significant for those who have worked at **0 companies**, as they might be new to the workforce and still exploring their options. As the number of companies worked increases, the total count of employees decreases, but the proportion of those who have left remains a key factor to analyze further.
    </p>
    """,
    unsafe_allow_html=True
)

# How does Work duration in current role impact Attrition?
st.markdown("---")
st.markdown("### How does Work duration in current role impact Attrition?")

yrscr_att = filtered_df.groupby(['YearsInCurrentRole', 'Attrition']).size().reset_index(name='Counts')
fig_yrscr_att = px.line(
    yrscr_att,
    x='YearsInCurrentRole',
    y='Counts',
    color='Attrition',
    title='Work duration in current role vs Attrition',
    labels={'YearsInCurrentRole': 'Years in Current Role'}
)
st.plotly_chart(fig_yrscr_att, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The line chart demonstrates a clear trend: attrition is highest for employees who have been in their current role for a **very short period**, particularly within the first one to two years. The number of employees leaving the organization decreases significantly as their tenure in the current role increases. This suggests that the initial phase of a new role is a critical period for employee retention, as dissatisfaction or a poor fit becomes apparent early on. Employees who stay in their role for a longer duration are less likely to leave.
    </p>
    """,
    unsafe_allow_html=True
)

# Does Hike percentage impact Attrition?
st.markdown("---")
st.markdown("### How does Salary Hike Percentage impact Attrition?")

hike_att = filtered_df.groupby(['PercentSalaryHike', 'Attrition']).size().reset_index(name='Counts')
fig_hike_att = px.line(
    hike_att,
    x='PercentSalaryHike',
    y='Counts',
    color='Attrition',
    title='Salary Hike Percentage vs Attrition',
    labels={'PercentSalaryHike': 'Percent Salary Hike'}
)
st.plotly_chart(fig_hike_att, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation: The line chart shows the relationship between an employee's salary hike percentage and attrition. It's clear that the attrition rate (the blue line) is higher for employees who receive **lower salary hike percentages** (11-15%). As the percentage of the salary hike increases, the number of employees who leave the organization generally decreases. This suggests that providing employees with a significant salary increase is a powerful tool for retention, as it demonstrates that their contributions are valued.
    </p>
    """,
    unsafe_allow_html=True
)

# Are managers a reason of people resigning??
st.markdown("---")
st.markdown("### Are managers a reason of people resigning?")

man_att = filtered_df.groupby(['YearsWithCurrentManager', 'Attrition']).size().reset_index(name='Counts')
fig_man_att = px.line(
    man_att,
    x='YearsWithCurrentManager',
    y='Counts',
    color='Attrition',
    title='Years with Current Manager vs Attrition',
    labels={'YearsWithCurrentManager': 'Years with Current Manager'}
)
st.plotly_chart(fig_man_att, use_container_width=True)

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    Observation We notice 3 major spikes in the attrition rate, when we are analyzing the relationship of an employee with their manager. At the very start, where the time spent with the manager is relatively less- people tend to leave their jobs- considering their relationship with their previous managers. At an average span of 2 years, when employees feel they need an improvement, they also tend to go for a change. When the time spent with the manager is slightly higher (about 7 years)- people tend to find their career progression stagnant, and tend to go for a change. But when the relative time spend with a manager is very high- people are satisfied with their work. Hence the chances of an employee resigning then is significantly low.
    </p>
    """,
    unsafe_allow_html=True
)



# --- START OF THE PREDICTION SECTION ---
st.markdown("---")
st.markdown("## 🔮 Predictive Attrition Analysis")
st.markdown("### How accurately can we predict employee attrition?")

st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: white; font-family: Arial;'>
    For this analysis, we will use two different Logistic Regression models. The second model, which uses a more comprehensive set of features, provides a more accurate prediction.
    </p>
    """,
    unsafe_allow_html=True
)

# Identify purely numerical columns and drop non-predictive ones
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols_filtered = [col for col in numerical_cols if col not in ['EmployeeCount', 'EmployeeNumber', 'StandardHours']]
X_1 = df[numerical_cols_filtered]
le = LabelEncoder()
y_1 = le.fit_transform(df['Attrition'])

# Create and save the correlation heatmap using Plotly Express
correlation_matrix = X_1.corr()
fig_heatmap = px.imshow(
    correlation_matrix,
    text_auto=True,
    title='Correlation Heatmap of Numerical Features',
    color_continuous_scale='RdBu',
    height=800,
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Split the data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, random_state=0)

# Train the logistic regression model
log_reg_1 = LogisticRegression(C=1000, max_iter=10000, solver='liblinear')
log_reg_1.fit(X_train_1, y_train_1)

# Display accuracy scores
st.markdown("---")
st.markdown("#### **Model Accuracy Scores (Numerical Features Only)**")
st.markdown(f"**Training Accuracy**: **`{log_reg_1.score(X_train_1, y_train_1):.3f}`**")
st.markdown(f"**Test Accuracy**: **`{log_reg_1.score(X_test_1, y_test_1):.3f}`**")


# One-hot encode the entire DataFrame
data_dummies = pd.get_dummies(df, drop_first=True)

# Define the features and target for the second model
features_to_drop = ['Attrition_Yes', 'EmployeeCount', 'EmployeeNumber', 'Over18_Y', 'StandardHours']
X_2 = data_dummies.drop(features_to_drop, axis=1, errors='ignore')
y_2 = data_dummies['Attrition_Yes'].values.ravel()

# Split the data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, random_state=0)

# Train the second logistic regression model
log_reg_2 = LogisticRegression(C=1000, max_iter=10000, solver='liblinear')
log_reg_2.fit(X_train_2, y_train_2)

# Display accuracy scores
st.markdown("---")
st.markdown("#### **Model Accuracy Scores (All Features)**")
st.markdown(f"**Training Accuracy**: **`{log_reg_2.score(X_train_2, y_train_2):.3f}`**")
st.markdown(f"**Test Accuracy**: **`{log_reg_2.score(X_test_2, y_test_2):.3f}`**")


# New observation and explanation section
st.markdown("---")
st.markdown("### 🔍 Observation and Explanation of the Models")
st.markdown("""
<p style='font-family:Arial;'>
After running two different models, we can make the following observations:
<ul>
    <li>The first model, which used only **numerical features**, served as a baseline. It provided a reasonable but limited predictive accuracy.</li>
    <li>The second model, which incorporated **all features (both numerical and categorical)** using a technique called **one-hot encoding**, demonstrated significantly higher accuracy.</li>
    <li>This is because categorical features like `Department`, `JobRole`, `MaritalStatus`, and `Overtime` contain crucial information about an employee's profile that is essential for predicting attrition.</li>
    <li>By converting these features into a format the model can understand, we provided a more complete picture of the data, which led to a **more robust and accurate predictive model**.</li>
    <li>The higher accuracy of the second model shows that a holistic view of the employee data is vital for building an effective attrition prediction system.</li>
</ul>
</p>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("## 🏁 Summary of Key Insights")
st.markdown("""
<p style='font-family:Arial;'>
We have checked the data, and have come upon to infer the following observations:
<br>
<b>Employee Career Stage and Attrition:</b>
<ul>
    <li>People tend to switch jobs more at the start of their careers. Once they find stability, they are more likely to stay long-term.</li>
</ul>

<b>Compensation and Motivation:</b>
<ul>
    <li><b>Salary and stock options</b> are great motivators. Higher pay and more stock options lead to lower attrition and higher employee loyalty.</li>
</ul>

<b>Work-Life Factors:</b>
<ul>
    <li><b>Work-life balance</b> is a major motivation factor. However, employees with a good work-life balance may still leave in search of better opportunities.</li>
    <li>Employees with good **Job Satisfaction** and **Environment Satisfaction** are more loyal. Dissatisfaction, particularly with a current project, increases the likelihood of an employee leaving.</li>
</ul>

<b>Departmental Impact on Attrition:</b>
<ul>
    <li>Departments with crucial performance targets (e.g., **Sales**) tend to have higher attrition compared to administrative departments (e.g., **Human Resources**).</li>
</ul>

<br>
<b>Highlights on the Data:</b>
<ul>
    <li>The dataset is limited to only 3 departments. To study attrition patterns more distinctly, it would be beneficial to have data from more departments and other organizations.</li>
</ul>
</p>
""", unsafe_allow_html=True)


