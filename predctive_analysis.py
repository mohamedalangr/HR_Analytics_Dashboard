import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(file_path="HR_Analytics_Clean.csv"):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(" ", "")
    return df

df = load_data()

# --- Predictive Model Training ---
@st.cache_resource
def train_model(data):
    # One-hot encode the entire DataFrame
    data_dummies = pd.get_dummies(data, drop_first=True)
    # Define the features and target for the model
    features_to_drop = ['Attrition_Yes', 'EmployeeCount', 'EmployeeNumber', 'Over18_Y', 'StandardHours']
    X = data_dummies.drop(features_to_drop, axis=1, errors='ignore')
    y = data_dummies['Attrition_Yes'].values.ravel()
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Train the logistic regression model
    log_reg = LogisticRegression(C=1000, max_iter=10000, solver='liblinear')
    log_reg.fit(X_train, y_train)
    return log_reg, X_test, y_test, X.columns, X

log_reg_model, X_test_df, y_test_series, feature_names, X_full_df = train_model(df)


# --- Dashboard Layout and Styling ---
st.set_page_config(page_title="HR Analytics Dashboard",
                   layout="wide",
                   page_icon="📊")

# Custom CSS for a clean look
st.markdown(
    """
<style>
    .reportview-container {
        background: #0E1117;
        color: white;
    }
    .css-1d391kg {
        padding: 0rem 1rem;
    }
    .css-1aumxpa {
        background-color: #1E212B;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .st-bv, .st-da, .st-bb, .st-ba, .st-bc, .st-bd {
        color: #B0B0B0;
    }
    .st-bh, .st-bi {
        background-color: #2F333F;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar for Filters and Prediction Form ---
st.sidebar.header("Filter Dashboard")
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


# --- Main Dashboard Sections ---
st.title("📊 HR Analytics Dashboard - Interactive Insights")
st.markdown("### Explore employee attrition & workforce factors with interactive filters")

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

# --- Dashboard Plots ---
st.markdown("---")
st.markdown("## 📈 Key Attrition Drivers")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    age_att = filtered_df.groupby(['Age', 'Attrition']).size().reset_index(name='Counts')
    fig_age_att = px.line(age_att, x='Age', y='Counts', color='Attrition', title='Attrition by Age', template="plotly_dark")
    st.plotly_chart(fig_age_att, use_container_width=True)

with col_plot2:
    filtered_df['MonthlyIncomeRounded'] = filtered_df['MonthlyIncome'].round(-3)
    rate_att = filtered_df.groupby(['MonthlyIncomeRounded', 'Attrition']).size().reset_index(name='Counts')
    fig_income = px.line(rate_att, x='MonthlyIncomeRounded', y='Counts', color='Attrition',
                         title='Attrition by Monthly Income', template="plotly_dark")
    st.plotly_chart(fig_income, use_container_width=True)

col_plot3, col_plot4 = st.columns(2)

with col_plot3:
    dept_att = filtered_df.groupby(['Department', 'Attrition']).size().reset_index(name='Counts')
    fig_dept_att = px.bar(dept_att, x='Department', y='Counts', color='Attrition',
                          title='Attrition by Department', template="plotly_dark")
    st.plotly_chart(fig_dept_att, use_container_width=True)

with col_plot4:
    fig_env_sat_percent = px.histogram(filtered_df,
        x='EnvironmentSatisfaction', color='Attrition', barnorm='percent', text_auto='.2f',
        title='Attrition by Environment Satisfaction', labels={'x': 'Environment Satisfaction Level'},
        category_orders={"EnvironmentSatisfaction": sorted(filtered_df['EnvironmentSatisfaction'].unique().tolist())}, template="plotly_dark")
    st.plotly_chart(fig_env_sat_percent, use_container_width=True)

col_plot5, col_plot6 = st.columns(2)

with col_plot5:
    jsats_att = filtered_df.groupby(['JobSatisfaction', 'Attrition']).size().reset_index(name='Counts')
    fig_jsats_att = px.area(
        jsats_att,
        x='JobSatisfaction',
        y='Counts',
        color='Attrition',
        title='Attrition by Job Satisfaction',
        category_orders={"JobSatisfaction": sorted(filtered_df['JobSatisfaction'].unique().tolist())},
        template="plotly_dark"
    )
    st.plotly_chart(fig_jsats_att, use_container_width=True)

with col_plot6:
    fig_stock_att_percent = px.histogram(
        filtered_df,
        x='StockOptionLevel',
        color='Attrition',
        barnorm='percent',
        text_auto='.2f',
        title='Attrition by Stock Option Level',
        labels={'x': 'Stock Option Level'},
        template="plotly_dark"
    )
    st.plotly_chart(fig_stock_att_percent, use_container_width=True)

col_plot7, col_plot8 = st.columns(2)

with col_plot7:
    wlb_counts = filtered_df.groupby(['WorkLifeBalance', 'Attrition']).size().reset_index(name='Counts')
    fig_wlb_percent_grouped = px.bar(
        wlb_counts,
        x='WorkLifeBalance',
        y='Counts',
        color='Attrition',
        barmode='group',
        title='Attrition by Work Life Balance',
        labels={'WorkLifeBalance': 'Work Life Balance Level', 'Counts': 'Number of Employees'},
        category_orders={"WorkLifeBalance": sorted(filtered_df['WorkLifeBalance'].unique().tolist())},
        text_auto=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig_wlb_percent_grouped, use_container_width=True)

with col_plot8:
    ncwrd_att = filtered_df.groupby(['NumCompaniesWorked', 'Attrition']).size().reset_index(name='Counts')
    fig_ncwrd_att = px.area(
        ncwrd_att,
        x='NumCompaniesWorked',
        y='Counts',
        color='Attrition',
        title='Attrition by Number of Companies Worked',
        labels={'NumCompaniesWorked': 'Number of Companies Worked'},
        template="plotly_dark"
    )
    st.plotly_chart(fig_ncwrd_att, use_container_width=True)

# Predictive Analysis Section
st.markdown("---")
st.markdown("## 🔮 Predictive Attrition Analysis")
st.markdown("### Model Performance")

col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    # Feature Importance Plot
    coeffs = pd.DataFrame(log_reg_model.coef_[0], index=feature_names, columns=['Coefficient'])
    coeffs['Absolute Value'] = np.abs(coeffs['Coefficient'])
    top_features = coeffs.sort_values('Absolute Value', ascending=False).head(15)

    fig_feat_imp = px.bar(top_features, x=top_features.index, y='Coefficient',
                          title='Top 15 Most Important Features for Prediction',
                          color='Coefficient',
                          color_continuous_scale=px.colors.diverging.RdBu,
                          template="plotly_dark")
    st.plotly_chart(fig_feat_imp, use_container_width=True)

with col_pred2:
    # Classification Report
    st.markdown("#### Model Classification Report (All Features)")
    y_pred = log_reg_model.predict(X_test_df)
    report = classification_report(y_test_series, y_pred, output_dict=True, target_names=['Stayed', 'Left'])
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Model Accuracy**")
    st.markdown(f"**Test Accuracy**: **`{log_reg_model.score(X_test_df, y_test_series):.3f}`**")


# --- Interactive Prediction Form (in sidebar) ---
st.sidebar.markdown("---")
st.sidebar.header("Predict Attrition for an Employee")
with st.sidebar.form(key='prediction_form'):
    st.markdown("**Enter employee details:**")
    age = st.number_input('Age', min_value=18, max_value=60, value=30)
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
    overtime = st.selectbox('Overtime', ['No', 'Yes'])
    department = st.selectbox('Department', ['Research & Development', 'Sales', 'Human Resources'])
    years_at_company = st.number_input('Years At Company', min_value=0, max_value=40, value=5)
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}[x])
    environment_satisfaction = st.selectbox('Environment Satisfaction', [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}[x])
    total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, value=10)
    job_level = st.selectbox('Job Level', [1, 2, 3, 4, 5])
    distance_from_home = st.number_input('Distance From Home', min_value=1, max_value=29, value=5)

    submit_button = st.form_submit_button(label='Predict Attrition')

if submit_button:
    input_data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company,
        'JobSatisfaction': job_satisfaction,
        'EnvironmentSatisfaction': environment_satisfaction,
        'TotalWorkingYears': total_working_years,
        'JobLevel': job_level,
        'DistanceFromHome': distance_from_home,
    }
    # Add one-hot encoded features
    input_data[f'Overtime_Yes'] = 1 if overtime == 'Yes' else 0
    for dept in ['Sales', 'Research & Development']:
        input_data[f'Department_{dept}'] = 1 if department == dept else 0
    for role in ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']:
        if role != 'Human Resources': # Default for drop_first
            input_data[f'JobRole_{role}'] = 1 if job_role == role else 0

    # Create a DataFrame for the prediction
    input_df = pd.DataFrame([input_data])
    
    # Align columns with the training data
    missing_cols = set(feature_names) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0
    input_df = input_df[feature_names]

    # Predict and display result
    prediction = log_reg_model.predict(input_df)
    prediction_proba = log_reg_model.predict_proba(input_df)
    
    st.sidebar.subheader("Prediction Result")
    if prediction[0] == 1:
        st.sidebar.error("Prediction: Employee is likely to leave")
    else:
        st.sidebar.success("Prediction: Employee is likely to stay")
    
    st.sidebar.markdown(f"**Probability of leaving**: `{prediction_proba[0][1]:.2f}`")


# --- Final Summary Section ---
st.markdown("---")
st.markdown("## 🏁 Summary of Key Insights")
st.markdown(
    """
    <p style='font-family:Arial;'>
    After reviewing the data, we've identified key factors influencing attrition:
    <br>
    <b>Employee Career Stage and Attrition:</b>
    <ul>
        <li>🧑‍💻 Employees at the start of their careers are more likely to switch jobs.</li>
    </ul>

    <b>Compensation and Motivation:</b>
    <ul>
        <li>💰 **Salary and stock options** are powerful motivators. Higher compensation and stock options lead to lower attrition.</li>
    </ul>

    <b>Work-Life Factors:</b>
    <ul>
        <li>🧘‍♀️ **Work-life balance** is critical for retention. Employees with low work-life balance and job satisfaction are a high flight risk.</li>
    </ul>

    <b>Departmental Impact on Attrition:</b>
    <ul>
        <li>🏢 Departments with high-pressure targets, like **Sales**, show higher attrition rates.</li>
    </ul>

    </p>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; margin-top: 2rem;'>
        <i>Dashboard created with Streamlit and Plotly</i>
    </div>
    """, unsafe_allow_html=True
)