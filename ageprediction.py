# Simple NHANES Age Prediction - Streamlit Version
# Easy to understand and explain - Perfect for Resume!
# All 5 Objectives + Classification + Real-time Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="NHANES Age Prediction", layout="wide")
st.title("🏥An Intelligent Healthcare Analytics Dashboard for Age Prediction Using NHANES Data and Machine Learning")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page:", [
    "📈 Main Analysis (5 Objectives)",
    "🎯 Age Group Classification", 
    "🔮 Predict New Patient Age"
])

# ============================================================================
# LOAD DATA
# ============================================================================
try:
    df = pd.read_csv(r"C:\Users\HP\Downloads\national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset\NHANES_age_prediction.csv")
    st.sidebar.success("✅ Data loaded successfully")
except:
    st.sidebar.error("⚠️ Could not load data file")
    st.stop()

# Create age groups for classification
df['age_group'] = df['RIDAGEYR'].apply(lambda x: 'Senior' if x >= 60 else 'Adult')

# ============================================================================
# PAGE 1: MAIN ANALYSIS (ALL 5 OBJECTIVES)
# ============================================================================
if page == "📈 Main Analysis (5 Objectives)":
    
    # OBJECTIVE 1: DATA LOADING
    st.header("📊 OBJECTIVE 1: DATA LOADING AND EXPLORATION")
    st.subheader(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 Rows:**")
        st.dataframe(df.head())
    with col2:
        st.write("**Statistics:**")
        st.dataframe(df.describe())
    
    # OBJECTIVE 2: VISUALIZATION
    st.header("📈 OBJECTIVE 2: DATA VISUALIZATION")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1 = px.histogram(df, x='RIDAGEYR', nbins=30, title='Age Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.scatter(df, x='RIDAGEYR', y='BMXBMI', title='Age vs BMI')
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig3 = px.imshow(corr_matrix, title='Correlation Heatmap', color_continuous_scale='RdBu')
        st.plotly_chart(fig3, use_container_width=True)
    
    # OBJECTIVE 3: DATA PREPARATION
    st.header("⚙️ OBJECTIVE 3: DATA PREPARATION")
    
    feature_columns = [col for col in df.columns if col not in ['RIDAGEYR', 'age_group']]
    X = df[feature_columns].fillna(df[feature_columns].median())
    y = df['RIDAGEYR']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", X_train.shape[0])
    with col2:
        st.metric("Testing Samples", X_test.shape[0])
    with col3:
        st.metric("Features", X_train.shape[1])
    
    st.success("✓ Data prepared: Missing values handled, Train-test split done, Features scaled")
    
    # Store in session for prediction page
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = scaler
        st.session_state['feature_columns'] = feature_columns
    
    # OBJECTIVE 4: TRAIN MODELS
    st.header("🤖 OBJECTIVE 4: TRAINING MACHINE LEARNING MODELS")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, (model_name, model) in enumerate(models.items()):
        progress_bar.progress((idx + 1) / len(models))
        
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results.append({
            'Model': model_name,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R²': test_r2,
            'Predictions': y_pred_test
        })
    
    progress_bar.empty()
    
    # Store models for prediction
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = models
    
    st.success("✓ All 7 models trained successfully")
    
    # OBJECTIVE 5: MODEL COMPARISON
    st.header("📊 OBJECTIVE 5: MODEL COMPARISON AND EVALUATION")
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df[['Model', 'Test RMSE', 'Test MAE', 'Test R²']].round(3), use_container_width=True)
    
    best_idx = results_df['Test R²'].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'Test R²']
    
    st.success(f"🏆 Best Model: **{best_model}** with R² Score of **{best_r2:.3f}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_rmse = px.bar(results_df, x='Model', y='Test RMSE', title='RMSE Comparison', color='Test RMSE')
        fig_rmse.update_xaxes(tickangle=45)
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_r2 = px.bar(results_df, x='Model', y='Test R²', title='R² Score Comparison', color='Test R²')
        fig_r2.update_xaxes(tickangle=45)
        st.plotly_chart(fig_r2, use_container_width=True)
    with col3:
        best_pred = results_df.loc[best_idx, 'Predictions']
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=y_test, y=best_pred, mode='markers', name='Predictions'))
        fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                        y=[y_test.min(), y_test.max()],
                                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
        fig_scatter.update_layout(title=f'Actual vs Predicted - {best_model}',
                                 xaxis_title='Actual Age', yaxis_title='Predicted Age')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.info("✓ All 5 Objectives Completed: Data Loading ✓ Visualization ✓ Preprocessing ✓ Training ✓ Evaluation ✓")

# ============================================================================
# PAGE 2: CLASSIFICATION
# ============================================================================
elif page == "🎯 Age Group Classification":
    st.header("🎯 Age Group Classification (Adult vs Senior)")
    st.write("**Classification Task:** Predict whether a person is Adult (18-59) or Senior (60+)")
    
    # Prepare classification data
    feature_columns = [col for col in df.columns if col not in ['RIDAGEYR', 'age_group']]
    X = df[feature_columns].fillna(df[feature_columns].median())
    
    le = LabelEncoder()
    y_cls = le.fit_transform(df['age_group'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.write(f"**Class Distribution:** Adult: {sum(y_cls==0)}, Senior: {sum(y_cls==1)}")
    
    # Train classification models
    cls_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    if st.button("🚀 Train Classification Models"):
        cls_results = []
        
        for name, model in cls_models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            cls_results.append({
                'Model': name,
                'Accuracy': accuracy
            })
        
        cls_df = pd.DataFrame(cls_results)
        
        st.subheader("📊 Classification Results")
        st.dataframe(cls_df.round(3), use_container_width=True)
        
        best_cls_idx = cls_df['Accuracy'].idxmax()
        best_cls_model = cls_df.loc[best_cls_idx, 'Model']
        best_accuracy = cls_df.loc[best_cls_idx, 'Accuracy']
        
        st.success(f"🏆 Best Model: **{best_cls_model}** with **{best_accuracy:.3f}** accuracy")
        
        # Accuracy comparison chart
        fig_acc = px.bar(cls_df, x='Model', y='Accuracy', 
                        title='Classification Accuracy Comparison',
                        color='Accuracy', color_continuous_scale='Viridis')
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Show classification report for best model
        best_model_obj = cls_models[best_cls_model]
        y_pred_best = best_model_obj.predict(X_test_scaled)
        
        st.subheader(f"📋 Classification Report - {best_cls_model}")
        report = classification_report(y_test, y_pred_best, target_names=le.classes_, output_dict=False)
        st.text(report)

# ============================================================================
# PAGE 3: REAL-TIME PREDICTION
# ============================================================================
else:  # Predict New Patient Age
    st.header("🔮 Predict Age for New Patient")
    st.write("Enter patient information to get age prediction")
    
    # Check if models are trained
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please run the Main Analysis first to train models")
        st.stop()
    
    trained_models = st.session_state['trained_models']
    scaler = st.session_state['scaler']
    feature_columns = st.session_state['feature_columns']
    
    st.subheader("📋 Enter Patient Information")
    
    # Create input fields
    input_data = {}
    col1, col2 = st.columns(2)
    
    # Common features with user-friendly inputs
    with col1:
        if 'RIAGENDR' in feature_columns:
            gender = st.radio("Gender", ["Male", "Female"])
            input_data['RIAGENDR'] = 1 if gender == "Male" else 2
        
        if 'BMXBMI' in feature_columns:
            input_data['BMXBMI'] = st.slider("BMI", 15.0, 40.0, 25.0, 0.5)
        
        if 'LBXGLU' in feature_columns:
            input_data['LBXGLU'] = st.slider("Glucose Level", 50.0, 300.0, 100.0, 5.0)
        
        if 'LBXIN' in feature_columns:
            input_data['LBXIN'] = st.slider("Insulin Level", 1.0, 30.0, 10.0, 0.5)
    
    with col2:
        if 'LBXGLT' in feature_columns:
            input_data['LBXGLT'] = st.slider("Glucose Tolerance", 20.0, 300.0, 100.0, 5.0)
        
        if 'PAQ605' in feature_columns:
            active = st.radio("Physically Active?", ["No", "Yes"])
            input_data['PAQ605'] = 1 if active == "Yes" else 0
        
        if 'DIQ010' in feature_columns:
            diabetes = st.radio("Diabetes?", ["No", "Yes"])
            input_data['DIQ010'] = 1 if diabetes == "Yes" else 0
    
    # Fill remaining features with median values
    for feature in feature_columns:
        if feature not in input_data:
            input_data[feature] = df[feature].median()
    
    # Model selection
    model_choice = st.selectbox("Select Prediction Model:", list(trained_models.keys()))
    
    # Predict button
    if st.button("🎯 Predict Age", type="primary"):
        
        # Prepare input
        input_array = np.array([[input_data[f] for f in feature_columns]])
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        model = trained_models[model_choice]
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.success("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Age", f"{prediction:.1f} years")
        with col2:
            age_group = "Senior (60+)" if prediction >= 60 else "Adult (18-59)"
            st.metric("Age Group", age_group)
        with col3:
            st.metric("Model Used", model_choice)
        
        # Compare predictions from all models
        st.subheader("📈 Predictions from All Models")
        
        all_predictions = []
        for name, mdl in trained_models.items():
            pred = mdl.predict(input_scaled)[0]
            all_predictions.append({
                'Model': name,
                'Predicted Age': pred,
                'Age Group': 'Senior' if pred >= 60 else 'Adult'
            })
        
        pred_df = pd.DataFrame(all_predictions)
        st.dataframe(pred_df, use_container_width=True)
        
        # Visualization
        fig_pred = px.bar(pred_df, x='Model', y='Predicted Age',
                         title='Age Predictions from All Models',
                         color='Age Group',
                         color_discrete_map={'Adult': 'green', 'Senior': 'blue'})
        fig_pred.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pred, use_container_width=True)
        