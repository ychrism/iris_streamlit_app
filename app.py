import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

def load_and_prepare_iris_data():
    """
    Load and preprocess the Iris dataset with additional metadata
    
    Returns:
    - Preprocessed DataFrame
    - Original feature names
    - Target names
    """
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    
    # Add species column
    df['species'] = df['target'].map({
        0: 'Setosa', 
        1: 'Versicolor', 
        2: 'Virginica'
    })
    
    return df

def create_advanced_visualizations(df):
    """
    Create multiple advanced visualization techniques
    """
    st.header("🔍 Advanced Data Exploration")
    
    # Select visualization type
    viz_type = st.selectbox(
        "Choose Visualization Technique", 
        [
            "Pair Plot", 
            "Violin Plot", 
            "Box Plot", 
            "Parallel Coordinates", 
            "PCA Visualization"
        ]
    )
    
    # Visualization logic
    plt.figure(figsize=(12, 8))
    
    if viz_type == "Pair Plot":
        fig = sns.pairplot(
            df, 
            hue='species', 
            plot_kws={'alpha': 0.6},
            diag_kws={'alpha': 0.6}
        )
        st.pyplot(fig)
        st.markdown("**Pair Plot Analysis**: Shows relationships between all feature pairs")
    
    elif viz_type == "Violin Plot":
        feature_cols = df.columns[:-2]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(feature_cols):
            sns.violinplot(
                x='species', 
                y=feature, 
                data=df, 
                ax=axes[i]
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Violin Plot**: Distribution of features across species")
    
    elif viz_type == "Box Plot":
        feature_cols = df.columns[:-2]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(feature_cols):
            sns.boxplot(
                x='species', 
                y=feature, 
                data=df, 
                ax=axes[i]
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Box Plot**: Feature distributions with quartiles")
    
    elif viz_type == "Parallel Coordinates":
        from pandas.plotting import parallel_coordinates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(
            df.drop('target', axis=1), 
            'species', 
            colormap=plt.cm.Set2
        )
        plt.title('Parallel Coordinates Plot')
        st.pyplot(fig)
        st.markdown("**Parallel Coordinates**: Visualize multivariate data")
    
    elif viz_type == "PCA Visualization":
        # PCA Transformation
        features = df.drop(['target', 'species'], axis=1)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        pca_df = pd.DataFrame(
            data=pca_result, 
            columns=['PC1', 'PC2']
        )
        pca_df['species'] = df['species']
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=pca_df, 
            x='PC1', 
            y='PC2', 
            hue='species', 
            palette='deep'
        )
        plt.title('PCA: Dimensionality Reduction')
        st.pyplot(plt.gcf())
        
        # Variance explanation
        variance_ratio = pca.explained_variance_ratio_
        st.markdown(f"""
        **PCA Variance Explanation**:
        - First Principal Component: {variance_ratio[0]*100:.2f}%
        - Second Principal Component: {variance_ratio[1]*100:.2f}%
        """)

def statistical_summary(df):
    """
    Provide detailed statistical summary
    """
    st.header("📊 Statistical Deep Dive")
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics by Species")
    desc_stats = df.groupby('species').describe()
    st.dataframe(desc_stats)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    feature_cols = df.columns[:-2]
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5
    )
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt.gcf())

def main():
    st.set_page_config(
        page_title="Iris Dataset Explorer", 
        page_icon="🌸", 
        layout="wide"
    )
    
    st.title("🌺 Comprehensive Iris Dataset Analysis")
    
    # Load data
    df = load_and_prepare_iris_data()
    
    # Sidebar navigation
    analysis_option = st.sidebar.radio(
        "Choose Analysis Type", 
        [
            "Data Overview", 
            "Data Visualizations", 
            "Statistical Analysis", 
            "Machine Learning Predictions"
        ]
    )
    
    if analysis_option == "Data Overview":
        st.header("Dataset Overview")
        st.dataframe(df)
        st.write(f"Total Samples: {len(df)}")
        st.write("Species Distribution:")
        st.dataframe(df['species'].value_counts())
    
    elif analysis_option == "Data Visualizations":
        create_advanced_visualizations(df)
    
    elif analysis_option == "Statistical Analysis":
        statistical_summary(df)
    
    elif analysis_option == "Machine Learning Predictions":
        st.header("🤖 Predictive Modeling")
        
        # Prepare data for modeling
        X = df.drop(['target', 'species'], axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        )
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_classifier.predict(X_test_scaled)
        
        # Display results
        st.subheader("Model Performance")
        st.text(classification_report(
            y_test, y_pred, 
            target_names=['Setosa', 'Versicolor', 'Virginica']
        ))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.subheader("Feature Importance")
        st.dataframe(feature_importance)

if __name__ == "__main__":
    main()