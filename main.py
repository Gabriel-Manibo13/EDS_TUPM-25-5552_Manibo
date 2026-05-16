import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning Core Frameworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# =========================================================================
# ⚙️ REGULATED REPOSITORY PATH CONFIGURATION
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Automatically generate required directory structure safely
for folder in [DATA_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)


# =========================================================================
# 🧹 FUNCTION 1: DATA INGESTION CORE
# =========================================================================
def load_raw_dataset():
    target_path = os.path.join(DATA_DIR, "dataset_original.csv")
    try:
        print(f"📥 Loading source data from: {target_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Source missing. Place 'dataset_original.csv' in data/")
        df = pd.read_csv(target_path)
        print(f"📊 Raw Matrix Dimensions Ingested: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ CRITICAL INGESTION BREAKDOWN: {e}")
        return None


# =========================================================================
# 🧹 FUNCTION 2: PIPELINE DATA CLEANING MACHINE
# =========================================================================
def execute_pipeline_cleaning(df):
    try:
        print("\n🧹 Phase 1: Initiating Data Cleaning Framework...")
        df.columns = df.columns.str.lower()
        
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"⚠️ {null_counts} null anomalies localized. Executing data imputation...")
            df = df.fillna(df.median(numeric_only=True))
        else:
            print("✅ Verification Complete: Zero missing/null instances found.")
            
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️ Dropping {duplicates} duplicate entries...")
            df = df.drop_duplicates()
        else:
            print("✅ Data Uniqueness Verified: Zero duplicate indices discovered.")
            
        df["label"] = df["label"].astype(str).str.lower()
        
        cleaned_path = os.path.join(DATA_DIR, "dataset_cleaned.csv")
        df.to_csv(cleaned_path, index=False)
        print(f"💾 Cleaned dataset committed to disk layout: {cleaned_path}")
        return df
    except Exception as e:
        print(f"❌ DATA PIPELINE CRASH CONTAINED: {e}")
        return None


# =========================================================================
# 📊 FUNCTION 3: MATH DISTRIBUTION ANALYTICS ENGINE
# =========================================================================
def compute_engineering_analytics(df):
    try:
        print("\n📊 Phase 2: Evaluating Distribution Analytics & Mathematical Matrix Shapes...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe().T
        stats["variance"] = df[numeric_cols].var()
        stats["skewness"] = df[numeric_cols].skew()
        
        report_df = stats[["mean", "50%", "std", "variance", "skewness"]].rename(columns={"50%": "median"})
        
        print("\n" + "="*70)
        print("📊 --- DESCRIPTIVE STATISTICS & DISTRIBUTION SHAPES (NUMPY/PANDAS) ---")
        print("="*70)
        print(report_df.round(3))
        print("="*70)
        
        report_df.to_csv(os.path.join(OUT_DIR, "data_profile_summary.csv"), index=True)
        return report_df
    except Exception as e:
        print(f"❌ ANALYTICS ENGINE BREAKDOWN: {e}")
        return None


# =========================================================================
# 🧪 FUNCTION 4: STOICHIOMETRIC AGRONOMIC ANALYSIS
# =========================================================================
def evaluate_agronomic_synergy(df):
    try:
        print("\n🧪 Phase 3: Mapping Nutrient Proportions (Macronutrient Synergy)...")
        df['n_p_ratio'] = df['n'] / (df['p'] + 1e-5)
        df['n_k_ratio'] = df['n'] / (df['k'] + 1e-5)
        
        synergy_matrix = df.groupby('label')[['n_p_ratio', 'n_k_ratio', 'ph', 'rainfall']].mean().round(2)
        synergy_matrix.columns = ['Mean_N_P_Ratio', 'Mean_N_K_Ratio', 'Mean_Soil_pH', 'Mean_Rainfall_mm']
        
        print("\n" + "="*75)
        print("🧪 --- ADVANCED MACRONUTRIENT SYNERGY RATIOS ---")
        print("="*75)
        print(synergy_matrix.to_string())
        print("="*75)
        
        synergy_matrix.to_csv(os.path.join(OUT_DIR, "npk_synergy_matrix.csv"), index=True)
        return synergy_matrix
    except Exception as e:
        print(f"❌ STOICHIOMETRIC MATH INTERPRETATION CRASHED: {e}")
        return None


# =========================================================================
# 🎨 FUNCTION 5: HIGH-DPI ACADEMIC VISUALIZATIONS
# =========================================================================
def generate_static_graphics(df):
    try:
        print("\n🎨 Phase 4: Constructing Matplotlib/Seaborn Academic Static Artifacts...")
        sns.set_theme(style="whitegrid")
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['n_p_ratio', 'n_k_ratio']]
        
        # 1. Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Soil/Climate Features", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=300)
        plt.close()
        
        # 2. Nutrient Outliers
        plt.figure(figsize=(10, 6))
        melted = df[['n', 'p', 'k']].melt(var_name="Nutrient", value_name="Value")
        sns.boxplot(x="Nutrient", y="Value", data=melted, palette="Set2", hue="Nutrient", legend=False)
        plt.title("Distribution Boundaries and Outliers: Macronutrients (N, P, K)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "nutrient_outliers.png"), dpi=300)
        plt.close()
        
        # 3. Comparative Bar Chart
        plt.figure(figsize=(10, 6))
        comp = df[df["label"].isin(["rice", "coffee"])]
        sns.barplot(x="label", y="rainfall", data=comp, estimator=np.mean, errorbar="sd", palette="pastel", hue="label", legend=False)
        plt.title("Comparative Analysis: Average Rainfall Requirement (Rice vs Coffee)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "comparative_rainfall.png"), dpi=300)
        plt.close()
        
        print("✅ Core visual assets saved at 300 DPI.")
    except Exception as e:
        print(f"❌ PLOTTER EXCEPTION: {e}")


# =========================================================================
# 🎬 FUNCTION 6: INTERACTIVE WEB MATRIX SIMULATIONS
# =========================================================================
def generate_interactive_animations(df):
    try:
        print("\n🎬 Phase 5: Rendering Plotly Dynamic Sandbox Visuals & Animations...")
        fig_scatter = px.scatter(
            df, x="temperature", y="humidity", animation_frame="label",
            size="rainfall", color="ph", hover_name="label", size_max=40,
            title="Dynamic Climate Boundaries: Temperature vs Humidity Over Crop Selections"
        )
        fig_scatter.write_html(os.path.join(OUT_DIR, "climate_dynamics_animation.html"))
        
        comp_df = df[df["label"].isin(["rice", "maize", "chickpea", "coffee"])]
        fig_density = px.histogram(
            comp_df, x="rainfall", color="label", marginal="box",
            animation_frame="label", title="Simulated Rainfall Operational Density Profile Shift Matrix",
            opacity=0.75
        )
        fig_density.write_html(os.path.join(OUT_DIR, "comparative_rainfall_density.html"))
        print("🎬 Sandbox HTML frameworks written to outputs folder.")
    except Exception as e:
        print(f"❌ INTERACTIVE ANIMATION BLOCK FAILURE: {e}")


# =========================================================================
# 🤖 FUNCTION 7: ML CORE ENSEMBLE TRAINING
# =========================================================================
def train_predictive_model(df):
    try:
        print("\n🤖 Phase 6: Initializing Machine Learning Predictive Core...")
        X = df[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("🌲 Fitting ensemble random forest matrix architecture (n_estimators=100)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    except Exception as e:
        print(f"❌ MODEL TRAINING EXCEPTION: {e}")
        return None, None, None


# =========================================================================
# 📈 FUNCTION 8: MODEL PERFORMANCE MATRIX AUDITOR
# =========================================================================
def evaluate_model_performance(model, X_test, y_test):
    try:
        print("\n📈 Phase 7: Executing Predictive Model Performance Audit...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Verified Model Classification Accuracy: {accuracy * 100:.2f}%")
        
        report = classification_report(y_test, y_pred)
        with open(os.path.join(OUT_DIR, "model_performance_report.txt"), "w") as f:
            f.write(report)
            
        labels_order = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt="d", cbar=True)
        plt.title("Crop Recommendation Engine: Confusion Matrix Verification", fontsize=14, pad=15)
        plt.ylabel('Actual Crop Varieties')
        plt.xlabel('Model Predicted Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "model_confusion_matrix.png"), dpi=300)
        plt.close()
        print("💾 Performance summaries and validation matrix graphics saved.")
    except Exception as e:
        print(f"❌ PERFORMANCE AUDIT EVALUATION CRASHED: {e}")


# =========================================================================
# 🔍 FUNCTION 9: STATISTICAL FEATURE IMPORTANCE TRACKER
# =========================================================================
def extract_feature_importance(model, feature_names):
    try:
        print("\n🔍 Phase 8: Extracting Feature Importance Weight Ratios...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feat_df = pd.DataFrame({
            'Feature': [feature_names[i].upper() for i in indices],
            'Importance Score': [importances[i] for i in indices]
        })
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance Score', y='Feature', data=feat_df, palette='viridis', hue='Feature', legend=False)
        plt.title("Statistical Feature Importance: Dominant Crop Recommendation Drivers", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "feature_importances.png"), dpi=300)
        plt.close()
        
        joblib.dump(model, os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"))
        print("🔒 Saved relative engineering weights map and serialized frozen model binary core.")
    except Exception as e:
        print(f"❌ FEATURE IMPORTANCE TRACKING FAILED: {e}")


# =========================================================================
# 🏁 PIPELINE ORCHESTRATION MAIN CONTROL ENTRY
# =========================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ECOSYSTEM RECONSTRUCTION INTERACTION PIPELINE | FINAL SUBMISSION CORE")
    print("🚀 STUDENT IDENTIFICATION: TUPM-25-5552")
    print("=" * 80)
    
    raw_data = load_raw_dataset()
    if raw_data is not None:
        cleaned_data = execute_pipeline_cleaning(raw_data)
        if cleaned_data is not None:
            compute_engineering_analytics(cleaned_data)
            evaluate_agronomic_synergy(cleaned_data)
            generate_static_graphics(cleaned_data)
            generate_interactive_animations(cleaned_data)
            
            # Machine Learning Subsystem
            model_core, test_X, test_y = train_predictive_model(cleaned_data)
            if model_core is not None:
                evaluate_model_performance(model_core, test_X, test_y)
                extract_feature_importance(model_core, ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall'])
                
            print("\n" + "=" * 80)
            print("🟢 SUCCESS: Reconstructed data engineering loop complete! Ready for evaluation.")
            print("=" * 80 + "\n")