# Credit Card Fraud Detection - Complete Analysis Pipeline
# 2-Day Academic Project Implementation

"""
🎯 PROJECT OVERVIEW
==================
Problem: Binary classification of credit card transactions (Fraud vs Legitimate)
Dataset: Credit Card Fraud Detection Dataset 2023 (550k+ transactions)
Target: 96-99% accuracy with strong business presentation value
Timeline: 2 days (Day 1: Foundation, Day 2: Advanced & Presentation)

📊 BUSINESS IMPACT
==================
- Credit card fraud costs $28+ billion annually
- Only 0.17% of transactions are fraudulent (severe class imbalance)
- Goal: Minimize false positives while maximizing fraud detection
"""

# =============================================================================
# DAY 1 MORNING: DATA LOADING & EXPLORATORY ANALYSIS
# =============================================================================

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import warnings

warnings.filterwarnings("ignore")

# Configure visualization settings
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print("🚀 Credit Card Fraud Detection Analysis Started!")
print("=" * 60)


# STEP 1: DATA LOADING
# ====================
def load_fraud_dataset(file_path="data/creditcard.csv"):
    """
    Load the credit card fraud dataset with memory optimization

    Dataset Options:
    1. Credit Card Fraud Detection Dataset 2023: 550k+ records
    2. Classic Kaggle Dataset: 284k records

    Download from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
    """
    try:
        print("📂 Loading dataset...")

        # Optimized data loading with proper dtypes
        dtype_dict = {
            "Class": "int8",  # Target variable (0/1)
            "Amount": "float32",  # Transaction amount
        }

        # Add V1-V28 feature columns as float32
        for i in range(1, 29):
            dtype_dict[f"V{i}"] = "float32"

        df = pd.read_csv(file_path, dtype=dtype_dict)

        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape[0]:,} transactions, {df.shape[1]} features")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return df

    except FileNotFoundError:
        print("❌ Dataset file not found!")
        print("📥 Please download the dataset from Kaggle:")
        print(
            "   1. Visit: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
        )
        print("   2. Download and place in data/ folder")
        print(
            "   3. Or use: kaggle datasets download -d nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
        )
        return None


# Load the dataset
df = load_fraud_dataset()

if df is not None:
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # =======================================

    print("\n📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 40)

    # Basic dataset information
    print("\n🔍 Dataset Overview:")
    print(f"• Total transactions: {len(df):,}")
    print(f"• Features: {df.shape[1]}")
    print(f"• Missing values: {df.isnull().sum().sum()}")
    print(f"• Duplicates: {df.duplicated().sum()}")

    # Class distribution analysis
    fraud_count = df["Class"].value_counts()
    fraud_rate = df["Class"].mean()

    print(f"\n💳 Transaction Classification:")
    print(f"• Legitimate transactions: {fraud_count[0]:,} ({(1-fraud_rate)*100:.3f}%)")
    print(f"• Fraudulent transactions: {fraud_count[1]:,} ({fraud_rate*100:.3f}%)")
    print(f"• Fraud rate: {fraud_rate:.5f} ({fraud_rate*100:.4f}%)")

    # Key insights for presentation
    print(f"\n🎯 KEY BUSINESS INSIGHTS:")
    print(f"• Dataset represents {len(df):,} real-world transactions")
    print(f"• Severe class imbalance: Only {fraud_rate*100:.3f}% are fraudulent")
    print(
        f"• Challenge: Detect {fraud_count[1]:,} fraud cases among {len(df):,} transactions"
    )

    # Statistical summary
    print("\n📈 Amount Statistics:")
    amount_stats = df.groupby("Class")["Amount"].describe()
    print(amount_stats)

    # =============================================================================
    # VISUALIZATION: EDA PLOTS FOR PRESENTATION
    # =============================================================================

    # Create comprehensive EDA plots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Class Distribution",
            "Transaction Amount Distribution",
            "Amount by Class (Box Plot)",
            "Feature Correlation Heatmap",
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "box"}, {"type": "heatmap"}],
        ],
    )

    # Plot 1: Class distribution
    fraud_counts = df["Class"].value_counts()
    fig.add_trace(
        go.Bar(
            x=["Legitimate", "Fraud"],
            y=fraud_counts.values,
            text=[f"{count:,}" for count in fraud_counts.values],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Amount distribution
    fig.add_trace(
        go.Histogram(x=df["Amount"], nbinsx=50, name="Amount Distribution"),
        row=1,
        col=2,
    )

    # Plot 3: Amount by class
    for class_val in [0, 1]:
        class_name = "Legitimate" if class_val == 0 else "Fraud"
        fig.add_trace(
            go.Box(y=df[df["Class"] == class_val]["Amount"], name=class_name),
            row=2,
            col=1,
        )

    # Plot 4: Correlation heatmap (sample of features)
    correlation_features = ["Amount"] + [f"V{i}" for i in range(1, 11)] + ["Class"]
    corr_matrix = df[correlation_features].corr()

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, title_text="Credit Card Fraud Detection - Exploratory Data Analysis"
    )
    fig.show()

    # =============================================================================
    # DAY 1 AFTERNOON: DATA PREPROCESSING & BASELINE MODELS
    # =============================================================================

    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # type: ignore
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore

    print("\n🔧 DATA PREPROCESSING")
    print("=" * 30)

    # Prepare features and target
    if "id" in df.columns:
        X = df.drop(["id", "Class"], axis=1)
    else:
        X = df.drop(["Class"], axis=1)
    y = df["Class"]

    print(f"• Features shape: {X.shape}")
    print(f"• Target distribution: {y.value_counts().to_dict()}")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"• Training set: {X_train.shape[0]:,} samples")
    print(f"• Test set: {X_test.shape[0]:,} samples")
    print(f"• Train fraud rate: {y_train.mean():.4f}")
    print(f"• Test fraud rate: {y_test.mean():.4f}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Data preprocessing completed!")

    # =============================================================================
    # BASELINE MODELS (DAY 1 AFTERNOON)
    # =============================================================================

    print("\n🤖 BASELINE MODEL TRAINING")
    print("=" * 35)

    # Model evaluation function
    def evaluate_model(model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = (
            roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
        )

        print(f"\n📊 {model_name} Results:")
        print(f"• Accuracy:  {accuracy:.4f}")
        print(f"• Precision: {precision:.4f}")
        print(f"• Recall:    {recall:.4f}")
        print(f"• F1-Score:  {f1:.4f}")
        print(f"• ROC-AUC:   {roc_auc}")

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    # Store results for comparison
    model_results = []

    # 1. Logistic Regression Baseline
    print("\n🔥 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    model_results.append(lr_results)

    # 2. Random Forest Baseline
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    model_results.append(rf_results)

    print("\n✅ Day 1 Baseline Models Completed!")
    print("🎯 Next: Day 2 will add XGBoost, SVM, and SMOTE balancing")

# =============================================================================
# DAY 2 MORNING: ADVANCED MODELS & CLASS IMBALANCE HANDLING
# =============================================================================

if df is not None:
    print("\n" + "=" * 60)
    print("🚀 DAY 2: ADVANCED MODELS & OPTIMIZATION")
    print("=" * 60)

    from xgboost import XGBClassifier  # type: ignore
    from sklearn.svm import SVC  # type: ignore
    from sklearn.model_selection import cross_val_score  # type: ignore

    # Handle class imbalance with SMOTE
    print("\n⚖️ HANDLING CLASS IMBALANCE")
    print("=" * 35)

    print("Applying SMOTE (Synthetic Minority Oversampling Technique)...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"• Original training distribution: {y_train.value_counts().to_dict()}")
    print(
        f"• Balanced training distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}"
    )
    print(
        f"• Training set size increased: {len(X_train_scaled):,} → {len(X_train_balanced):,}"
    )

    # 3. XGBoost with balanced data
    print("\n🚀 Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)
    xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    model_results.append(xgb_results)

    # 4. Support Vector Machine
    # print("\n🎯 Training SVM...")
    # svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    # svm_model.fit(X_train_balanced, y_train_balanced)
    # svm_results = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
    # model_results.append(svm_results)

    # =============================================================================
    # MODEL COMPARISON & ANALYSIS
    # =============================================================================

    print("\n📊 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 45)

    # Create results DataFrame
    results_df = pd.DataFrame(model_results)
    print("\n🏆 Final Model Performance:")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # Find best model
    best_model_idx = results_df["f1"].idxmax()
    best_model = results_df.iloc[best_model_idx]

    print(f"\n🥇 BEST PERFORMING MODEL: {best_model['model']}")
    print(f"• F1-Score: {best_model['f1']:.4f}")
    print(f"• Accuracy: {best_model['accuracy']:.4f}")
    print(f"• Precision: {best_model['precision']:.4f}")
    print(f"• Recall: {best_model['recall']:.4f}")

    # Feature importance analysis (for tree-based models)
    if hasattr(rf_model, "feature_importances_"):
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)

        feature_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_rf": rf_model.feature_importances_,
                "importance_xgb": xgb_model.feature_importances_,
            }
        ).sort_values("importance_rf", ascending=False)

        print("Top 10 Most Important Features (Random Forest):")
        print(
            feature_importance.head(10)[["feature", "importance_rf"]].to_string(
                index=False
            )
        )

    # =============================================================================
    # BUSINESS INSIGHTS & RECOMMENDATIONS
    # =============================================================================

    print("\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)

    # Calculate business impact
    total_transactions = len(y_test)
    actual_frauds = y_test.sum()

    # Use best model predictions
    if best_model["model"] == "XGBoost":
        best_predictions = xgb_model.predict(X_test_scaled)
    elif best_model["model"] == "Random Forest":
        best_predictions = rf_model.predict(X_test_scaled)
    # elif best_model["model"] == "SVM":
    #     best_predictions = svm_model.predict(X_test_scaled)
    else:
        best_predictions = lr_model.predict(X_test_scaled)

    detected_frauds = ((best_predictions == 1) & (y_test == 1)).sum()
    false_positives = ((best_predictions == 1) & (y_test == 0)).sum()

    print(f"📈 FRAUD DETECTION PERFORMANCE:")
    print(f"• Total test transactions: {total_transactions:,}")
    print(f"• Actual fraud cases: {actual_frauds:,}")
    print(f"• Fraud cases detected: {detected_frauds:,}")
    print(f"• Detection rate: {detected_frauds/actual_frauds*100:.1f}%")
    print(f"• False positives: {false_positives:,}")
    print(
        f"• False positive rate: {false_positives/(total_transactions-actual_frauds)*100:.3f}%"
    )

    # Business value calculation
    avg_fraud_amount = df[df["Class"] == 1]["Amount"].mean()
    avg_legit_amount = df[df["Class"] == 0]["Amount"].mean()

    fraud_prevented = detected_frauds * avg_fraud_amount
    investigation_cost = (
        false_positives * 25
    )  # Assume $25 per false positive investigation

    print(f"\n💰 ESTIMATED BUSINESS IMPACT:")
    print(f"• Average fraud amount: ${avg_fraud_amount:.2f}")
    print(f"• Fraud prevented: ${fraud_prevented:,.2f}")
    print(f"• Investigation costs: ${investigation_cost:,.2f}")
    print(f"• Net savings: ${fraud_prevented - investigation_cost:,.2f}")

    print(f"\n🎯 KEY RECOMMENDATIONS:")
    print(f"• Deploy {best_model['model']} model for real-time scoring")
    print(f"• Set alert threshold to balance detection vs false positives")
    print(f"• Monitor model performance and retrain monthly")
    print(f"• Focus on top {len(feature_importance.head(10))} features for efficiency")

    print("\n✅ ANALYSIS COMPLETE!")
    print("🚀 Ready for presentation and dashboard deployment!")

else:
    print("⚠️ Please download the dataset to continue with the analysis.")
    print("📥 Dataset download instructions:")
    print("1. Visit Kaggle and download the Credit Card Fraud Detection Dataset 2023")
    print("2. Place the CSV file in the data/ folder")
    print("3. Rerun this notebook")

# Save results for dashboard
if "model_results" in locals():
    import joblib  # type: ignore

    # Create outputs directory
    import os

    os.makedirs("outputs", exist_ok=True)

    # Save model results
    joblib.dump(model_results, "outputs/model_results.pkl")
    joblib.dump(results_df, "outputs/results_dataframe.pkl")

    if "feature_importance" in locals():
        feature_importance.to_csv("outputs/feature_importance.csv", index=False)

    print("\n💾 Results saved to outputs/ folder for dashboard use!")


# Create models directory
os.makedirs("outputs/models", exist_ok=True)

# Save all trained models
print("💾 Saving trained models...")

# Save the models
joblib.dump(lr_model, "outputs/models/logistic_regression_model.pkl")
joblib.dump(rf_model, "outputs/models/random_forest_model.pkl")
joblib.dump(xgb_model, "outputs/models/xgboost_model.pkl")

# Save the scaler (important for predictions)
joblib.dump(scaler, "outputs/models/feature_scaler.pkl")

# Save the best model specifically
joblib.dump(rf_model, "outputs/models/best_model.pkl")

print("✅ Models saved successfully!")
print("\n📁 Saved models:")
print("• outputs/models/logistic_regression_model.pkl")
print("• outputs/models/random_forest_model.pkl")
print("• outputs/models/xgboost_model.pkl")
print("• outputs/models/feature_scaler.pkl")
print("• outputs/models/best_model.pkl (Random Forest)")

# Verify file sizes
for file in os.listdir("outputs/models/"):
    file_path = f"outputs/models/{file}"
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"• {file}: {size_mb:.1f} MB")
