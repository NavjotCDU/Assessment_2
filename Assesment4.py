"""
PRT564 - DATA ANALYTICS AND VISUALISATION
Assessment 4: Group Project Report - Python Code
Group SYD G13 | Charles Darwin University | 2026

Dataset: Labour Force, Australia - January 2026
Source:  https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/jan-2026
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve)
from sklearn.feature_selection import SelectFromModel
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# 1. DATA COLLECTION & INTEGRATION
#    (Reproducing ABS Labour Force Table 1 – Jan 2016 to Jan 2026)
# =============================================================================
print("=" * 65)
print("SECTION 1: DATA COLLECTION & INTEGRATION")
print("=" * 65)

# --- 1a. Build the ABS unemployment dataset inline ---
# Monthly data: Date | Trend (%) | Seasonally Adjusted (%)
# Source: ABS Labour Force Australia, Jan 2026 release
abs_data = {
    "Date": pd.date_range(start="2016-01-01", periods=121, freq="MS"),
    "Trend_pct": [
        5.8,5.7,5.7,5.7,5.7,5.7,5.8,5.7,5.7,5.7,5.7,5.7,  # 2016
        5.7,5.7,5.7,5.6,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.4,  # 2017
        5.5,5.5,5.5,5.6,5.6,5.5,5.4,5.3,5.3,5.2,5.2,5.1,  # 2018
        5.1,5.1,5.1,5.2,5.2,5.2,5.2,5.3,5.3,5.3,5.3,5.2,  # 2019
        5.2,5.3,5.8,6.1,6.9,7.1,7.4,7.5,7.1,6.8,6.8,6.6,  # 2020
        6.6,6.3,6.1,5.9,5.6,5.4,5.2,5.1,5.0,5.0,4.9,4.9,  # 2021
        4.5,4.2,4.0,3.9,3.9,3.8,3.6,3.5,3.5,3.4,3.4,3.4,  # 2022
        3.5,3.5,3.6,3.7,3.7,3.7,3.8,3.8,3.9,3.9,3.9,3.9,  # 2023
        4.0,4.0,3.9,3.9,4.0,4.0,4.1,4.1,4.2,4.1,4.1,4.1,  # 2024
        4.0,4.0,4.0,4.1,4.0,4.0,4.1,4.0,4.0,4.0,4.0,3.9,  # 2025
        4.0,                                                 # Jan 2026
    ],
    "Seas_Adj_pct": [
        6.0,5.7,5.7,5.6,5.7,5.8,5.8,5.8,5.5,5.6,5.7,5.7,
        5.7,5.9,5.8,5.7,5.6,5.5,5.6,5.5,5.3,5.4,5.4,5.5,
        5.5,5.6,5.6,5.6,5.6,5.4,5.3,5.2,5.2,5.2,5.1,5.1,
        5.0,5.1,5.1,5.2,5.2,5.2,5.2,5.3,5.2,5.3,5.2,5.1,
        5.1,5.3,5.9,7.1,7.5,7.5,7.5,6.8,6.9,6.9,6.8,6.6,
        6.4,5.9,5.8,5.7,5.1,4.9,4.9,4.7,4.6,5.1,4.8,4.9,
        4.2,4.0,3.9,3.9,3.9,3.5,3.5,3.5,3.5,3.4,3.4,3.4,
        3.5,3.6,3.5,3.7,3.7,3.6,3.8,3.7,3.8,3.9,4.0,3.9,
        4.1,3.8,3.8,3.9,4.0,3.9,4.1,4.2,4.2,4.1,4.1,4.2,
        4.0,4.2,4.0,4.1,4.1,3.9,4.0,3.8,3.9,3.9,3.9,3.8,
        4.1,
    ]
}

df = pd.DataFrame(abs_data)
print(f"ABS dataset loaded: {df.shape[0]} observations × {df.shape[1]} variables")
print(df.head())

# --- 1b. Secondary dataset – ABS GDP growth (quarterly, % change) ---
# Integrated to demonstrate heterogeneous pipeline (from ABS National Accounts)
gdp_quarters = pd.date_range(start="2016-01-01", periods=41, freq="QS")
gdp_values = [
    0.9,0.4,0.6,1.1, 0.5,0.8,0.7,1.0, 0.6,0.6,0.7,0.7,
    0.7,0.7,0.6,0.5, -1.0,-7.0,3.4,3.1, 1.9,0.8,0.8,0.7,
    0.8,0.6,0.5,0.5, 0.4,0.2,0.2,0.4, 0.2,0.3,0.4,0.5,
    0.4,0.3,0.5,0.4, 0.3
]
gdp_df = pd.DataFrame({"Date": gdp_quarters, "GDP_growth_pct": gdp_values})

# Merge: forward-fill quarterly GDP to monthly frequency
df["Quarter"] = df["Date"].dt.to_period("Q").dt.start_time
df = df.merge(
    gdp_df.rename(columns={"Date": "Quarter"}),
    on="Quarter", how="left"
)
df["GDP_growth_pct"] = df.groupby("Quarter")["GDP_growth_pct"].transform("first")
df.drop(columns=["Quarter"], inplace=True)

print(f"\nAfter GDP integration: {df.shape}")
print(df[["Date","Trend_pct","Seas_Adj_pct","GDP_growth_pct"]].tail())

# 2. DATA PREPROCESSING & FEATURE ENGINEERING

print("\n" + "=" * 65)
print("SECTION 2: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 65)

# 2a. Missing value check
print("\nMissing values:\n", df.isnull().sum())

# 2b. Date-derived features
df["Month_Index"]   = (df["Date"].dt.year - 2016) * 12 + df["Date"].dt.month - 1
df["Year"]          = df["Date"].dt.year
df["Month"]         = df["Date"].dt.month
df["Quarter_num"]   = df["Date"].dt.quarter

# 2c. Rolling / lag features (signal temporal context to classifiers)
df["Seas_lag1"]     = df["Seas_Adj_pct"].shift(1)
df["Seas_lag3"]     = df["Seas_Adj_pct"].shift(3)
df["Rolling_mean3"] = df["Seas_Adj_pct"].rolling(3).mean()
df["Rolling_std3"]  = df["Seas_Adj_pct"].rolling(3).std()
df["MoM_change"]    = df["Seas_Adj_pct"].diff()          # month-on-month change

# Drop rows with NaN from lag/rolling (first 3 rows)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 2d. Target label – unemployment category
def categorise(val):
    if val < 5.0:
        return "Low"
    elif val <= 6.0:
        return "Medium"
    else:
        return "High"

df["Category"] = df["Seas_Adj_pct"].apply(categorise)
print("\nClass distribution:")
print(df["Category"].value_counts())

# 2e. Encode target
le = LabelEncoder()
df["Category_enc"] = le.fit_transform(df["Category"])  # High=0, Low=1, Medium=2
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# 2f. Feature set & scaling
FEATURES = [
    "Month_Index","Year","Month","Quarter_num",
    "Trend_pct","GDP_growth_pct",
    "Seas_lag1","Seas_lag3","Rolling_mean3","Rolling_std3","MoM_change"
]
X = df[FEATURES].copy()
y = df["Category_enc"].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

print(f"\nFinal feature matrix: {X_scaled.shape}")
print("Feature summary:\n", X_scaled.describe().round(3))

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EDA – ABS Labour Force Australia (2016–2026)", fontsize=15, fontweight="bold")

# Plot 1: Unemployment trend over time
ax = axes[0, 0]
ax.plot(df["Date"], df["Trend_pct"],      label="Trend",       color="#1f77b4", linewidth=2)
ax.plot(df["Date"], df["Seas_Adj_pct"],   label="Seas. Adj.",  color="#ff7f0e", linewidth=1.5, linestyle="--")
ax.set_title("Unemployment Rate Over Time")
ax.set_ylabel("Rate (%)")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(6))

# Plot 2: Category distribution
ax = axes[0, 1]
cat_counts = df["Category"].value_counts().reindex(["Low","Medium","High"])
colors = ["#2ca02c","#ff7f0e","#d62728"]
ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="white", linewidth=1.2)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(cat_counts.values):
    ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# Plot 3: Correlation heatmap
ax = axes[0, 2]
corr_cols = ["Seas_Adj_pct","Trend_pct","GDP_growth_pct","MoM_change","Rolling_mean3"]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, vmin=-1, vmax=1, linewidths=0.5)
ax.set_title("Correlation Heatmap")

# Plot 4: Unemployment by category (boxplot)
ax = axes[1, 0]
order = ["Low","Medium","High"]
palette = {"Low":"#2ca02c","Medium":"#ff7f0e","High":"#d62728"}
sns.boxplot(data=df, x="Category", y="Seas_Adj_pct", order=order, palette=palette, ax=ax)
ax.set_title("Seas. Adj. Rate by Category")
ax.set_ylabel("Rate (%)")

# Plot 5: GDP growth vs Unemployment
ax = axes[1, 1]
sc = ax.scatter(df["GDP_growth_pct"], df["Seas_Adj_pct"],
                c=df["Category_enc"], cmap="RdYlGn_r", alpha=0.7, s=50)
ax.set_title("GDP Growth vs Unemployment")
ax.set_xlabel("GDP Growth (%)")
ax.set_ylabel("Seas. Adj. Rate (%)")
plt.colorbar(sc, ax=ax, label="Category (0=High, 1=Low, 2=Med)")

# Plot 6: Rolling mean & std
ax = axes[1, 2]
ax.plot(df["Date"], df["Rolling_mean3"],  color="#9467bd", linewidth=2,  label="3m Rolling Mean")
ax.fill_between(df["Date"],
                df["Rolling_mean3"] - df["Rolling_std3"],
                df["Rolling_mean3"] + df["Rolling_std3"],
                alpha=0.2, color="#9467bd", label="±1 Std Dev")
ax.set_title("3-Month Rolling Statistics")
ax.set_ylabel("Rate (%)")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(6))

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("EDA plots saved → eda_plots.png")

# =============================================================================
# 4. CLASSIFICATION MODELS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4: CLASSIFICATION MODELS")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------------------------------------------------------
# MODEL A: Logistic Regression
# --------------------------------------------------------------------------
print("\n--- MODEL A: Logistic Regression ---")
lr_params = {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs"]}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params, cv=cv, scoring="f1_macro", n_jobs=-1
)
lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_
print(f"Best params: {lr_grid.best_params_}")
lr_pred = lr_best.predict(X_test)
lr_prob = lr_best.predict_proba(X_test)

lr_acc  = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average="macro", zero_division=0)
lr_rec  = recall_score(y_test, lr_pred, average="macro", zero_division=0)
lr_f1   = f1_score(y_test, lr_pred, average="macro", zero_division=0)
lr_roc  = roc_auc_score(y_test, lr_prob, multi_class="ovr", average="macro")

print(f"  Accuracy : {lr_acc:.4f}")
print(f"  Precision: {lr_prec:.4f}")
print(f"  Recall   : {lr_rec:.4f}")
print(f"  F1-macro : {lr_f1:.4f}")
print(f"  ROC-AUC  : {lr_roc:.4f}")

lr_cv_f1 = cross_val_score(lr_best, X_scaled, y, cv=cv, scoring="f1_macro")
print(f"  CV F1 (5-fold): {lr_cv_f1.mean():.4f} ± {lr_cv_f1.std():.4f}")

# --------------------------------------------------------------------------
# MODEL B: Decision Tree (with hyperparameter tuning)
# --------------------------------------------------------------------------
print("\n--- MODEL B: Decision Tree Classifier ---")
dt_params = {
    "max_depth":        [3, 5, 7, 10, None],
    "min_samples_split":[2, 5, 10],
    "criterion":        ["gini","entropy"]
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=cv, scoring="f1_macro", n_jobs=-1
)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
print(f"Best params: {dt_grid.best_params_}")
dt_pred = dt_best.predict(X_test)
dt_prob = dt_best.predict_proba(X_test)

dt_acc  = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, average="macro", zero_division=0)
dt_rec  = recall_score(y_test, dt_pred, average="macro", zero_division=0)
dt_f1   = f1_score(y_test, dt_pred, average="macro", zero_division=0)
dt_roc  = roc_auc_score(y_test, dt_prob, multi_class="ovr", average="macro")

print(f"  Accuracy : {dt_acc:.4f}")
print(f"  Precision: {dt_prec:.4f}")
print(f"  Recall   : {dt_rec:.4f}")
print(f"  F1-macro : {dt_f1:.4f}")
print(f"  ROC-AUC  : {dt_roc:.4f}")

dt_cv_f1 = cross_val_score(dt_best, X_scaled, y, cv=cv, scoring="f1_macro")
print(f"  CV F1 (5-fold): {dt_cv_f1.mean():.4f} ± {dt_cv_f1.std():.4f}")

# --------------------------------------------------------------------------
# MODEL C: Random Forest (bonus / feature importance)
# --------------------------------------------------------------------------
print("\n--- MODEL C: Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)

rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average="macro", zero_division=0)
rf_roc  = roc_auc_score(y_test, rf_prob, multi_class="ovr", average="macro")
rf_cv_f1= cross_val_score(rf, X_scaled, y, cv=cv, scoring="f1_macro")
print(f"  Accuracy : {rf_acc:.4f} | F1-macro : {rf_f1:.4f} | ROC-AUC : {rf_roc:.4f}")
print(f"  CV F1 (5-fold): {rf_cv_f1.mean():.4f} ± {rf_cv_f1.std():.4f}")

# =============================================================================
# 5. MODEL EVALUATION & VISUALISATION
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5: MODEL EVALUATION & DISCUSSION")
print("=" * 65)

class_names = le.classes_  # ['High', 'Low', 'Medium']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Model Evaluation – Classification Results", fontsize=15, fontweight="bold")

# --- Confusion matrices ---
for ax, pred, title in zip(
    [axes[0,0], axes[0,1]],
    [lr_pred, dt_pred],
    ["Logistic Regression", "Decision Tree"]
):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {title}")

# --- Feature importances (RF) ---
ax = axes[0, 2]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
importances.plot.barh(ax=ax, color="#1f77b4", edgecolor="white")
ax.set_title("Feature Importances (Random Forest)")
ax.set_xlabel("Importance")

# --- Metric comparison bar chart ---
ax = axes[1, 0]
metrics_df = pd.DataFrame({
    "Model":     ["Logistic Reg.", "Decision Tree", "Random Forest"],
    "Accuracy":  [lr_acc, dt_acc, rf_acc],
    "F1-macro":  [lr_f1,  dt_f1,  rf_f1],
    "ROC-AUC":   [lr_roc, dt_roc, rf_roc],
})
x = np.arange(len(metrics_df))
w = 0.25
ax.bar(x - w, metrics_df["Accuracy"], w, label="Accuracy",  color="#1f77b4")
ax.bar(x,     metrics_df["F1-macro"], w, label="F1-macro",  color="#ff7f0e")
ax.bar(x + w, metrics_df["ROC-AUC"], w, label="ROC-AUC",   color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# --- CV box plot ---
ax = axes[1, 1]
cv_data = [lr_cv_f1, dt_cv_f1, rf_cv_f1]
bp = ax.boxplot(cv_data, labels=["Logistic\nReg.", "Decision\nTree", "Random\nForest"],
                patch_artist=True)
colors_box = ["#aec7e8","#ffbb78","#98df8a"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
ax.set_title("5-Fold CV F1-Macro Distribution")
ax.set_ylabel("F1-Macro")

# --- Decision tree visualisation ---
ax = axes[1, 2]
plot_tree(dt_best, max_depth=3, feature_names=FEATURES,
          class_names=class_names, filled=True, ax=ax, fontsize=7)
ax.set_title("Decision Tree (depth ≤ 3)")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Model evaluation plots saved → model_evaluation.png")

# --- Classification reports ---
print("\nClassification Report – Logistic Regression:")
print(classification_report(y_test, lr_pred, target_names=class_names, zero_division=0))

print("Classification Report – Decision Tree:")
print(classification_report(y_test, dt_pred, target_names=class_names, zero_division=0))

# --- Statistical t-test: compare CV F1 scores ---
print("\n--- Statistical Test: Paired t-test (LR vs DT CV F1-macro) ---")
t_stat, p_val = stats.ttest_rel(lr_cv_f1, dt_cv_f1)
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value     = {p_val:.4f}")
if p_val < 0.05:
    print("  Result: Statistically significant difference (p < 0.05)")
else:
    print("  Result: No statistically significant difference (p ≥ 0.05)")

print("\n--- Statistical Test: Paired t-test (LR vs RF CV F1-macro) ---")
t2, p2 = stats.ttest_rel(lr_cv_f1, rf_cv_f1)
print(f"  t-statistic = {t2:.4f} | p-value = {p2:.4f}")

# =============================================================================
# 6. SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 6: FINAL SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    "Model":         ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy":      [round(lr_acc,4),  round(dt_acc,4),  round(rf_acc,4)],
    "Precision":     [round(lr_prec,4), round(dt_prec,4), "—"],
    "Recall":        [round(lr_rec,4),  round(dt_rec,4),  "—"],
    "F1-macro":      [round(lr_f1,4),   round(dt_f1,4),   round(rf_f1,4)],
    "ROC-AUC":       [round(lr_roc,4),  round(dt_roc,4),  round(rf_roc,4)],
    "CV-F1 (mean)":  [round(lr_cv_f1.mean(),4),
                      round(dt_cv_f1.mean(),4),
                      round(rf_cv_f1.mean(),4)],
})
print(summary.to_string(index=False))

print("\n✓ All outputs generated. Plots: eda_plots.png, model_evaluation.png")
print("  Upload code to GitHub: https://github.com/NavjotCDU/Assessment_2.git")