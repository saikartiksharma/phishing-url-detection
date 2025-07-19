# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     classification_report,
#     roc_curve,
#     auc,
#     log_loss,
#     recall_score # Used for calculating specificity
# )
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC, LinearSVC

# # Set better visual styles for plots
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("viridis") # Using a colorblind-friendly palette
# sns.set_context("talk")    # Larger context for better visibility

# # === Load Dataset ===
# # Ensure 'dataset_full.csv' is in the same directory or provide the full path
# dataset_path = "dataset_full.csv" # Change this if your file is elsewhere
# try:
#     df = pd.read_csv(dataset_path)
#     print(f"Dataset loaded successfully from {dataset_path}.")
# except FileNotFoundError:
#     print(f"Error: {dataset_path} not found. Please ensure the file is in the correct directory.")
#     # Exit the script if the dataset is not found
#     exit()
# except Exception as e:
#     print(f"An error occurred while loading the dataset: {e}")
#     exit()

# # === Select Feature Columns and Target Variable ===
# # These are the features identified as relevant for the model
# selected_features = [
#     'domain_in_ip', 'length_url', 'url_shortened', 'qty_at_url',
#     'qty_slash_url', 'qty_hyphen_domain', 'qty_dot_domain',
#     'tls_ssl_certificate', 'time_domain_expiration', 'qty_dot_file',
#     'qty_equal_url', 'qty_params', 'qty_exclamation_url',
#     'qty_redirects', 'qty_ip_resolved', 'url_google_index',
#     'domain_google_index',
#     'phishing' # This is the target variable (0 for legitimate, 1 for phishing)
# ]

# # Check if all selected features exist in the dataframe
# missing_features = [col for col in selected_features if col not in df.columns]
# if missing_features:
#     print(f"Error: Missing columns in the dataset: {missing_features}")
#     print("Please check the dataset file and the selected_features list.")
#     exit()

# # Filter the dataframe to keep only the selected columns
# df = df[selected_features]

# # === Preprocess Data ===
# # Separate features (X) and target (y)
# X = df.drop('phishing', axis=1)
# y = df['phishing']

# # Handle missing values: Fill NaN values in features with 0
# # This is a simple imputation strategy; consider more advanced methods if needed
# X = X.fillna(0)

# # Scale the features: Standardize features by removing the mean and scaling to unit variance
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data into training and testing sets (80% train, 20% test)
# # random_state ensures reproducibility of the split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class distribution

# print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

# # === Evaluation Function ===
# def evaluate_model(model, name, X_test, y_test):
#     """
#     Evaluates a classification model using various metrics and prints the results.
#     Also generates and saves a confusion matrix plot.

#     Args:
#         model: The trained scikit-learn classification model.
#         name (str): The name of the model for printing and file naming.
#         X_test: The test features.
#         y_test: The true test labels.
#     """
#     print(f"\n--- Evaluating {name} ---")

#     # Make predictions on the test set
#     y_pred = model.predict(X_test)

#     # --- Basic Metrics ---
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")

#     # --- Confusion Matrix and Classification Report ---
#     # Confusion Matrix: [[TN, FP], [FN, TP]]
#     cm = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:\n", cm)
#     # Classification Report includes Precision, Recall, F1-Score for each class
#     print("Classification Report:\n", classification_report(y_test, y_pred))

#     # --- Additional Classification Metrics ---
#     # Log Loss (requires probability predictions)
#     # Measures the performance of a classification model where the prediction is a probability value between 0 and 1.
#     if hasattr(model, "predict_proba"):
#         y_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (phishing=1)
#         try:
#             logloss = log_loss(y_test, y_proba)
#             print(f"Log Loss: {logloss:.4f}")
#         except ValueError as e:
#             print(f"Could not calculate Log Loss: {e}. Check target variable encoding (should be 0 and 1).")
#     else:
#         print("Log Loss not available (model does not have predict_proba method).")

#     # Specificity (True Negative Rate)
#     # Specificity = TN / (TN + FP)
#     # In binary classification, recall for the negative class (usually 0, 'Legitimate') is specificity.
#     # We assume 0 is the negative class ('Legitimate') and 1 is the positive class ('Phishing').
#     try:
#         # Calculate recall for the negative class (pos_label=0)
#         specificity = recall_score(y_test, y_pred, pos_label=0)
#         print(f"Specificity (True Negative Rate): {specificity:.4f}")
#     except ValueError as e:
#         print(f"Could not calculate Specificity: {e}. Check target variable encoding or pos_label.")
#     except Exception as e:
#         print(f"An unexpected error occurred while calculating Specificity: {e}")


#     # --- Enhanced Confusion Matrix Plot ---
#     plt.figure(figsize=(10, 8))

#     # Calculate percentages for annotation on the heatmap
#     # Avoid division by zero if a class has no true instances
#     cm_sum = np.sum(cm, axis=1, keepdims=True)
#     cm_perc = np.zeros_like(cm).astype(float)
#     for i in range(cm.shape[0]):
#         if cm_sum[i, 0] > 0:
#             cm_perc[i, :] = cm[i, :] / cm_sum[i, 0] * 100

#     # Create annotation strings combining counts and percentages
#     annot = np.empty_like(cm).astype(str)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             annot[i, j] = f'{cm[i, j]}\n({cm_perc[i, j]:.1f}%)'

#     # Create the heatmap
#     ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', # 'Blues' colormap for a standard look
#                      cbar_kws={'label': 'Count'}, square=True, # Add a color bar with label
#                      linewidths=.5, linecolor='black') # Add lines to separate cells

#     # Improve labels and title
#     ax.set_xlabel('Predicted Labels', fontsize=14, labelpad=10)
#     ax.set_ylabel('True Labels', fontsize=14, labelpad=10)
#     plt.title(f"{name} Confusion Matrix", fontsize=16, pad=20)

#     # Add custom labels for better readability (assuming 0=Legitimate, 1=Phishing)
#     tick_labels = ['Legitimate', 'Phishing']
#     ax.set_xticklabels(tick_labels)
#     ax.set_yticklabels(tick_labels)
#     plt.yticks(rotation=0) # Ensure y-axis labels are horizontal

#     plt.tight_layout() # Adjust layout to prevent labels overlapping
#     # Save the plot to a file
#     plot_filename = f"{name.replace(' ', '_').lower()}_confusion_matrix.png"
#     plt.savefig(plot_filename, dpi=300, bbox_inches='tight') # Save with high resolution
#     plt.close() # Close the plot to free up memory
#     print(f"Confusion matrix plot saved as {plot_filename}")


# # === ROC Curve Plot Function ===
# def plot_roc_curves(models, X_test, y_test):
#     """
#     Plots ROC (Receiver Operating Characteristic) curves for multiple classification models
#     on the same plot for comparison.

#     Args:
#         models (list): A list of tuples, where each tuple contains (model_name, trained_model).
#         X_test: The test features.
#         y_test: The true test labels.
#     """
#     print("\nGenerating ROC curves comparison plot...")
#     plt.figure(figsize=(12, 10))

#     # Custom colors and line styles for better distinction between lines
#     colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1']
#     line_styles = ['-', '--', '-.', ':', '-', '--']

#     for i, (name, model) in enumerate(models):
#         # For ROC curve, we need probability scores or decision function scores
#         # predict_proba is preferred for probability-based models
#         # decision_function is used for models like SVM that don't output probabilities directly
#         y_scores = None
#         if hasattr(model, "predict_proba"):
#             try:
#                 y_scores = model.predict_proba(X_test)[:, 1] # Probability of the positive class
#             except Exception as e:
#                  print(f"Warning: Could not get predict_proba for {name}: {e}")
#         elif hasattr(model, "decision_function"):
#             try:
#                  y_scores = model.decision_function(X_test) # Distance from the decision boundary
#             except Exception as e:
#                  print(f"Warning: Could not get decision_function for {name}: {e}")
#         else:
#             print(f"Warning: Model {name} does not have predict_proba or decision_function. Cannot plot ROC curve.")
#             continue # Skip plotting for this model

#         if y_scores is not None:
#             # Calculate False Positive Rate (fpr) and True Positive Rate (tpr)
#             fpr, tpr, _ = roc_curve(y_test, y_scores)
#             # Calculate Area Under the Curve (AUC)
#             roc_auc = auc(fpr, tpr)

#             # Plot the ROC curve
#             plt.plot(fpr, tpr,
#                      label=f"{name} (AUC = {roc_auc:.3f})",
#                      color=colors[i % len(colors)],
#                      linestyle=line_styles[i % len(line_styles)],
#                      linewidth=3)

#             # Add markers to the line for better visibility
#             # Plot markers at regular intervals along the curve
#             marker_indices = np.linspace(0, len(fpr)-1, 7).astype(int) # Choose 7 points to mark
#             plt.plot(fpr[marker_indices], tpr[marker_indices],
#                      'o', color=colors[i % len(colors)],
#                      markersize=8, fillstyle='none', mew=2) # Hollow markers

#     # Add the random chance line (a diagonal line from (0,0) to (1,1))
#     plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', alpha=0.7, linewidth=2)

#     # Enhance plot elements
#     plt.xlabel("False Positive Rate", fontsize=14, labelpad=10)
#     plt.ylabel("True Positive Rate", fontsize=14, labelpad=10)
#     plt.title("ROC Curve Comparison", fontsize=18, pad=20)

#     # Enhance legend and grid
#     plt.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9, # Add a frame to the legend
#                edgecolor='black', facecolor='white')
#     plt.grid(alpha=0.3) # Add a subtle grid

#     # Add background shading to highlight the region of good performance (top-left)
#     plt.fill_between([0, 1], [0, 1], [1, 1], color='green', alpha=0.05) # Shade the area above the random line

#     # Set plot limits
#     plt.xlim([-0.01, 1.01]) # Slightly extend limits to avoid cutting off points
#     plt.ylim([-0.01, 1.01])
#     plt.tight_layout() # Adjust layout

#     # Save the plot
#     plt.savefig("roc_curves_comparison.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("ROC curves comparison plot generated as roc_curves_comparison.png")


# # === Generate Feature Importance Plot (for applicable models) ===
# def plot_feature_importance(models, X_columns):
#     """
#     Plots feature importance for tree-based models (models with a feature_importances_ attribute).

#     Args:
#         models (list): A list of tuples, where each tuple contains (model_name, trained_model).
#         X_columns (pd.Index): The names of the features.
#     """
#     print("\nGenerating feature importance plot...")

#     feature_importances_data = {}
#     feature_importances_data['Feature'] = X_columns.tolist() # Convert index to list for DataFrame

#     # Collect feature importances from applicable models
#     # Only models like Decision Trees and Random Forests have feature_importances_
#     for name, model in models:
#         if hasattr(model, 'feature_importances_'):
#             feature_importances_data[name] = model.feature_importances_
#         # Linear models (like LinearSVC) have coefficients (coef_), but feature_importances_
#         # is the standard for this type of plot, typically used for ensemble/tree models.

#     # Check if any models with feature_importances_ were found
#     if len(feature_importances_data) <= 1: # Only 'Feature' column exists
#         print("No models with 'feature_importances_' attribute found to plot feature importance.")
#         return

#     feature_importances_df = pd.DataFrame(feature_importances_data)

#     # Sort features by the importance of the first tree-based model found
#     # This provides a consistent order for comparison
#     sort_by_col = None
#     for col in feature_importances_df.columns:
#         if col != 'Feature':
#             sort_by_col = col
#             break # Use the first model's importance for sorting

#     if sort_by_col:
#         feature_importances_df = feature_importances_df.sort_values(sort_by_col, ascending=False)
#     else:
#         # If no tree models were found, this block won't be reached due to the check above
#         pass # Should not happen if the initial check works

#     feature_importances_df = feature_importances_df.set_index('Feature')

#     # Plot horizontal bar chart for feature importance
#     ax = feature_importances_df.plot(kind='barh', figsize=(14, 10),
#                                      color=['#FF9999', '#66B2FF', '#C2C2F0'][:len(feature_importances_df.columns)]) # Use distinct colors

#     # Enhance the plot
#     plt.xlabel('Feature Importance (Gini Importance)', fontsize=14, labelpad=10)
#     plt.ylabel('Features', fontsize=14, labelpad=10)
#     plt.title('Feature Importance Comparison (Tree-based Models)', fontsize=18, pad=20)
#     plt.grid(axis='x', alpha=0.3) # Add horizontal grid lines
#     plt.legend(fontsize=12, frameon=True) # Add a legend with a frame
#     plt.gca().invert_yaxis() # Invert y-axis to have the most important feature at the top
#     plt.tight_layout() # Adjust layout

#     # Add value labels to the bars
#     for container in ax.containers:
#         ax.bar_label(container, fmt='%.3f', padding=3) # Add value labels with 3 decimal places

#     # Save the plot
#     plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("Feature importance plot generated as feature_importance.png")


# # === Create a directory to save models if it doesn't exist ===
# models_dir = "models"
# os.makedirs(models_dir, exist_ok=True)
# print(f"\nEnsured directory '{models_dir}' exists for saving models.")


# # === Train & Save Models ===

# # Initialize list to store trained models for later plotting
# trained_models = []

# # 1. Decision Tree Classifier
# dt = DecisionTreeClassifier(random_state=42)
# print("\nTraining Decision Tree Classifier...")
# dt.fit(X_train, y_train)
# evaluate_model(dt, "Decision Tree", X_test, y_test)
# dt_model_path = os.path.join(models_dir, "decision_tree_model.pkl")
# joblib.dump(dt, dt_model_path)
# print(f"Decision Tree model trained and saved to {dt_model_path}.")
# trained_models.append(("Decision Tree", dt))


# # 2. Random Forest Classifier
# # n_estimators: The number of trees in the forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
# print("\nTraining Random Forest Classifier...")
# rf.fit(X_train, y_train)
# evaluate_model(rf, "Random Forest", X_test, y_test)
# rf_model_path = os.path.join(models_dir, "random_forest_model.pkl")
# joblib.dump(rf, rf_model_path)
# print(f"Random Forest model trained and saved to {rf_model_path}.")
# trained_models.append(("Random Forest", rf))


# # 3. Linear Support Vector Machine (SVM)
# # LinearSVC is suitable for large datasets and linear classification
# # Increased max_iter for potential convergence issues on some datasets
# linear_svc = LinearSVC(max_iter=20000, random_state=42)
# print("\nTraining Linear SVM...")
# # LinearSVC does not have predict_proba by default, it uses decision_function for ROC curve.
# try:
#     linear_svc.fit(X_train, y_train)
#     evaluate_model(linear_svc, "Linear SVM", X_test, y_test)
#     linear_svc_model_path = os.path.join(models_dir, "linear_svm_model.pkl")
#     joblib.dump(linear_svc, linear_svc_model_path)
#     print(f"Linear SVM model trained and saved to {linear_svc_model_path}.")
#     trained_models.append(("Linear SVM", linear_svc))
# except Exception as e:
#     print(f"Error training Linear SVM: {e}. Consider increasing max_iter or checking data scaling/distribution.")
#     # Add a placeholder None if training fails, so it's not included in plotting
#     trained_models.append(("Linear SVM", None))


# # 4. Support Vector Machine (SVM) with Radial Basis Function (RBF) Kernel
# # SVC with RBF kernel can capture non-linear relationships
# # probability=True is needed to get predict_proba for Log Loss and ROC curve plotting
# rbf_svm = SVC(kernel='rbf', probability=True, random_state=42)
# print("\nTraining SVM (RBF Kernel)...")
# try:
#     rbf_svm.fit(X_train, y_train)
#     evaluate_model(rbf_svm, "SVM (RBF Kernel)", X_test, y_test)
#     rbf_svm_model_path = os.path.join(models_dir, "rbf_svm_model.pkl")
#     joblib.dump(rbf_svm, rbf_svm_model_path)
#     print(f"SVM (RBF Kernel) model trained and saved to {rbf_svm_model_path}.")
#     trained_models.append(("SVM RBF", rbf_svm))
# except Exception as e:
#     print(f"Error training SVM (RBF Kernel): {e}. Consider reducing dataset size or adjusting parameters.")
#     # Add a placeholder None if training fails
#     trained_models.append(("SVM RBF", None))


# # === Plot ROC Curves for All Successfully Trained Models ===
# # Filter out models that failed to train (where the model object is None)
# models_for_roc_plot = [(name, model) for name, model in trained_models if model is not None]

# if models_for_roc_plot:
#     plot_roc_curves(models_for_roc_plot, X_test, y_test)
# else:
#     print("\nNo models successfully trained or available with probability/decision function for ROC curve plotting.")


# # === Generate Feature Importance Plot (for applicable models - tree-based) ===
# # Only tree-based models (Decision Tree, Random Forest) have feature_importances_
# importance_models = [
#     (name, model) for name, model in trained_models
#     if model is not None and hasattr(model, 'feature_importances_')
# ]

# if importance_models:
#     plot_feature_importance(importance_models, X.columns)
# else:
#     print("\nNo tree-based models successfully trained to plot feature importance.")


# print("\n--- Script Finished ---")
# print(f"All successfully trained models evaluated and saved in the '{models_dir}' directory.")
# print("Generated visualizations: confusion matrices (per model), ROC curves comparison, and feature importance plot (for tree models).")
# print("Check the current directory for saved model files (.pkl) and plot images (.png).")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    auc,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

# Set better visual styles for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis") # Using a colorblind-friendly palette
sns.set_context("talk")    # Larger context for better visibility

# === Load Dataset ===
# Ensure 'dataset_full.csv' is in the same directory or provide the full path
dataset_path = "dataset_full.csv" # Change this if your file is elsewhere
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully from {dataset_path}.")
except FileNotFoundError:
    print(f"Error: {dataset_path} not found. Please ensure the file is in the correct directory.")
    # Exit the script if the dataset is not found
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# === Select Feature Columns and Target Variable ===
# These are the features identified as relevant for the model
selected_features = [
    'domain_in_ip', 'length_url', 'url_shortened', 'qty_at_url',
    'qty_slash_url', 'qty_hyphen_domain', 'qty_dot_domain',
    'tls_ssl_certificate', 'time_domain_expiration', 'qty_dot_file',
    'qty_equal_url', 'qty_params', 'qty_exclamation_url',
    'qty_redirects', 'qty_ip_resolved', 'url_google_index',
    'domain_google_index',
    'phishing' # This is the target variable (0 for legitimate, 1 for phishing)
]

# Check if all selected features exist in the dataframe
missing_features = [col for col in selected_features if col not in df.columns]
if missing_features:
    print(f"Error: Missing columns in the dataset: {missing_features}")
    print("Please check the dataset file and the selected_features list.")
    exit()

# Filter the dataframe to keep only the selected columns
df = df[selected_features]

# === Preprocess Data ===
# Separate features (X) and target (y)
X = df.drop('phishing', axis=1)
y = df['phishing']

# Handle missing values: Fill NaN values in features with 0
# This is a simple imputation strategy; consider more advanced methods if needed
X = X.fillna(0)

# Scale the features: Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class distribution

print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

# === Create a directory to save models if it doesn't exist ===
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
print(f"\nEnsured directory '{models_dir}' exists for saving models.")

# === Train & Save Models ===

# Initialize list to store trained models and their names for later evaluation
trained_models = []

# 1. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
print("\nTraining Decision Tree Classifier...")
dt.fit(X_train, y_train)
dt_model_path = os.path.join(models_dir, "decision_tree_model.pkl")
joblib.dump(dt, dt_model_path)
print(f"Decision Tree model trained and saved to {dt_model_path}.")
trained_models.append(("Decision Tree", dt))


# 2. Random Forest Classifier
# n_estimators: The number of trees in the forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
print("\nTraining Random Forest Classifier...")
rf.fit(X_train, y_train)
rf_model_path = os.path.join(models_dir, "random_forest_model.pkl")
joblib.dump(rf, rf_model_path)
print(f"Random Forest model trained and saved to {rf_model_path}.")
trained_models.append(("Random Forest", rf))


# 3. Linear Support Vector Machine (SVM)
# LinearSVC is suitable for large datasets and linear classification
# Increased max_iter for potential convergence issues on some datasets
linear_svc = LinearSVC(max_iter=20000, random_state=42)
print("\nTraining Linear SVM...")
try:
    linear_svc.fit(X_train, y_train)
    linear_svc_model_path = os.path.join(models_dir, "linear_svm_model.pkl")
    joblib.dump(linear_svc, linear_svc_model_path)
    print(f"Linear SVM model trained and saved to {linear_svc_model_path}.")
    # LinearSVC does not have predict_proba, but has decision_function for ROC/AUC
    trained_models.append(("Linear SVM", linear_svc))
except Exception as e:
    print(f"Error training Linear SVM: {e}. Consider increasing max_iter or checking data scaling/distribution.")
    # Add a placeholder None if training fails
    trained_models.append(("Linear SVM", None))


# 4. Support Vector Machine (SVM) with Radial Basis Function (RBF) Kernel
# SVC with RBF kernel can capture non-linear relationships
# probability=True is needed to get predict_proba for AUC calculation
rbf_svm = SVC(kernel='rbf', probability=True, random_state=42)
print("\nTraining SVM (RBF Kernel)...")
try:
    rbf_svm.fit(X_train, y_train)
    rbf_svm_model_path = os.path.join(models_dir, "rbf_svm_model.pkl")
    joblib.dump(rbf_svm, rbf_svm_model_path)
    print(f"SVM (RBF Kernel) model trained and saved to {rbf_svm_model_path}.")
    # Changed the name here from "SVM RBF" to "RBF"
    trained_models.append(("RBF", rbf_svm))
except Exception as e:
    print(f"Error training SVM (RBF Kernel): {e}. Consider reducing dataset size or adjusting parameters.")
    # Add a placeholder None if training fails
    trained_models.append(("RBF", None))


# === Calculate AUC Scores ===
def calculate_auc_scores(models, X_test, y_test):
    """
    Calculates the AUC score for each trained model.

    Args:
        models (list): A list of tuples, where each tuple contains (model_name, trained_model).
        X_test: The test features.
        y_test: The true test labels.

    Returns:
        dict: A dictionary where keys are model names and values are their AUC scores.
    """
    print("\nCalculating AUC scores...")
    auc_scores = {}
    for name, model in models:
        if model is not None:
            y_scores = None
            # Get scores for AUC calculation (prefer predict_proba, fallback to decision_function)
            if hasattr(model, "predict_proba"):
                try:
                    y_scores = model.predict_proba(X_test)[:, 1] # Probability of the positive class
                except Exception as e:
                     print(f"Warning: Could not get predict_proba for {name} for AUC: {e}")
            elif hasattr(model, "decision_function"):
                try:
                     y_scores = model.decision_function(X_test) # Distance from the decision boundary
                except Exception as e:
                     print(f"Warning: Could not get decision_function for {name} for AUC: {e}")
            else:
                print(f"Warning: Model {name} does not have predict_proba or decision_function. Cannot calculate AUC.")


            if y_scores is not None:
                try:
                    # Calculate AUC
                    fpr, tpr, _ = roc_curve(y_test, y_scores)
                    roc_auc = auc(fpr, tpr)
                    auc_scores[name] = roc_auc
                    print(f"{name} AUC: {roc_auc:.4f}")
                except Exception as e:
                    print(f"Error calculating AUC for {name}: {e}")
        else:
            print(f"Skipping AUC calculation for {name} as the model failed to train.")

    return auc_scores

# === Plot AUC Bar Graph ===
def plot_auc_bargraph(auc_scores):
    """
    Generates a bar graph comparing the AUC scores of different models.

    Args:
        auc_scores (dict): A dictionary where keys are model names and values are their AUC scores.
    """
    if not auc_scores:
        print("\nNo AUC scores available to plot the bar graph.")
        return

    print("\nGenerating AUC comparison bar graph...")
    # Convert dictionary to pandas Series for easy plotting
    auc_series = pd.Series(auc_scores).sort_values(ascending=False)

    plt.figure(figsize=(10, 7)) # Adjust figure size

    # Create the bar plot
    ax = sns.barplot(x=auc_series.index, y=auc_series.values, palette="viridis")

    # Add titles and labels
    plt.title("Model AUC Comparison", fontsize=18, pad=20)
    plt.xlabel("Model", fontsize=14, labelpad=10)
    plt.ylabel("AUC Score", fontsize=14, labelpad=10)
    plt.ylim(0, 1.0) # AUC scores are between 0 and 1

    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout() # Adjust layout
    # Save the plot
    plot_filename = "model_auc_comparison_bargraph.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC comparison bar graph generated as {plot_filename}.")


# === Main Execution for AUC Calculation and Plotting ===
# Filter out models that failed to train (where the model object is None)
successfully_trained_models = [(name, model) for name, model in trained_models if model is not None]

if successfully_trained_models:
    # Calculate AUC scores
    auc_results = calculate_auc_scores(successfully_trained_models, X_test, y_test)

    # Plot AUC bar graph
    plot_auc_bargraph(auc_results)
else:
    print("\nNo models successfully trained to calculate AUC and plot the bar graph.")


print("\n--- Script Finished ---")
print(f"All successfully trained models saved in the '{models_dir}' directory.")
print("Generated visualization: AUC comparison bar graph.")
print("Check the current directory for saved model files (.pkl) and the AUC bar graph image (.png).")
