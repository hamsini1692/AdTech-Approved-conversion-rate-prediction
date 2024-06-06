from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title(title, fontsize=18)
    plt.ylabel('True', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    st.pyplot(plt)

def get_metrics(y_train, y_val, y_test, predictions, probabilities):
    sets = ['train', 'val', 'test']
    metrics_results = {}

    for set_name in sets:
        y_true = eval(f'y_{set_name}')
        y_pred = predictions[set_name]

        # RidgeClassifierCV does not support predict_proba, so AUC is not calculated
        auc_text = "AUC: Not available (model does not support predict_proba)"

        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics_results[set_name] = {
            "AUC": auc_text,
            "Classification Report": report,
            "Confusion Matrix": cm
        }

    return metrics_results
