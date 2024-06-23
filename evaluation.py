import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns

from datasetclass import TrainDataset, ValDataset
from setup import transformations
from model import CustomResNetForEvaluation

def evaluate_model(model_path, data_dir, test_csv, plot_dir, batch_size=32, device='cuda:1', num_classes=7):

    os.makedirs(f'{plot_dir}/performance', exist_ok=True)
    os.makedirs(f'{plot_dir}/fairness', exist_ok=True)
    os.makedirs(f'{plot_dir}/roc_auc', exist_ok=True)
    os.makedirs(f'{plot_dir}/age_groups', exist_ok=True)

    # Define the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Selected device:", device)

    # Load the trained model state dictionary
    state_dict = torch.load(model_path)

    # Create an instance of CustomResNetForEvaluation
    model = CustomResNetForEvaluation(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Define the test dataset and data loader
    test_dataset = ValDataset(data_dir=data_dir, csv_file=test_csv, transform=transformations())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define lists to store predictions and targets
    test_predictions = []
    test_targets = []
    test_probs = []
    age_groups = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    age_group_metrics = {age_group: {'preds': [], 'targets': []} for age_group in age_groups}

    # Iterate through the test dataset and make predictions
    with torch.no_grad():
        for images, labels, ages in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

            for label, pred, age in zip(labels.cpu().numpy(), predicted.cpu().numpy(), ages.cpu().numpy()):
                age_group = age_groups[age]  # Map age index back to age group
                age_group_metrics[age_group]['preds'].append(pred)
                age_group_metrics[age_group]['targets'].append(label)

    # Convert predictions and targets to numpy arrays
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)

    # Calculate overall evaluation metrics
    accuracy = accuracy_score(test_targets, test_predictions)
    precision = precision_score(test_targets, test_predictions, average='macro')
    recall = recall_score(test_targets, test_predictions, average='macro')
    f1 = f1_score(test_targets, test_predictions, average='macro')

    # Store the metrics in a DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })

    # Save the DataFrame to a CSV file
    metrics_path = os.path.join(plot_dir, 'overall_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)


    print(f"Saved overall metrics to {metrics_path}")

    race_names = {
        0: 'East Asian',
        1: 'Indian',
        2: 'Black',
        3: 'White',
        4: 'Middle Eastern',
        5: 'Latino_Hispanic',
        6: 'Southeast Asian'
    }

    # Compute confusion matrix
    cm = confusion_matrix(test_targets, test_predictions)

    # Print overall evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Initialize dictionaries to store aggregated metrics
    metrics_summary = {
        'age_group': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'equalized_odds_tpr': [],
        'equalized_odds_fpr': [],
        'demographic_parity': []
    }

    # Plot overall confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=race_names.values(), yticklabels=race_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=22.5)
    plt.yticks(rotation=22.5)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/performance/confusion_matrix.png')
    plt.close()

    # Fairness analysis for each age group
    for age_group, metrics in age_group_metrics.items():
        preds = np.array(metrics['preds'])
        targets = np.array(metrics['targets'])

        if len(preds) > 0:
            accuracy = accuracy_score(targets, preds)
            precision = precision_score(targets, preds, average='macro')
            recall = recall_score(targets, preds, average='macro')
            f1 = f1_score(targets, preds, average='macro')
            cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

            # Compute Equalized Odds
            tprs = []
            fprs = []
            for i in range(num_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - (tp + fn + fp + np.sum(cm[:, i]))

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tprs.append(tpr)
                fprs.append(fpr)

            # Compute Demographic Parity
            ppvs = []
            for i in range(num_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                ppvs.append(ppv)

            metrics_summary['age_group'].append(age_group)
            metrics_summary['accuracy'].append(accuracy)
            metrics_summary['precision'].append(precision)
            metrics_summary['recall'].append(recall)
            metrics_summary['f1_score'].append(f1)
            metrics_summary['equalized_odds_tpr'].append(np.mean(tprs))
            metrics_summary['equalized_odds_fpr'].append(np.mean(fprs))
            metrics_summary['demographic_parity'].append(np.mean(ppvs))

            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=race_names.values(),
                        yticklabels=race_names.values())
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for Age Group {age_group}')
            plt.xticks(rotation=22.5)
            plt.yticks(rotation=22.5)
            plt.savefig(f'{plot_dir}/age_groups/confusion_matrix_{age_group}.png')
            plt.close()

    # Create dataframes for performance and fairness metrics
    performance_df = pd.DataFrame({
        'Age Group': metrics_summary['age_group'],
        'Accuracy': metrics_summary['accuracy'],
        'Precision': metrics_summary['precision'],
        'Recall': metrics_summary['recall'],
        'F1 Score': metrics_summary['f1_score']
    })

    fairness_df = pd.DataFrame({
        'Age Group': metrics_summary['age_group'],
        'Equalized Odds TPR': metrics_summary['equalized_odds_tpr'],
        'Equalized Odds FPR': metrics_summary['equalized_odds_fpr'],
        'Demographic Parity': metrics_summary['demographic_parity']
    })

    # Save the dataframes as CSV files
    performance_df.to_csv(f'{plot_dir}/performance/performance_metrics.csv', index=False)
    fairness_df.to_csv(f'{plot_dir}/fairness/fairness_metrics.csv', index=False)

    # Plotting the performance metrics for each age group
    age_groups = metrics_summary['age_group']
    x = np.arange(len(age_groups))
    width = 0.5

    # Define figure size
    fig_size = (12, 8)

    # Creating a separate plot for each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        fig, ax = plt.subplots(figsize=fig_size)
        # Plot bars for each metric
        rects = ax.bar(x, metrics_summary[metric], width, label=metric.title(), color='skyblue')

        # Labeling and titling
        ax.set_xlabel('Age Groups')
        ax.set_ylabel('Scores')
        ax.set_title(f'{metric.title()} by Age Group')
        ax.set_xticks(x)
        ax.set_xticklabels(age_groups, rotation=45)

        # Adding value labels on top of each bar for clarity
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.legend()
        fig.tight_layout()
        plt.ylim([0.55, 0.8])
        plt.savefig(f'{plot_dir}/performance/{metric}_by_age_group.png')
        plt.close()

    # Plotting Equalized Odds for each age group
    fig, ax = plt.subplots(figsize=(12, 8))
    rects5 = ax.bar(x, metrics_summary['equalized_odds_tpr'], width, label='Equalized Odds TPR', color='skyblue')
    rects6 = ax.bar(x + width, metrics_summary['equalized_odds_fpr'], width, label='Equalized Odds FPR', color='coral')

    ax.set_xlabel('Age Groups')
    ax.set_ylabel('Scores')
    ax.set_title('Equalized Odds by Age Group')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_summary['age_group'])
    ax.legend()

    ax.set_ylim(0.0, 1.0)

    # Annotate values on top of the bars
    for rect in rects5 + rects6:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(f'{plot_dir}/fairness/equalized_odds_by_age_group.png')

    # Plotting Demographic Parity for each age group
    fig, ax = plt.subplots(figsize=(12, 8))

    rects7 = ax.bar(x, metrics_summary['demographic_parity'], width, label='Demographic Parity (PPR)')

    ax.set_xlabel('Age Groups')
    ax.set_ylabel('Scores')
    ax.set_title('Demographic Parity (PPR) by Age Group')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_summary['age_group'])
    ax.legend()

    ax.set_ylim(0, 1)

    # Annotate values on top of the Demographic Parity bars
    for rect in rects7:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(f'{plot_dir}/fairness/demographic_parity_by_age_group.png')

    # Plotting ROC Curve and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_targets, test_probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{race_names[i]} ROC curve (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve and AUC')
    plt.legend(loc="lower right")
    plt.savefig(f'{plot_dir}/roc_auc/roc_auc.png')
    plt.close()

    return performance_df, fairness_df
