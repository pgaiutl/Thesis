import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from datasetclass import TrainDataset, ValDataset
from model import CustomResNet
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def calculate_age_weights(csv_file):
    annotations = pd.read_csv(csv_file)
    age_distribution = annotations.iloc[:, 1].value_counts().sort_index()
    total_samples = len(annotations)
    age_weights = total_samples / (len(age_distribution) * age_distribution)
    age_weight_map = {age: weight for age, weight in zip(age_distribution.index, age_weights)}
    return [age_weight_map[age] for age in annotations.iloc[:, 1]]


def create_dataloaders(train_csv, val_csv, test_csv, data_dir, transform, batch_size, use_weighted_sampling=False):
    train_age_weights = calculate_age_weights(train_csv)

    train_dataset = TrainDataset(data_dir=data_dir, csv_file=train_csv, transform=transform,
                                 age_weights=train_age_weights if use_weighted_sampling else None)

    val_dataset = ValDataset(data_dir=data_dir, csv_file=val_csv, transform=transform)

    test_dataset = None
    if test_csv is not None:
        test_dataset = ValDataset(data_dir=data_dir, csv_file=test_csv, transform=transform)

    if use_weighted_sampling:
        sampler = WeightedRandomSampler(weights=train_age_weights, num_samples=len(train_age_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, patience=5, use_weighted_loss=False, trial=None):
    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]:")
        for i, data in enumerate(train_loader):
            images = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if use_weighted_loss and len(data) > 2:
                weights = data[2].to(device)
                weighted_loss = (loss * weights).mean()
            else:
                weighted_loss = loss.mean()

            weighted_loss.backward()
            optimizer.step()

            running_train_loss += weighted_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f" Batch [{i + 1}/{len(train_loader)}], Train Loss: {weighted_loss.item():.4f}, Train Accuracy: {(100 * correct / total):.2f}%")

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_accuracy = 100 * correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data[0].to(device)
                val_labels = val_data[1].to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = 100 * val_correct / val_total

        if trial:
            trial.report(epoch_val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        train_accuracy_list.append(epoch_train_accuracy)
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_accuracy)
        val_loss_list.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")

        scheduler.step()

    return best_model_state, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list


class WeightedLoss(nn.Module):
    def __init__(self, age_weights):
        super(WeightedLoss, self).__init__()
        self.age_weights = torch.tensor(age_weights, dtype=torch.float32).to(device)

    def forward(self, outputs, targets):
        weights = self.age_weights[targets]
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()


def error_analysis(model, val_loader):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img, label, pred in zip(images.cpu(), labels.cpu(), predicted.cpu()):
                if label != pred:
                    misclassified.append((img, label, pred))

    return misclassified


def save_best_params(params, file_path):
    with open(file_path, 'w') as f:
        json.dump(params, f)


def load_best_params(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params


def objective(trial, train_csv, val_csv, data_dir, transform, use_weighted_loss, use_weighted_sampling):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    step_size = trial.suggest_int('step_size', 5, 30)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    custom_weights_path = trial.suggest_categorical('custom_weights_path',
                                                    ['/home/u983034/Models/res34_fair_align_multi_7_20190809.pt'])

    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(train_csv, val_csv, None, data_dir, transform, batch_size,
                                                     use_weighted_sampling)

    # Initialize model, criterion, optimizer, scheduler
    model = CustomResNet(num_classes=7, custom_weights_path=custom_weights_path).to(device)
    criterion = nn.CrossEntropyLoss() if not use_weighted_loss else WeightedLoss(calculate_age_weights(train_csv))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train model
    best_model_state, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list = train_model(
        model, train_loader, val_loader, num_epochs=10,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        patience=5, use_weighted_loss=use_weighted_loss, trial=trial
    )

    # Return best validation accuracy
    return max(val_accuracy_list)


def hyperparameter_tuning(train_csv, val_csv, data_dir, transform, n_trials=5, use_weighted_loss=False,
                          use_weighted_sampling=False, save_path='best_params.json'):
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_csv, val_csv, data_dir, transform, use_weighted_loss,
                                           use_weighted_sampling), n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Save best parameters to file
    save_best_params(best_params, save_path)

    return best_params, best_value


def train_and_save_model(best_params, model_save_path, performance_plot_path, train_csv, val_csv, test_csv, data_dir,
                         transform, use_weighted_sampling=False, use_weighted_loss=False):

    os.makedirs(performance_plot_path, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv, val_csv, test_csv, data_dir, transform,
        best_params['batch_size'], use_weighted_sampling=use_weighted_sampling
    )
    model = CustomResNet(num_classes=7, custom_weights_path=best_params['custom_weights_path']).to(device)

    criterion = nn.CrossEntropyLoss() if not use_weighted_loss else WeightedLoss(calculate_age_weights(train_csv))
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = StepLR(optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])

    best_model_state, train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list = train_model(
        model, train_loader, val_loader, num_epochs=50,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        use_weighted_loss=use_weighted_loss, patience=10


    torch.save(best_model_state, model_save_path)

    # Plot training accuracy and loss
    plt.figure()
    plt.plot(train_accuracy_list, label='Train Accuracy')
    plt.plot(val_accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(performance_plot_path, 'training_validation_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(performance_plot_path, 'training_validation_loss.png'))
    plt.close()

    return model, best_model_state

def load_and_plot_tables(base_dir, plot_dir):
    models = ['baseline', 'weighted_sampling', 'weighted_loss']
    model_display_names = {'baseline': 'Baseline', 'weighted_sampling': 'Weighted Sampling',
                           'weighted_loss': 'Weighted Loss'}
    metric_types = ['performance', 'fairness']


    plt.rc('font', family='DejaVu Serif', size=10)
    plt.rc('figure', autolayout=True)

    for metric in metric_types:
        combined_data = pd.DataFrame()

        for model in models:
            csv_path = os.path.join(base_dir, model, metric, f'{metric}_metrics.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['Model'] = model_display_names[model]
                combined_data = pd.concat([combined_data, df], ignore_index=True)

        if not combined_data.empty:
            metrics = combined_data.columns.drop(['Age Group', 'Model'])
            for single_metric in metrics:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('off')
                combined_data[single_metric] = combined_data[single_metric].round(3)
                pivot_table = combined_data.pivot_table(index='Age Group', columns='Model', values=single_metric,
                                                        aggfunc='first')

                # Create the table
                table = ax.table(cellText=pivot_table.values, rowLabels=pivot_table.index,
                                 colLabels=pivot_table.columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.4)

                # Set all borders to invisible
                for key, cell in table.get_celld().items():
                    cell.visible_edges = ''  # Make all edges invisible


                ax.axhline(y=1, color='black', linewidth=1.5)
                ax.axhline(y=0.9, color='black', linewidth=1.5)
                ax.axhline(y=0.01, color='black', linewidth=1.5)

                plt.title(f'{single_metric.capitalize()} Comparison across Models', fontsize=12, weight='bold')
                plt.savefig(os.path.join(plot_dir, f'{metric}_{single_metric}_comparison.png'), format='png', dpi=300)
                plt.close()

    print("Individual metric tables for each model have been saved in PNG format.")


def display_hyperparameters(base_dir, best_params_paths, model_names):

    hyperparameters_df = pd.DataFrame()

    for model, file_name in best_params_paths.items():

        file_path = os.path.join(base_dir, file_name)
        try:
            with open(file_path, 'r') as file:
                params = json.load(file)
                params.pop('custom_weights_path', None)
                if 'learning_rate' in params:
                    params['learning_rate'] = "{:.6f}".format(params['learning_rate'])
                elif 'lr' in params:
                    params['lr'] = "{:.6f}".format(params['lr'])
                if 'gamma' in params:
                    params['gamma'] = "{:.4f}".format(params['gamma'])

            df = pd.DataFrame([params], columns=params.keys())
            df['Model'] = model_names[model]
            hyperparameters_df = pd.concat([hyperparameters_df, df], ignore_index=True)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue


    if not hyperparameters_df.empty:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.axis('off')
        ax.axhline(y=0.8, color='black', linewidth=1.5)
        ax.axhline(y=0.675, color='black', linewidth=1.5)
        ax.axhline(y=0.2, color='black', linewidth=1.5)
        table = ax.table(cellText=hyperparameters_df.values, colLabels=hyperparameters_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)
        ax.set_title('Hyperparameters Configuration for Models', pad=1)


        plt.savefig(os.path.join(base_dir, 'hyperparameters_table.png'))
        plt.show()
        plt.close()


def load_and_plot_overall_metrics(base_dir, plot_dir):
    models = ['baseline', 'weighted_sampling', 'weighted_loss']
    all_metrics = pd.DataFrame()

    plt.rc('font', family='DejaVu Serif', size=10)
    plt.rc('figure', autolayout=True)

    for model in models:
        csv_path = os.path.join(base_dir, model, 'overall_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.round(4)
            df['Model'] = model
            all_metrics = pd.concat([all_metrics, df], ignore_index=True)

    if not all_metrics.empty:
        pivot_table = all_metrics.pivot(index="Metric", columns="Model", values="Value")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        table = ax.table(cellText=pivot_table.values, rowLabels=pivot_table.index,
                         colLabels=pivot_table.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

        for key, cell in table.get_celld().items():
            cell.visible_edges = ''  

        ax.axhline(y=0.7, color='black', linewidth=1.5)
        ax.axhline(y=0.615, color='black', linewidth=1.5)
        ax.axhline(y=0.3, color='black', linewidth=1.5)

        plt.title('Comparison of accuracy, f1-score, precision, and recall per model', fontsize=12, weight='bold', y=0.8)
        plot_file_path = os.path.join(plot_dir, 'overall_metrics_comparison.png')
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Saved overall metrics comparison table at {plot_file_path}")