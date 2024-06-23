import os
import json
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models')

from setup import device, transformations
from utils import (save_best_params, load_best_params, train_and_save_model,
                   hyperparameter_tuning, error_analysis, load_and_plot_tables,
                   display_hyperparameters, load_and_plot_overall_metrics)
from evaluation import evaluate_model
from model import CustomResNet
from datasetclass import ValDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting the main script...")

    # Split the existing validation dataset into validation and test sets
    val_csv = "/home/u983034/Data/fairface_label_val.csv"
    annotations = pd.read_csv(val_csv)
    val_annotations, test_annotations = train_test_split(annotations, test_size=0.5, random_state=42)

    # Save new splits
    val_annotations.to_csv("/home/u983034/Data/fairface_label_val_split.csv", index=False)
    test_annotations.to_csv("/home/u983034/Data/fairface_label_test_split.csv", index=False)

    # Paths to datasets
    data_dir = "/home/u983034/Data/"
    train_csv = os.path.join(data_dir, "fairface_label_train.csv")
    val_csv = os.path.join(data_dir, "fairface_label_val_split.csv")
    test_csv = os.path.join(data_dir, "fairface_label_test_split.csv")
    transform = transformations()


    logging.info("Starting hyperparameter tuning...")

    best_params_dir = os.path.join("/home/u983034/Models", "best_params")
    os.makedirs(best_params_dir, exist_ok=True)

    # Tune and save the best parameters for the Baseline model
    best_params_bl_path = os.path.join(best_params_dir, 'best_params_bl.json')
    best_params_bl, best_accuracy_bl = hyperparameter_tuning(
        train_csv, val_csv, data_dir, transform,
        n_trials=5, use_weighted_loss=False, use_weighted_sampling=False,
        save_path=best_params_bl_path
    )
    logging.info(f"Best Hyperparameters for Baseline: {best_params_bl}")
    logging.info(f"Best Cross-Validation Accuracy for Baseline: {best_accuracy_bl:.4f}")

    # Tune and save the best parameters for the Weighted Sampling model
    best_params_ws_path = os.path.join(best_params_dir, 'best_params_ws.json')
    best_params_ws, best_accuracy_ws = hyperparameter_tuning(
        train_csv, val_csv, data_dir, transform,
        n_trials=5, use_weighted_loss=False, use_weighted_sampling=True,
        save_path=best_params_ws_path
    )
    logging.info(f"Best Hyperparameters for Weighted Sampling: {best_params_ws}")
    logging.info(f"Best Cross-Validation Accuracy for Weighted Sampling: {best_accuracy_ws:.4f}")

    # Tune and save the best parameters for the Weighted Loss model
    best_params_wl_path = os.path.join(best_params_dir, 'best_params_wl.json')
    best_params_wl, best_accuracy_wl = hyperparameter_tuning(
        train_csv, val_csv, data_dir, transform,
        n_trials=20, use_weighted_loss=True, use_weighted_sampling=False,
        save_path=best_params_wl_path
    )
    logging.info(f"Best Hyperparameters for Weighted Loss: {best_params_wl}")
    logging.info(f"Best Cross-Validation Accuracy for Weighted Loss: {best_accuracy_wl:.4f}")

    logging.info("Completed hyperparameter tuning. Proceeding to model training...")


    # Paths to best parameter JSON files
    best_params_dir = os.path.join("/home/u983034/Models", "best_params")
    best_params_bl_path = os.path.join(best_params_dir, 'best_params_bl.json')
    best_params_ws_path = os.path.join(best_params_dir, 'best_params_ws.json')
    best_params_wl_path = os.path.join(best_params_dir, 'best_params_wl.json')

    logging.info("Loading best hyperparameters and starting model training...")

    # Load the best parameters and train the Baseline model
    best_params_bl = load_best_params(best_params_bl_path)
    model_bl, best_model_state_bl = train_and_save_model(
        best_params_bl, '/home/u983034/Models/resnet_bl_best.pth',
        '/home/u983034/Plots/performance/bl', train_csv, val_csv, test_csv, data_dir, transform,
        use_weighted_sampling=False, use_weighted_loss=False
    )

    # Load the best parameters for evaluation
    best_params_dir = os.path.join("/home/u983034/Models", "best_params")
    best_params_bl_path = os.path.join(best_params_dir, 'best_params_bl.json')
    best_params_ws_path = os.path.join(best_params_dir, 'best_params_ws.json')
    best_params_wl_path = os.path.join(best_params_dir, 'best_params_wl.json')

    logging.info("Loading best hyperparameters for evaluation...")

    best_params_bl = load_best_params(best_params_bl_path)
    best_params_ws = load_best_params(best_params_ws_path)
    best_params_wl = load_best_params(best_params_wl_path)

    # Call the display_hyperparameters function
    model_names = {
        'baseline': 'Baseline',
        'weighted_sampling': 'Weighted Sampling',
        'weighted_loss': 'Weighted Loss'
    }
    best_params_paths = {
        'baseline': 'best_params_bl.json',
        'weighted_sampling': 'best_params_ws.json',
        'weighted_loss': 'best_params_wl.json'
    }

    display_hyperparameters(best_params_dir, best_params_paths, model_names)

    # Evaluate the Baseline model on the test set
    logging.info("Starting Baseline evaluation...")
    evaluate_model('/home/u983034/Models/resnet_bl_best.pth', data_dir, test_csv, '/home/u983034/Plots/baseline',
                   batch_size=best_params_bl['batch_size'], device=device)
                   
    # Evaluate the Weighted Sampling model on the test set
    logging.info("Starting Weighted Sampling evaluation...")
    evaluate_model('/home/u983034/Models/resnet_ws_best.pth', data_dir, test_csv,
                   '/home/u983034/Plots/weighted_sampling',
                   batch_size=best_params_ws['batch_size'], device=device)

    # Evaluate the Weighted Loss model on the test set
    logging.info("Starting Weighted Loss evaluation...")
    evaluate_model('/home/u983034/Models/resnet_wl_best.pth', data_dir, test_csv, '/home/u983034/Plots/weighted_loss',
                   batch_size=best_params_wl['batch_size'], device=device)

    # Generate and save comparison tables for model metrics
    logging.info("Generating and saving comparison tables for performance and fairness metrics...")
    models_directory = '/home/u983034/Plots'
    load_and_plot_tables(models_directory, models_directory)
    load_and_plot_overall_metrics(models_directory, models_directory)



