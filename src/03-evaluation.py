import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import config
from model_defs import FlagDataset, simple_collate, create_model
from utils import setup_logger

logger = setup_logger()

CLASS_NAMES = list(config.LABEL_MAP.keys())

def evaluate():
    logger.info("Starting Final Evaluation pipeline on TEST set...")

    try:
        x_path = os.path.join(config.TEST_DATA_DIR, config.X_TEST_NAME)
        y_path = os.path.join(config.TEST_DATA_DIR, config.Y_TEST_NAME)
        l_path = os.path.join(config.TEST_DATA_DIR, config.L_TEST_NAME)

        X_test = np.load(x_path)
        y_test = np.load(y_path)
        l_test = np.load(l_path)
        
        logger.info(f"Test data loaded successfully. Total samples: {len(X_test)}")
        
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}. Please run preprocessing first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Evaluation device: {device}")
    
    test_dataset = FlagDataset(X_test, y_test, l_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=simple_collate 
    )

    model = create_model(device)

    model_path = os.path.join(config.MODEL_PARAMETERS_DIR, config.MODEL_PARAMETERS_PATH)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model weights loaded from: {model_path}")
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {model_path}. Run training first!")
        return

    model.eval()
    all_predictions = []
    all_ground_truths = []

    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, predicted_indices = torch.max(outputs, 1)
            
            all_predictions.extend(predicted_indices.cpu().numpy())
            all_ground_truths.extend(labels.cpu().numpy())

    logger.info("Generating classification report...")
    
    all_class_ids = list(range(len(CLASS_NAMES)))

    report = classification_report(
        all_ground_truths, 
        all_predictions, 
        labels=all_class_ids, 
        target_names=CLASS_NAMES, 
        zero_division=0
    )
    
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS".center(60))
    print("="*60)
    print(report)
    print("="*60 + "\n")

    cm = confusion_matrix(all_ground_truths, all_predictions, labels=all_class_ids)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
        logger.info(f"Created results directory: {config.RESULTS_DIR}")

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    
    save_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {save_path}")
    logger.info("Evaluation process finished successfully.")

if __name__ == "__main__":
    evaluate()