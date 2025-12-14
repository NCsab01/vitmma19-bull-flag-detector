from utils import setup_logger
import config
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from model_defs import FlagDataset, simple_collate, create_model

logger = setup_logger()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train():
    logger.info("Starting training pipeline...")

    logger.info("1. CONFIGURATION")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"Weight Decay: {config.WEIGHT_DECAY}")
    logger.info(f"Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if not os.path.exists(config.MODEL_PARAMETERS_DIR):
        os.makedirs(config.MODEL_PARAMETERS_DIR)

    logger.info("2. DATA PROCESSING")
    try:
        x_train = np.load(os.path.join(config.TRAIN_DATA_DIR, config.X_TRAIN_NAME))
        y_train = np.load(os.path.join(config.TRAIN_DATA_DIR, config.Y_TRAIN_NAME))
        l_train = np.load(os.path.join(config.TRAIN_DATA_DIR, config.L_TRAIN_NAME))
        
        x_val = np.load(os.path.join(config.TRAIN_DATA_DIR, config.X_VAL_NAME))
        y_val = np.load(os.path.join(config.TRAIN_DATA_DIR, config.Y_VAL_NAME))
        l_val = np.load(os.path.join(config.TRAIN_DATA_DIR, config.L_VAL_NAME))
        
        logger.info("Data loaded successfully.")
        logger.info(f"Train samples: {len(x_train)}")
        logger.info(f"Validation samples: {len(x_val)}")
        logger.info(f"Input Shape: {x_train.shape}")
        
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
        return

    train_dataset = FlagDataset(x_train, y_train, l_train, augment=True)
    val_dataset = FlagDataset(x_val, y_val, l_val)
    
    target_list = torch.tensor(y_train)
    class_count = [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    class_weights_sampler = 1.0 / torch.tensor(class_count, dtype=torch.float) 
    sample_weights = np.array([class_weights_sampler[t] for t in target_list])
    sample_weights = torch.from_numpy(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
  
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=simple_collate
    )  
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=simple_collate
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)

    logger.info("3. MODEL ARCHITECTURE")
    total_params, trainable_params = count_parameters(model)
    logger.info(str(model))
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable Parameters: {trainable_params}")
    logger.info(f"Non-trainable Parameters: {total_params - trainable_params}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY 
    )
    
    best_val_loss = float('inf')
    counter = 0 

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) 
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total if total > 0 else 0

        logger.info(f"Epoch [{epoch+1}/{config.EPOCHS}] | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Accuracy: {accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            save_path = os.path.join(config.MODEL_PARAMETERS_DIR, config.MODEL_PARAMETERS_PATH)
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    logger.info("Training process finished.")

if __name__ == "__main__":
    train()