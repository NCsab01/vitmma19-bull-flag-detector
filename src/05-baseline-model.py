from utils import setup_logger
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
import config

logger = setup_logger()

def load_data():
    print("Loading data...")
    try:
        data_dir = config.TRAIN_DATA_DIR
        if not os.path.exists(os.path.join(data_dir, config.X_TRAIN_NAME)):
            logger.error(f"Training data file not found in {data_dir}. Please run preprocessing first.")
            
        X_train = np.load(os.path.join(data_dir, config.X_TRAIN_NAME))
        y_train = np.load(os.path.join(data_dir, config.Y_TRAIN_NAME))
        l_train = np.load(os.path.join(data_dir, config.L_TRAIN_NAME))

        eval_dir = config.TEST_DATA_DIR
        x_test_path = os.path.join(eval_dir, config.X_TEST_NAME)

        if not os.path.exists(x_test_path):
             X_test = np.load(os.path.join(config.TEST_DATA_DIR, config.X_TEST_NAME))
             y_test = np.load(os.path.join(config.TEST_DATA_DIR, config.Y_TEST_NAME))
             l_test = np.load(os.path.join(config.TEST_DATA_DIR, config.L_TEST_NAME))
        else:
             X_test = np.load(x_test_path)
             y_test = np.load(os.path.join(eval_dir, config.Y_TEST_NAME))
             l_test = np.load(os.path.join(eval_dir, config.L_TEST_NAME))
        
        return (X_train, y_train, l_train), (X_test, y_test, l_test)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def extract_features(X, lengths):
    print("Running Feature Engineering...")
    features = []
    
    for i in range(len(X)):
        real_len = lengths[i]
        closes = X[i, :real_len, 3]
        
        if len(closes) < 2:
            features.append([0]*6) 
            continue

        total_return = closes[-1] - closes[0]
        std_dev = np.std(closes)
        
        x_axis = np.arange(len(closes)).reshape(-1, 1)
        model = LinearRegression().fit(x_axis, closes)
        slope = model.coef_[0]
        
        price_range = np.max(closes) - np.min(closes)
        
        mid_point = len(closes) // 2
        first_half_return = closes[mid_point] - closes[0]
        second_half_return = closes[-1] - closes[mid_point]
        
        features.append([total_return, std_dev, slope, price_range, first_half_return, second_half_return])
        
    return np.array(features)

def run_baseline():
    train_data, test_data = load_data()
    if train_data is None: return

    X_train_raw, y_train, l_train = train_data
    X_test_raw, y_test, l_test = test_data

    X_train_feats = extract_features(X_train_raw, l_train)
    X_test_feats = extract_features(X_test_raw, l_test)
    
    X_train_feats = np.nan_to_num(X_train_feats)
    X_test_feats = np.nan_to_num(X_test_feats)

    print(f"\nBaseline Feature Matrix size: {X_train_feats.shape}")

    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_feats, y_train)

    print("\n--- BASELINE RESULTS (TEST SET) ---")
    y_pred = clf.predict(X_test_feats)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    
    target_names = list(config.LABEL_MAP.keys())
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nFeature Importance:")
    feature_names = ["Total Return", "Volatility (Std)", "Trend (Slope)", "Price Range", "1st Half Ret", "2nd Half Ret"]
    importances = clf.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, ax=ax)
    
    plt.title('Baseline Confusion Matrix (Test Set)')
    plt.tight_layout()
    
    save_path = os.path.join(config.RESULTS_DIR, 'baseline_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nBaseline confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    run_baseline()