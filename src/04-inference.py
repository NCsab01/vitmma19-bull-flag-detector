import torch
import numpy as np
import pandas as pd
import os
import config
from model_defs import create_model

def load_and_preprocess_csv(file_path, max_length):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing columns in {file_path}")
            return None

        ohlc_data = df[required_cols].values 
        
        if len(ohlc_data) == 0:
            return None

        min_val = np.min(ohlc_data)
        max_val = np.max(ohlc_data)
        
        if (max_val - min_val) == 0:
            ohlc_data = ohlc_data - min_val
        else:
            ohlc_data = (ohlc_data - min_val) / (max_val - min_val)

        current_len = len(ohlc_data)
        
        if current_len > max_length:
            ohlc_data = ohlc_data[:max_length]
        elif current_len < max_length:
            padding_len = max_length - current_len
            ohlc_data = np.pad(ohlc_data, ((0, padding_len), (0, 0)), 'constant', constant_values=0)

        tensor = torch.tensor(ohlc_data, dtype=torch.float32).unsqueeze(0)
        return tensor

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def run_inference():
    inference_dir = config.INFERENCE_DIR
    results_dir = config.RESULTS_DIR

    print(f"Starting inference...")
    print(f"Reading from: {inference_dir}")
    
    if not os.path.exists(inference_dir):
        print(f"Error: Inference directory not found: {inference_dir}")
        return

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(device)
    model_path = os.path.join(config.MODEL_PARAMETERS_DIR, config.MODEL_PARAMETERS_PATH)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Run training first.")
        return

    model.eval() 
    id_to_label = {v: k for k, v in config.LABEL_MAP.items()}

    files = [f for f in os.listdir(inference_dir) if f.endswith('.csv')]
    
    if not files:
        print("No CSV files found in inference directory.")
        return

    print(f"\nFound {len(files)} files.\n")
    print(f"{'FILE NAME':<40} | {'PREDICTION':<25} | {'CONFIDENCE'}")
    print("-" * 80)

    results_txt_path = os.path.join(results_dir, "inference_results.txt")
    
    with open(results_txt_path, "w") as f_out:
        f_out.write(f"{'FILE NAME':<40} | {'PREDICTION':<25} | {'CONFIDENCE'}\n")
        f_out.write("-" * 80 + "\n")

        with torch.no_grad():
            for file_name in files:
                full_path = os.path.join(inference_dir, file_name)
                
                input_tensor = load_and_preprocess_csv(full_path, config.MAX_SEQUENCE_LENGTH)
                
                if input_tensor is None:
                    continue

                input_tensor = input_tensor.to(device)

                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
                confidence, predicted_idx = torch.max(probs, 1)
                
                pred_idx_val = predicted_idx.item()
                conf_val = confidence.item() * 100
                
                pred_label = id_to_label.get(pred_idx_val, "Unknown")
                
                output_line = f"{file_name:<40} | {pred_label:<25} | {conf_val:.2f}%"
                print(output_line)
                f_out.write(output_line + "\n")

    print(f"\nInference finished. Results saved to: {results_txt_path}")

if __name__ == "__main__":
    run_inference()