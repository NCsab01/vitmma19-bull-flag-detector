from utils import setup_logger
import config
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
import re
import requests
import zipfile
import shutil


logger = setup_logger()


def download_and_extract_sharepoint(sharepoint_url, target_dir):
    logger.info("Starting Data Update Process...")

    if "?download=1" not in sharepoint_url and "&download=1" not in sharepoint_url:
        if "?" in sharepoint_url:
            download_url = sharepoint_url + "&download=1"
        else:
            download_url = sharepoint_url + "?download=1"
    else:
        download_url = sharepoint_url

    temp_zip_path = "temp_data_download.zip"

    try:
        if os.path.exists(target_dir):
            logger.info(f"Removing old data directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        os.makedirs(target_dir)
        logger.info(f"Created fresh directory: {target_dir}")

        logger.info(f"Downloading ZIP from SharePoint...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
        logger.info("Download complete. Extracting...")

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        logger.info(f"SUCCESS: Data updated in '{target_dir}'")

    except Exception as e:
        logger.error(f"Failed to download/extract data: {e}")
        return False
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
            
    return True


def load_df_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        if pd.api.types.is_numeric_dtype(df['timestamp']):
            first_row_timestamp = df.iloc[0]['timestamp']
            
            unit = 'ms' if first_row_timestamp > 1e11 else 's'
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
            
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
 
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file ({file_path}): {e}")
        return None


def get_labeling_info_from_json(json_file_full_path):
    try:
        with open(json_file_full_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except Exception as e:
        logger.error(f"Error reading JSON file ({json_file_full_path}): {e}")
        return {}

    labels = {}
    
    if isinstance(data, list):
        for item in data:
            file_path_in_json = item.get('file_upload')
            annotations = item.get('annotations')

            if file_path_in_json and annotations:
                clean_filename = os.path.basename(file_path_in_json)
                if '-' in clean_filename:
                    clean_filename = clean_filename.split('-', 1)[-1]

                if clean_filename not in labels:
                    labels[clean_filename] = []

                try:
                    result_list = annotations[0].get('result', [])
                    
                    for result_item in result_list:
                        value = result_item.get('value', {})
                        timeserieslabels = value.get('timeserieslabels')
                        start_time = value.get('start')
                        end_time = value.get('end')

                        if timeserieslabels and len(timeserieslabels) > 0:
                            label_string = timeserieslabels[0]
                            labels[clean_filename].append({
                                    'id': config.LABEL_MAP[label_string],
                                    'name': label_string,
                                    'start': start_time,
                                    'end': end_time
                                })
                except Exception:
                    continue
                
    return labels


def convert_to_numpy_padded(sequence_list, max_length):
    padded_sequences = []
    for sequence in sequence_list:
        padding_length = max_length - len(sequence)
        if padding_length < 0: 
            padded_sequence = sequence[:max_length]
        else:
            padded_sequence = np.pad(sequence, ((0, padding_length), (0, 0)), 'constant', constant_values=0.0)
        padded_sequences.append(padded_sequence)
    return np.stack(padded_sequences, axis=0)


def preprocess():
    logger.info("Data processing pipeline started...")
    
    root_directory = config.RAW_DATA_ROOT_DIR
        
    download_and_extract_sharepoint(config.SHAREPOINT_DATA_URL, root_directory)
    
    if not os.path.exists(root_directory):
        logger.error(f"Directory not found: {root_directory}")
        return

    all_sequences = []
    all_labels = []
    all_lengths = []
    max_sequence_length = -1
    
    total_segments_extracted = 0

    logger.info(f"Scanning directory structure: {root_directory}")

    for root, directories, files in os.walk(root_directory):
        json_files = [filename for filename in files if filename.endswith('.json')]
        
        if not json_files:
            continue
            
        json_file_full_path = os.path.join(root, json_files[0])
        logger.info(f"Processing folder: {root}")
        
        labels = get_labeling_info_from_json(json_file_full_path)
        
        if not labels:
            continue

        csv_files = [filename for filename in files if filename.endswith('.csv')]
        
        for csv_file in csv_files:
            clean_filename = csv_file
            
            if '-' in csv_file:
                 possible_clean_name = csv_file.split('-', 1)[-1]
                 if possible_clean_name in labels:
                     clean_filename = possible_clean_name
            
            if clean_filename in labels:
                annotations = labels[clean_filename]
                full_csv_path = os.path.join(root, csv_file)
                
                df = load_df_from_csv(full_csv_path)
                
                if df is None:
                    continue
                
                if len(annotations) == 1:
                    annotation = annotations[0]
                    ohlc_data = df.loc[:, ['open', 'high', 'low', 'close']].values
                    
                    if len(ohlc_data) > 0:
                        min_val = np.min(ohlc_data)
                        max_val = np.max(ohlc_data)
                        
                        if (max_val - min_val) == 0:
                            ohlc_data = ohlc_data - min_val
                        else:
                            ohlc_data = (ohlc_data - min_val) / (max_val - min_val)
                        
                        if(ohlc_data.shape[0] < config.MIN_SEQUENCE_LENGTH or ohlc_data.shape[0] > config.MAX_SEQUENCE_LENGTH):
                            logger.warning(f"Skipping file due to invalid length ({ohlc_data.shape[0]}): {csv_file} ({annotation['name']})")
                            continue
                        
                        all_sequences.append(ohlc_data)
                        all_lengths.append(len(ohlc_data))
                        all_labels.append(annotation['id'])
                        
                        if len(ohlc_data) > max_sequence_length:
                            max_sequence_length = len(ohlc_data)
                        
                        logger.info(f"Extracted full file: {csv_file} ({annotation['name']})")
                        total_segments_extracted += 1

                else:
                    logger.info(f"Multi-label file: {csv_file} ({len(annotations)} segments)")
                    
                    for annotation in annotations:
                        timestamp_pattern = r"^\d{10,14}$"          
                        if re.match(timestamp_pattern, str(annotation['start'])):
                            start_timestamp_annotation = int(annotation['start'])
                            end_timestamp_annotation = int(annotation['end'])
                            unit = 'ms' if start_timestamp_annotation > 1e11 else 's'
                            start_timestamp = pd.to_datetime(start_timestamp_annotation, unit=unit, utc=True)
                            end_timestamp = pd.to_datetime(end_timestamp_annotation, unit=unit, utc=True)
                        else:
                            start_timestamp = pd.to_datetime(annotation['start'], utc=True)
                            end_timestamp = pd.to_datetime(annotation['end'], utc=True)
                            
                        df_slice = df.loc[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
                        
                        if not df_slice.empty:
                            ohlc_slice = df_slice.loc[:, ['open', 'high', 'low', 'close']].values
                            
                            if len(ohlc_slice) > 0:
                                min_val = np.min(ohlc_slice)
                                max_val = np.max(ohlc_slice)
                                
                                if (max_val - min_val) == 0:
                                    ohlc_slice = ohlc_slice - min_val
                                else:
                                    ohlc_slice = (ohlc_slice - min_val) / (max_val - min_val)
                            else:
                                continue
                            
                            if(ohlc_slice.shape[0] < config.MIN_SEQUENCE_LENGTH or ohlc_slice.shape[0] > config.MAX_SEQUENCE_LENGTH):
                                logger.warning(f"Skipping segment due to invalid length ({ohlc_slice.shape[0]}): {csv_file} ({annotation['name']})")
                                continue
                            
                            all_sequences.append(ohlc_slice)
                            all_lengths.append(len(ohlc_slice))
                            all_labels.append(annotation['id'])
                            
                            if len(ohlc_slice) > max_sequence_length:
                                max_sequence_length = len(ohlc_slice)
                            
                            total_segments_extracted += 1
                        else:
                            logger.warning(f"No data found for segment: {annotation['name']}")

    if not all_sequences:
        logger.error("No valid data segments found. Preprocessing aborted.")
        return

    X_all = convert_to_numpy_padded(all_sequences, max_sequence_length)
    Y_all = np.array(all_labels, dtype=np.int64)
    L_all = np.array(all_lengths, dtype=np.int64)
    
    X_train, X_temp, Y_train, Y_temp, L_train, L_temp = train_test_split(
        X_all, Y_all, L_all, test_size=0.3, random_state=42
    )

    X_val, X_test, Y_val, Y_test, L_val, L_test = train_test_split(
        X_temp, Y_temp, L_temp, test_size=0.5, random_state=42
    )

    logger.info("Saving processed datasets to disk...")
    
    if not os.path.exists(config.TRAIN_DATA_DIR):
        os.makedirs(config.TRAIN_DATA_DIR)
        
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.X_TRAIN_NAME), X_train)
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.Y_TRAIN_NAME), Y_train)
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.L_TRAIN_NAME), L_train)
    
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.X_VAL_NAME), X_val)
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.Y_VAL_NAME), Y_val)
    np.save(os.path.join(config.TRAIN_DATA_DIR, config.L_VAL_NAME), L_val)

    if not os.path.exists(config.TEST_DATA_DIR):
        os.makedirs(config.TEST_DATA_DIR)

    np.save(os.path.join(config.TEST_DATA_DIR, config.X_TEST_NAME), X_test)
    np.save(os.path.join(config.TEST_DATA_DIR, config.Y_TEST_NAME), Y_test)
    np.save(os.path.join(config.TEST_DATA_DIR, config.L_TEST_NAME), L_test)
    
    logger.info("Data processing finished successfully.")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"MIN LEN: {L_all.min()}")
    logger.info(f"MAX LEN: {L_all.max()}")
    logger.info(f"AVG LEN: {int(np.mean(L_all))}")

if __name__ == "__main__":
    preprocess()