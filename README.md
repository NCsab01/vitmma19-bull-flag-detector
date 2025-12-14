# Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Nagy Csaba Dániel
- **Aiming for +1 Mark**: No

### Solution Description

### 1. Problem Definition
Technical analysis relies on identifying geometric chart patterns (e.g., Flags, Pennants, Wedges) to forecast market movements. Manual detection is subjective and non-scalable. This project automates the classification of these complex patterns within noisy financial time-series data using a **1D Convolutional Neural Network (1D-CNN)**. The task is defined as a multi-class classification problem where the input is a 1D sequence of historical price data.

### 2. Model Architecture
A **1D-CNN** architecture was selected over traditional RNNs due to its efficiency in detecting local substructures (e.g., sudden drops, consolidations) regardless of their temporal position.

* **Input Layer:** Accepts normalized time-series sequences of fixed length.
* **Feature Extraction:** Utilizes multiple 1D Convolutional layers to capture hierarchical features. Batch Normalization and ReLU activation are applied to ensure training stability and non-linearity.
* **Classification Head:** A Fully Connected (Dense) layer maps extracted features to class probabilities via a Softmax activation function.

### 3. Training Methodology
A rigorous pipeline was implemented to address data quality challenges:

* **Data Preprocessing:** Raw price data was normalized (Z-score/MinMax scaling) to facilitate convergence.
* **Curated Data Strategy:** Exploratory analysis revealed significant label noise. A "Quality over Quantity" approach was adopted, using a strictly curated, manually verified subset for training and validation.
* **Baseline Comparison:** A **Random Forest** classifier was trained on statistical features (volatility, trend slope, total return) to establish a performance benchmark.
* **Optimization:** The DL model was trained using the **Adam optimizer** and **Cross-Entropy Loss**, with early stopping to prevent overfitting on the limited dataset.

---

## Evaluation and Results

Evaluation was conducted on the high-quality curated subset to ensure validity. The proposed Deep Learning model is compared against the Random Forest baseline.

### 1. Baseline Performance (Random Forest)
The baseline model, operating on extracted statistical features, achieved an overall **Accuracy of 23%**.

* **Feature Importance:** The analysis highlighted **Total Return (0.197)**, **1st Half Return (0.196)**, and **Price Range (0.180)** as the most critical indicators.
* **Limitations:** The model struggled to generalize (Weighted F1-score: 0.20). While it identified some *Bearish Normal* patterns (Precision: 0.40), it failed to detect complex structures like Wedges.

### 2. Deep Learning Model Performance
The proposed 1D-CNN demonstrated significant improvement over the baseline.

* **Overall Accuracy:** Achieved **38%**, outperforming the baseline by **15 percentage points**.
* **Class-Specific Behavior:** The model showed strong sensitivity to the **Bearish Normal** class, achieving a **Recall of 1.00** (100% detection rate) and an F1-score of 0.56.
* **Trade-offs:** Due to the small dataset size, the model exhibited bias towards the dominant class, resulting in low precision for underrepresented classes (e.g., Bullish Normal).

### 3. Summary
On the curated dataset, the **Deep Learning pipeline successfully outperforms the Random Forest baseline**. However, results indicate high variance due to data scarcity, with accuracy fluctuating between **0.22 and 0.46** across runs. While the current model demonstrates strong capability in detecting bearish trends, further data augmentation or a larger clean dataset is recommended to improve performance on minority classes.

**Performance Metrics:**

* **Baseline (Random Forest):** Achieved an accuracy of **0.23**.
* **Proposed Model:** Achieved an accuracy of **0.38** in the current run.

**Analysis:**

On this curated dataset, the proposed model successfully outperforms the baseline. However, it is important to note that due to the limited size of the selected subset, the training process exhibits high variance. The model's accuracy typically fluctuates between **0.22 and 0.46** across different runs.

While the model shows promise on clean data, preliminary tests suggest that on the larger, noisier dataset, the performance gap between the baseline and the deep learning model narrows significantly, often yielding similar results.

### Data Preparation and Configuration

To prepare the dataset for the pipeline, the training data must be organized into subdirectories where each folder contains the relevant time-series CSV files along with a single JSON file containing the corresponding labels. Additionally, a separate folder named `inference` should be created to store the unlabeled CSV files intended for prediction. Once this directory structure is established, the entire collection must be compressed into a single archive named `data.zip`. This archive should then be uploaded to SharePoint, ensuring that the file is shared with permissions that allow access. Finally, generate a shareable link to the uploaded file and update the `src/config.py` script by pasting this URL into the appropriate configuration variable, which allows the application to automatically download and utilize the dataset.

### Extra Credit Justification

I didn't choose the extra task.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

*   You can mount an arbitrary local directory to the container when running the application. It is not necessary to manually place any input data into this folder, as the system is designed to automatically download the required dataset from SharePoint upon initialization. The primary purpose of mounting a volume is to retrieve the output; all results generated during the process including logs, trained models, and inference predictions will be saved to this directory, allowing you to access the files directly on your host machine.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `05-baseline-model.py`: Trains a Random Forest baseline model.
    - `model_defs.py`: Model architecture and Dataset definitions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `.gitignore`: Standard Python gitignore to exclude unnecessary files.
    - `LICENSE`: MIT License.
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
