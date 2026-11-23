# Machine Learning Assignment 2
## Bank Marketing Subscription Prediction

**Student ID:** 2026CT05204  
**GitHub repository:** https://github.com/2026ct05204/ML_Assignment2_2026CT05204

---

## 1. Problem Statement

The objective of this project is to build and compare machine learning classification models that predict whether a bank customer subscribes to a term deposit. The target variable is `deposit`, with two possible classes: `yes` and `no`.

This is a binary classification problem using customer details and bank marketing campaign attributes.

## 2. Dataset Description

- **Dataset:** Bank Marketing Dataset
- **Source:** Kaggle – https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
- **Local dataset file:** `dataset/bank.csv`
- **Records:** 11,162
- **Columns:** 17
- **Target variable:** `deposit`
- **Target classes:** `yes` and `no`

This dataset was selected because it provides a real world binary classification problem involving customer subscription prediction. It contains both numerical and categorical attributes allowing the implementation and comparison of different preprocessing techniques and classification algorithms. It also contains more than 1000 records and sufficient predictor features to satisfy the requirements.

### Target distribution

The latest successful execution reported the following distribution:

| Target class | Records | Percentage |
|---|---:|---:|
| no | 5,873 | 52.62% |
| yes | 5,289 | 47.38% |
| **Total** | **11,162** | **100%** |

## 3. Data Preprocessing

The implementation performs the following steps:

1. Loads the CSV dataset and displays its dimensions, first five rows and data types.
2. Checks missing values and duplicate records.
3. Removes exact duplicate rows. The latest execution reported zero duplicate rows.
4. Excludes the `duration` feature from model training because call duration is only known after the marketing call and would not be available before contact.
5. Treats the categorical value `unknown` as unavailable information.
6. Imputes missing numerical values using the median.
7. Imputes missing categorical values using the most frequent category.
8. Applies one-hot encoding to categorical features.
9. Applies standard scaling to numerical features.
10. Splits the data into training and testing sets using an 80:20 ratio.
11. Uses stratification to preserve the target-class proportions in both sets.
12. Uses random state `42` for reproducibility.

The preprocessing operations are included in Scikit-learn pipelines so that the imputers, encoder and scaler are fitted as part of model training using the training data.

### Features used

The latest execution identified six numerical features:

- `age`
- `balance`
- `day`
- `campaign`
- `pdays`
- `previous`

It identified nine categorical features:

- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`
- `contact`
- `month`
- `poutcome`

The `duration` field is excluded, leaving 15 predictor features before categorical encoding.

### Train-test split

| Set | Records |
|---|---:|
| Training | 8,929 |
| Testing | 2,233 |
| **Total** | **11,162** |

The latest execution showed the same target proportions in the training and testing sets: 52.62% class `no` and 47.38% class `yes`.

## 4. Machine Learning Models

Five classification algorithms are implemented in the single Python source file `model/ML_Assignment_2.py`:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Each model is trained and evaluated using the same train-test split and preprocessing approach.

## 5. Model Configuration

Logistic Regression: 
    Max Iterations:1500
    Class weight:Balanced
    Random State: 42

    Logistic Regression is used as baseline classification algorithm. Balanced class weights help help account for differencies in class frequencies

Decision Tree Classifier
    Max Depth: 8
    Class weight: Balanced
    Random weight: 42

    Limiting tree depth helps control model complexity and reduce overfitting

KNN
    Number of neighbours:7

    Seven neighbours selected to consider nearby observations when making predictions

Gaussian Naive Bayes
    Configuration: Scikit-learn default parameters

    Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes theorem. It assumes that numerical features follow Gaussian distribution within each class. 

Random Forest Classifier
    Number of estimators: 200
    Max Depth:200
    Class Weight:Balanced
    Random state:42
    
    Random forest combines multiple decision trees to improve predictive performance and reduce the limitation of an individual decision tree. 

## 6. Evaluation Metrics

The models are compared using six evaluation metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **ROC AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** A balanced measure of classification quality that considers all four confusion-matrix outcomes.

## 7. Model Comparison Results

The following values are from the latest successful execution in the BITS Virtual Lab on 23 November 2026. Values are rounded to four decimal places from the terminal output.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6928 | 0.7406 | 0.7123 | 0.5898 | 0.6453 | 0.3838 |
| Decision Tree | 0.6852 | 0.7254 | 0.7822 | 0.4650 | 0.5833 | 0.3868 |
| KNN | 0.6753 | 0.7168 | 0.6907 | 0.5699 | 0.6245 | 0.3481 |
| Gaussian Naive Bayes | 0.6713 | 0.7203 | 0.7580 | 0.4499 | 0.5647 | 0.3560 |
| Random Forest | **0.7197** | **0.7680** | 0.7707 | 0.5813 | **0.6627** | **0.4434** |

### Observations

- **Random Forest** achieved the highest accuracy (0.7197), ROC AUC (0.7680), F1 score (0.6627) and MCC (0.4434) in this run. It showed the strongest overall balance across these reported metrics.
- **Decision Tree** achieved the highest precision (0.7822), but its recall was 0.4650.
- **Logistic Regression** achieved the highest recall (0.5898) among the five models.
- **KNN** achieved an accuracy of 0.6753 and an F1 score of 0.6245.
- **Gaussian Naive Bayes** achieved an accuracy of 0.6713 and an AUC of 0.7203.

### Overall result

Based on the latest run, Random Forest is the strongest overall model when considering accuracy, AUC, F1 score and MCC together. Decision Tree has the highest precision, while Logistic Regression has the highest recall. The most suitable model can depend on whether the application places greater importance on precision or recall.

## 8. Graphical Comparison

The program generates the following visualizations in the `results/` directory:

| File | Description |
|---|---|
| `model_comparison.png` | Combined comparison of the model evaluation metrics |
| `accuracy_comparison.png` | Accuracy comparison |
| `auc_comparison.png` | ROC AUC comparison |
| `precision_comparison.png` | Precision comparison |
| `recall_comparison.png` | Recall comparison |
| `f1_comparison.png` | F1 score comparison |
| `mcc_comparison.png` | Matthews Correlation Coefficient comparison |

The numerical comparison is also saved as `results/model_comparison.csv`.

## 9. Saved Models and Output Files

The program saves the five fitted model pipelines as Joblib files in the `model/` directory:

- `logistic_regression.joblib`
- `decision_tree.joblib`
- `knn.joblib`
- `gaussian_naive_bayes.joblib`
- `random_forest.joblib`

The held-out test data is saved as `test_data.csv` in the project root.

## 10. Project Structure

The project is organised as follows:

```text
ML_2_2026CT05204/
|
|-- dataset/
|   |-- bank.csv
|
|-- model/
|   |-- ML_Assignment_2.py
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- gaussian_naive_bayes.joblib
|   |-- random_forest.joblib
|
|-- results/
|   |-- model_comparison.csv
|   |-- model_comparison.png
|   |-- accuracy_comparison.png
|   |-- auc_comparison.png
|   |-- precision_comparison.png
|   |-- recall_comparison.png
|   |-- f1_comparison.png
|   |-- mcc_comparison.png
|
|-- test_data.csv
|-- requirements.txt
|-- README.md
```

`model/ML_Assignment_2.py` is the single source-code file containing all five model implementations, preprocessing, training, evaluation and output-generation steps. The `.joblib` files are the saved fitted model pipelines.

## 11. Software Requirements

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file contain the dependencies and compatible version constraints used by this project.

## 12. How to Run the Project

1. Clone or download the GitHub repository.
2. Open the project root folder in Visual Studio Code.
3. Ensure the dataset is available at `dataset/bank.csv`.
4. Open the VS Code terminal in the project root.
5. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

6. Run the assignment implementation:

```bash
python model/ML_Assignment_2.py
```

The script loads the dataset, preprocesses the data, trains and evaluates the five classifiers, and generates the comparison CSV, graphs, test data and saved model files in the project directories.

## 12. Github Repository Link
Repository: https://github.com/2026ct05204/ML_Assignment2_2026CT05204

The Repository contains:
    - Complete python source code
    - Saved trained models
    - Dataset
    - requirements.txt
    - bank.csv
    - README.md
    - Evaluation results and graph


## 13. Conclusion

This project demonstrates a complete binary classification workflow using the Bank Marketing dataset. Five machine learning algorithms were implemented and compared using accuracy, ROC AUC, precision, recall, F1 score and MCC.

In the latest recorded execution, Random Forest achieved the strongest overall results across accuracy, AUC, F1 score and MCC. The experiment also shows why multiple evaluation metrics are useful when comparing classifiers, rather than relying on accuracy alone.

---

**GitHub repository:** https://github.com/2026ct05204/ML_Assignment2_2026CT05204
