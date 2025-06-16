## 💳 Credit Card Fraud Detection Using Logistic Regression

This project focuses on detecting fraudulent credit card transactions using machine learning techniques, specifically **Logistic Regression**. The dataset used is highly imbalanced, making this a great case study for handling such challenges using **resampling techniques** like SMOTE and **model evaluation metrics** beyond accuracy.

> 🚀 This project builds on the Kaggle notebook [Credit Fraud - Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets) by [@janiobachmann](https://www.kaggle.com/janiobachmann), with additional interpretation, improvements, and visualizations implemented in Python.

---

## 📂 Project Structure

```
├── file.ipynb                # Jupyter Notebook with full code and outputs
├── README.md                 # Project documentation (this file)
└── dataset/                  # (Optional) Directory to store the dataset locally
```

---

## 📊 Dataset

* **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size**: 284,807 transactions
* **Fraudulent Transactions**: 492 (\~0.17%)
* **Features**:

  * 28 anonymized PCA components: `V1`, `V2`, ..., `V28`
  * `Time` and `Amount`
  * `Class`: Target variable (`0` = Legit, `1` = Fraud)

---

## 🧠 Objectives

* Understand and preprocess the imbalanced dataset.
* Apply Logistic Regression for binary classification.
* Use **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.
* Evaluate model performance using appropriate metrics:

  * Confusion Matrix
  * Precision, Recall, F1-Score
  * ROC AUC Curve

---

## 🔍 Key Techniques Used

1. **Exploratory Data Analysis (EDA)**:

   * Visualizations using `matplotlib` and `seaborn`
   * Feature distribution comparison between fraud and non-fraud classes

2. **Data Preprocessing**:

   * Feature scaling using `StandardScaler`
   * Handling class imbalance with **SMOTE**

3. **Model Training**:

   * Logistic Regression with scikit-learn
   * Train-test split using `train_test_split`

4. **Evaluation**:

   * Confusion Matrix & Classification Report
   * ROC AUC score to measure the trade-off between recall and specificity

---

## 📈 Results

After applying SMOTE and training the model, the Logistic Regression classifier yielded the following results:

| Metric         | Score  |
| -------------- | ------ |
| Accuracy       | \~94%  |
| Recall (Fraud) | \~90%  |
| Precision      | \~99%  |
| F1 Score       | \~94%  |
| ROC AUC        | \~0.97 |

These results show a good balance between detecting actual fraud cases (high recall) and avoiding false alarms (high precision), which is critical in financial applications.

> 📌 Note: The notebook includes detailed visualizations like ROC curves, confusion matrices, and class distribution charts.

---

## 📷 Visualizations

The notebook contains various plots, including:

* Fraud vs Non-Fraud Transaction Distribution
* SMOTE Oversampling Effect
* ROC Curve for Model Evaluation
* Confusion Matrix Heatmap

---

## ⚙️ Technologies Used

* **Python 3**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**

---

## 📌 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/credit-card-fraud-logistic.git
   cd credit-card-fraud-logistic
   ```

2. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook file.ipynb
   ```


## 🙌 Acknowledgments
* Dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Logistic Regression explanation: [Scikit-learn documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)


