# TPC (Two-Phase Classification)

## 📌 Overview

This project implements a **two-stage classification framework** for handling **imbalanced datasets**, based on the following research paper:

> 📄 *A Novel Classification Method for Imbalanced Data Using Two-Phase Cost-Sensitive SVM and Genetic Algorithm*
> 🔗 https://www.mdpi.com/2073-8994/14/3/567 

The method integrates:

* Cost-sensitive Support Vector Machine (SVM)
* Genetic Algorithm (GA) for hyperparameter optimization

This project focuses on **method implementation and understanding**, rather than exact reproduction of experimental results.

---

## 📖 Paper Contribution

The key contributions of the paper include:

* 🔹 A **two-stage learning framework** for imbalanced data
* 🔹 Use of **cost-sensitive SVM** to address class imbalance
* 🔹 Integration of **Genetic Algorithm (GA)** for parameter tuning
* 🔹 A general pipeline applicable to imbalanced learning tasks


---

## 📊 Architecture

![TPC Flowchart](images/tpc_flow.png)

*Figure: Two-stage GA + SVM framework for imbalanced classification*

---

## 🧠 Methodology

### 🔹 Stage 1: Data Rebalancing

* Model: `LinearSVC`
* Uses **asymmetric cost-sensitive SVM**
* Adjusts:

  * Gamma
  * Cost_min (minority)
  * Cost_maj (majority)
* Output:

  * Balanced dataset

---

### 🔹 Stage 2: Classification

* Model: `SVC (RBF kernel)`
* Uses balanced dataset from Stage 1
* GA optimizes:

  * Gamma
  * Cost
* Output:

  * Final predictions

---

## ⚙️ Features

* Handles **imbalanced classification problems**
* GA optimization:

  * Selection
  * Crossover
  * Mutation
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F-measure
  * G-mean
  * AUC

---

## 📂 Dataset

### 📊 Dataset Source

* Yelp Review Dataset
  🔗 https://www.yelp.com/dataset

---

### 📊 Dataset Construction (from paper)

| Dataset | Total  | Ratio | Minority | Majority |
| ------- | ------ | ----- | -------- | -------- |
| Yelp_α  | 14,927 | 1:10  | 1,300    | 13,627   |
| Yelp_β  | 16,227 | 1:5   | 2,600    | 13,627   |
| Yelp_γ  | 18,169 | 1:3   | 4,542    | 13,627   |

📌 Label definition:

* 1–2 ⭐ → Negative (minority)
* 4–5 ⭐ → Positive (majority)
* 3 ⭐ → Removed

---

## 📈 Results (from paper)

The performance of the proposed method (**TPC**) is evaluated using multiple metrics derived from the confusion matrix:

* Accuracy
* Precision
* Recall
* F-measure
* Specificity
* G-mean
* AUC (Area Under Curve)

---

### 🔹 Overall Performance

Across all datasets (Yelp_α, Yelp_β, Yelp_γ):

* Accuracy: **> 90%**
* AUC: **> 85%**
* F-measure: consistently higher than baseline methods
* Balanced performance across minority and majority classes

📌 TPC outperforms:

* RBF-SVM
* Cost-sensitive SVM
* Random Forest
* SMOTE-SVM / B1-SMOTE-SVM
* Easy Ensemble Classifier (EEC)

---

### 🔹 Detailed Metric Analysis

#### ✅ Accuracy

* Measures overall classification performance
* TPC achieves the **highest accuracy among all methods**
* Most methods exceed 90%, but TPC is consistently best

---

#### ✅ Precision

* Measures correctness of positive predictions
* TPC maintains **stable precision (~balanced range)**
* Avoids excessive false positives (common in imbalanced learning)

---

#### ✅ Recall

* Measures ability to detect minority class
* TPC significantly improves recall
* Ensures minority class is **not ignored**

---

#### ✅ F-measure

* Harmonic mean of Precision and Recall
* TPC shows **strong robustness**, especially under high imbalance
* Indicates balanced improvement across both classes

---

#### ✅ Specificity

* Measures correct classification of majority class
* TPC maintains high specificity
* Avoids sacrificing majority class performance

---

#### ✅ G-mean

* Balance between Recall and Specificity
* TPC achieves **high and stable G-mean**
* Demonstrates balanced classification ability

---

#### ✅ AUC

* Measures model discrimination ability
* TPC achieves **AUC > 0.85**
* Comparable to top-performing methods (SMOTE, EEC)
* Indicates strong ability to distinguish classes

---

### 🔹 Key Insights

* ✔ TPC improves **both minority and majority classification**
* ✔ Avoids common problems:

  * Overfitting (SMOTE)
  * Information loss (undersampling)
* ✔ Provides **balanced and robust performance across all metrics**

---

### 📌 Summary

| Metric      | TPC Performance            |
| ----------- | -------------------------- |
| Accuracy    | ⭐ Excellent (>90%)         |
| Precision   | ⭐ Stable                   |
| Recall      | ⭐ High (minority improved) |
| F-measure   | ⭐ Strong                   |
| Specificity | ⭐ High                     |
| G-mean      | ⭐ Balanced                 |
| AUC         | ⭐ >85%                     |

---

📖 These results demonstrate that TPC is a **robust and balanced solution for imbalanced classification problems**.

---

## 🚀 How to Run

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
python TPC.py
```

---

## 🧩 Notes

* Research-oriented implementation
* GA parameters affect performance
* Dataset preprocessing required

---

## 👨‍💻 Author

* Wen-Yen(Hank) Hsu
