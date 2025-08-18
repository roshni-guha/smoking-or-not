# Smoking Prediction using Machine Learning

This project explores different machine learning models to predict whether a person is a smoker or not based on health-related features.  
It includes data preprocessing, visualization, model comparison, and evaluation on test data.

---

## Dataset
- Source: `smoking.csv` (plus `X_train.csv`, `X_test.csv`, `Y_train.csv`, `Y_test.csv`)
- Features include:
  - `gender` (M/F → converted to 1/0)
  - `oral` (Y/N → converted to 1/0)
  - `tartar` (Y/N → converted to 1/0)
  - Additional continuous health metrics (e.g., age, height, weight, cholesterol, etc.)
- Target:  
  - `smoking` (0 = Non-smoker, 1 = Smoker)

---

## Preprocessing
1. Converted categorical variables (`gender`, `oral`, `tartar`) to numeric values.
2. Set `ID` as the dataframe index.
3. Handled missing values (none found).
4. Normalized/standardized features using `StandardScaler` for models sensitive to scale (KNN, SVM).
5. Split dataset into **train/test** using provided CSVs.

---

## Data Visualization
- Distribution of smokers vs non-smokers (`plt.hist`).
- Density plots for continuous variables.
- Correlation heatmap of all features.

---

## Models Tested
The following models were trained and evaluated with **10-fold cross-validation**:

- Decision Tree Classifier (CART)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (NB)
- Support Vector Machine (SVM)

Both **raw data** and **scaled data** were tested.

---

## Results

### Without Scaling
| Model | Mean Accuracy | Std Dev | Run Time |
|-------|--------------|---------|----------|
| CART  | ~0.686 | 0.009 | ~3.2s |
| KNN   | ~0.688 | 0.008 | ~1.7s |
| NB    | ~0.703 | 0.005 | ~0.1s |
| SVM   | ~0.729 | 0.008 | ~203s |

**SVM performed best on unscaled data.**

---

### With Scaling (StandardScaler)
| Model       | Mean Accuracy | Std Dev | Run Time |
|-------------|--------------|---------|----------|
| Scaled CART | ~0.685 | 0.006 | ~5.8s |
| Scaled NB   | ~0.703 | 0.005 | ~0.2s |
| Scaled KNN  | ~0.711 | 0.005 | ~3.4s |
| Scaled SVM  | ~0.758 | 0.005 | ~376s |

**Scaled SVM achieved the best performance (≈75.8% accuracy).**

---

## Final Model
- **Algorithm:** Support Vector Machine (SVM) with scaling
- **Training time:** ~40s
- **Test Accuracy:** Reported via `accuracy_score`
- **Evaluation Metrics:**
  - Confusion matrix
  - Classification report

---

## Dependencies
- Python 3.12+
- pandas
- matplotlib
- scikit-learn
- numpy

Install all dependencies via:

```bash
pip install -r requirements.txt
