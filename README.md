## Binary Classification with Logistic Regression

This script demonstrates how to perform binary classification using **Logistic Regression** with **scikit-learn**, applied to a dataset of social network advertisements. The goal is to predict whether a user purchases a product based on their age and estimated salary.

---

### 1. Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

* **NumPy**: For numerical array operations.
* **Matplotlib**: For data visualization.
* **Pandas**: For handling and manipulating structured data.

---

### 2. Importing the Dataset

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

* Loads the dataset from a CSV file.
* `X`: Feature matrix (input variables such as Age and Estimated Salary).
* `y`: Target vector (whether the user purchased the product, 0 or 1).

---

### 3. Splitting the Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

* Splits the dataset into:

  * 75% training data
  * 25% test data
* `random_state=0` ensures reproducibility.

---

### 4. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

* Standardizes features by removing the mean and scaling to unit variance.
* Important for distance-based algorithms and better model convergence.

---

### 5. Training the Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```

* Initializes and trains a **Logistic Regression** classifier using the training set.

---

### 6. Making Predictions

```python
print(classifier.predict(sc.transform([[30,87000]])))
```

* Predicts the outcome for a single new user with age 30 and salary 87,000.

```python
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

* Predicts results on the test set.
* Concatenates and prints predicted vs actual outcomes for comparison.

---

### 7. Evaluating the Model

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

* Computes the **confusion matrix** and **accuracy score** to evaluate performance.
* Confusion matrix shows True Positives, False Positives, True Negatives, and False Negatives.

---

### 8. Visualizing the Results (Training Set)

```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
# Create mesh grid
# Plot decision boundary
# Scatter plot of training points
```

* Plots the decision boundary learned by the model over the training data.
* Uses inverse-transformed features to display original scales (Age and Salary).

---

### 9. Visualizing the Results (Test Set)

```python
X_set, y_set = sc.inverse_transform(X_test), y_test
# Repeat mesh grid and plotting process
```

* Visualizes how well the classifier generalizes to unseen (test) data.
* Highlights model performance by showing predicted vs actual regions.
