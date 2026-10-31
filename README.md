# 🧭 The Foundations of a Machine-Learning Project

> **Good models start before training—with useful data, sound evaluation, and honest testing.**

## 🗺️ Trail Map

- [The Machine-Learning Workflow](#-the-machine-learning-workflow)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering and Selection](#-feature-engineering-and-selection)
- [Training, Validation, and Test Data](#-training-validation-and-test-data)
- [Overfitting, Underfitting, and Regularization](#️-overfitting-underfitting-and-regularization)
- [Cross-Validation and Hyperparameter Tuning](#-cross-validation-and-hyperparameter-tuning)
- [A Practical Python Example](#-a-practical-python-example)
- [Common Pitfalls](#️-common-pitfalls)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔄 The Machine-Learning Workflow

A machine-learning project is more than choosing an algorithm and calling `.fit()`. A typical workflow looks like this:

1. Define the problem and the decision the model should support.
2. Inspect the data and understand how it was collected.
3. Split the data in a way that reflects how the model will be used.
4. Prepare features using information from the training data.
5. Train a baseline model.
6. Use validation or cross-validation to compare approaches.
7. Evaluate the chosen approach on held-out test data.
8. Review errors, limitations, and deployment risks.

These steps are connected. For example, preprocessing choices should be evaluated as part of the model pipeline, and the data split should reflect whether the task involves time, groups, or future predictions.

## 🧹 Data Preprocessing

Real-world data often arrives with missing values, inconsistent formats, repeated records, or categories that algorithms cannot use directly. **Preprocessing** turns raw data into a representation suitable for a model.

### Start by inspecting the data

Before changing anything, check:

- Number of rows and columns
- Data types and value ranges
- Missing values and duplicate records
- Category frequencies
- Target-label distribution, if there is a label
- Outliers and unexpected values
- How and when each field was collected

Do not remove or replace data automatically without understanding what it means. A value that appears unusual may be a genuine case rather than an error.

### Handling missing values

Common options include:

- **Remove rows or columns:** Reasonable when only a small amount of information is missing and removing it will not distort the data.
- **Impute numerical values:** Use a statistic such as the median or mean, selected based on the feature and data distribution.
- **Impute categorical values:** Use the most frequent category or a deliberate “unknown” category.
- **Add a missingness indicator:** Useful when the fact that a value is missing may itself carry information.

The imputation method should be learned from training data only. Scikit-learn provides tools such as `SimpleImputer` for common imputation strategies. ([scikit-learn: Imputation](https://scikit-learn.org/stable/modules/impute.html))

### Encoding categorical features

Many machine-learning algorithms need numerical inputs, so text categories often need to be encoded.

- **One-hot encoding:** Creates a separate binary feature for each category. It is a common choice when categories have no meaningful order.
- **Ordinal encoding:** Maps categories to numbers. Use it only when the order is meaningful or when the model can appropriately handle the representation.
- **Other encodings:** High-cardinality categories may need approaches designed for many distinct values. Choose carefully and fit them without leaking target information.

Do not assign arbitrary numbers to unordered categories and assume the model will treat them as equivalent. For example, encoding `red = 1`, `blue = 2`, and `green = 3` can create a false ordering.

### Scaling numerical features

Some models are sensitive to feature scale. Standardization or another scaling method can be useful for methods such as logistic regression, SVMs, KNN, and PCA. Tree-based models often need less scaling.

Fit the scaler using training data, then use that same fitted scaler to transform validation and test data. Scikit-learn’s preprocessing tools provide standardization, normalization, encoding, and other transformations. ([scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html))

## 🧱 Feature Engineering and Selection

### Feature engineering

**Feature engineering** means creating or transforming input variables so they better represent the problem.

Examples include:

- Extracting hour, day of week, or month from a timestamp
- Calculating time since a previous event
- Combining related measurements into a meaningful ratio
- Converting text into numerical representations
- Grouping rare categories where appropriate

A useful feature must be available at the time a prediction is made. A feature calculated using future information can cause data leakage, even if it appears predictive during training.

### Feature selection

**Feature selection** means choosing which available features to use. Reasons to remove a feature may include:

- It leaks the target or future information.
- It is an identifier that encourages memorization.
- It is mostly missing or unreliable.
- It duplicates other information.
- It is not available in the intended production setting.
- It creates privacy, fairness, or operational concerns.

Feature selection can reduce complexity, but removing features solely because they seem unimportant in one experiment can be risky. Evaluate selection inside the training process, not by repeatedly inspecting the final test set.

### Features are not explanations

A feature’s usefulness for prediction does not show that it causes the outcome. Correlation, predictive importance, and causal effect are different ideas.

## 🧪 Training, Validation, and Test Data

To evaluate how a model might perform on new examples, keep some data separate from training.

- **Training set:** Used to fit the model and preprocessing steps.
- **Validation set:** Used to compare models, select features, or tune hyperparameters.
- **Test set:** Held back for a final evaluation after development choices are complete.

A simple workflow is:

```text
Training data → Fit models and preprocessing
Validation data → Compare choices and tune settings
Test data → Final evaluation of the selected approach
```

If you repeatedly change the model based on test results, the test set has influenced development and no longer provides an unbiased final check.

### Choose a split that matches the data

A random split may work for independent examples. Other data structures need different approaches:

- **Time-dependent data:** Train on earlier records and test on later records.
- **Grouped data:** Keep related examples, such as records from the same customer, in the same partition when appropriate.
- **Imbalanced classification:** Use stratification when it matches the task and data structure.

A random split can give misleading results if it mixes future information into training or places near-duplicate records on both sides.

### Data leakage

**Data leakage** happens when training or model selection uses information that would not be available at prediction time. It can make evaluation scores look better than real-world performance.

Examples include:

- Scaling or imputing the full dataset before splitting
- Selecting features using the complete dataset
- Including a field created after the outcome occurred
- Placing near-duplicate or related records in both training and test sets

Scikit-learn recommends pipelines as one way to ensure preprocessing is fit on the appropriate training portion during validation. ([scikit-learn: Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html))

## ⚖️ Overfitting, Underfitting, and Regularization

### Overfitting

A model **overfits** when it learns patterns specific to its training data—including noise—that do not carry over to new examples.

Common signs:

- Very strong training performance
- Much weaker validation or test performance
- Predictions that change sharply with small data changes

Possible responses include simplifying the model, adding regularization, gathering more representative data, or reducing leakage and noisy features.

### Underfitting

A model **underfits** when it is too simple, poorly configured, or insufficiently trained to capture useful patterns.

It tends to perform poorly on both training data and held-out data. A more appropriate model, useful features, or different training settings may help.

### Regularization

**Regularization** discourages models from becoming unnecessarily complex. It can help control overfitting by penalizing large coefficients or otherwise limiting model flexibility.

For linear models, common approaches include:

- **L1 regularization:** Can push some coefficients to zero.
- **L2 regularization:** Shrinks coefficients toward zero.
- **Elastic Net:** Combines L1 and L2 penalties.

The appropriate strength of regularization is a hyperparameter and should be selected using validation or cross-validation.

## 🔁 Cross-Validation and Hyperparameter Tuning

### Cross-validation

In **k-fold cross-validation**, the training data is split into `k` parts, or folds. The model is trained on all but one fold and evaluated on the remaining fold. This process repeats so each fold is used for validation.

```text
Fold 1: Train on folds 2–5, validate on fold 1
Fold 2: Train on folds 1, 3–5, validate on fold 2
...
Combine validation scores to compare approaches
```

Cross-validation can give a more stable estimate than relying on a single training/validation split. It does not fix a poor split strategy: use time-aware or group-aware validation when the data requires it.

### Hyperparameters

**Model parameters** are learned during training. **Hyperparameters** are settings chosen before or around training, such as:

- Tree depth
- Number of neighbors in KNN
- SVM regularization strength
- Number of trees
- Learning rate

### Search methods

- **Grid search:** Tries each specified combination. It is straightforward but can be expensive.
- **Randomized search:** Samples combinations from specified distributions. It can cover a wider search space with a limited budget.
- **Successive halving:** Starts with many candidates and allocates more resources to promising ones.

Scikit-learn provides search tools such as `GridSearchCV` and `RandomizedSearchCV`. The full preprocessing-and-model pipeline can be searched together. ([scikit-learn: Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html), [Pipelines](https://scikit-learn.org/stable/modules/compose.html))

Keep the final test set separate from hyperparameter tuning. Using test scores to choose settings turns the test set into another validation set.

## 🧑‍💻 A Practical Python Example

This example combines missing-value handling, categorical encoding, numerical scaling, a model, and cross-validation. It uses scikit-learn’s built-in Titanic dataset loader where available in the installed version. If the loader is unavailable, replace it with a local dataset containing a binary target, numerical columns, and categorical columns.

Install scikit-learn:

```bash
pip install scikit-learn
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Example data: replace X and y with your own features and target.
# X should be a pandas DataFrame.
# y should contain the target labels.
#
# Example:
# import pandas as pd
# data = pd.read_csv("your_dataset.csv")
# X = data.drop(columns=["target"])
# y = data["target"]

# Declare these based on your dataset.
numeric_features = ["age", "fare"]
categorical_features = ["class", "embarked", "sex"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ]
)

# Split once before cross-validation. Keep the test set untouched
# until model selection is complete.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

parameter_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best cross-validation F1:", search.best_score_)

# Use the test set only for the final evaluation.
test_score = search.score(X_test, y_test)
print("Final test F1:", test_score)
```

The example’s feature names are placeholders; change them to match your data. Use a scoring metric appropriate to your task. For imbalanced classification, accuracy may not be informative enough.

The pipeline keeps imputation, encoding, scaling, and classification together. During cross-validation, each transformation is fit using that fold’s training portion rather than the full dataset. ([scikit-learn: Pipelines](https://scikit-learn.org/stable/modules/compose.html))

## 🧯 Common Pitfalls

- **Preprocessing before splitting:** This can leak information from validation or test data.
- **Tuning against the test set:** Repeated test-set decisions make its final score optimistic.
- **Using a random split for every problem:** Time-ordered and grouped data may need specialized splitting.
- **Treating missingness as meaningless:** Missing values may have a cause or pattern worth investigating.
- **Encoding categories as arbitrary numbers:** This can imply an order that does not exist.
- **Creating features with future information:** A feature must be available at the moment of prediction.
- **Keeping identifiers without justification:** IDs can encourage memorization rather than generalization.
- **Choosing a metric by habit:** Select evaluation metrics based on the costs of mistakes.
- **Assuming cross-validation removes all bias:** The split, data quality, and experiment design still matter.
- **Overinterpreting feature importance:** Predictive usefulness does not prove causation.

## 📝 Field Notes

- Inspect data before deciding how to clean it.
- Learn preprocessing rules from training data only.
- Use features that would genuinely be available at prediction time.
- Separate training, validation, and test roles.
- Choose a split that respects time, groups, and class balance where relevant.
- Cross-validation helps compare approaches; it does not replace a final untouched test set.
- Pipelines help keep transformations and models together and reduce leakage risks.
- A good evaluation process is part of the model—not an afterthought.

## 🔗 Further Reading

- [scikit-learn: Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn: Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)
- [scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)

---

*Reliable machine learning depends on more than a clever algorithm. Good data preparation, careful validation, and honest testing help reveal whether a model has learned a pattern that can actually generalize.*
