# MLE Challenge

## Overview

This repository contains a challenge solution for a machine learning engineer role.
The challenge consists on four main steps:
1. *Part I:* Implement `model.py` by selecting a model using Jupyter notebook code.
2. *Part II:* Implement `api.py` for prediction using trained model.
3. *Part III:* Deploy the API in favorite cloud provider (GCP).
4. *Part IV:* Implement CI/CD process in GitHub workflows.

In this solution I chose the Logistic Regression model developed for a flight delay classification task.
The model was selected based on the evaluation of six different configurations,
considering feature selection and class imbalance handling.

### Selected Model: Logistic Regression with Top 10 Features and Class Balancing

The final model was trained using the top 10 most important features obtained by an Exploratory Data Analysis (EDA)
done in Jupyter notebook:

1. OPERA_Latin American Wings
2. MES_7
3. MES_10
4. OPERA_Grupo LATAM
5. MES_12
6. TIPOVUELO_I
7. MES_4
8. MES_11
9. OPERA_Sky Airline
10. OPERA_Copa Air

And applied weight balancing to handle class imbalance.

```python
n_y0 = (y_train == 0).sum()
n_y1 = (y_train == 1).sum()
n_total = len(y_train)

model = LogisticRegression(class_weight={1: n_y0 / n_total, 0: n_y1 / n_total})
```

The model is versioned and stored in the repository under `models/{version}/`.

## Model Selection Process

I experimented with Logistic Regression (LR) and XGBoost (XGB) under different scenarios:

| Model | Top 10 Features | Class Balancing |
|--------|--------------|---------------|
| Logistic Regression | No | No |
| Logistic Regression | Yes | Yes (Selected) |
| Logistic Regression | Yes | No |
| XGBoost | No | No |
| XGBoost | Yes | Yes |
| XGBoost | Yes | No |

I selected the logistic regression model trained with top 10 features and class balancing.
Reviewing And model classification reports, which this one is the chosen one:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.88      | 0.52   | 0.65     | 18,294  |
| **1** | 0.25      | 0.69   | 0.36     | 4,214   |

| Metric        | Value  | Support |
|--------------|--------|---------|
| **Accuracy**  | 0.55   | 22,508  |

And EDA conclusions:

* There is no noticeable difference in results between XGBoost and LogisticRegression.
* Does not decrease the performance of the model by reducing the features to the 10 most important.
* Improves the model’s performance when balancing classes, since it increases the recall of class “1”.

Given the requirement for the model to be **highly available, scalable, and capable of handling a substantial number of requests**, the following considerations were made:

- **Logistic Regression** was chosen over XGBoost because it is computationally lighter, ensuring faster inference times.
- **Feature selection** using only the top 10 most important features reduces the input data size, optimizing processing efficiency.
- **Class balancing with weight adjustment** significantly lowers the overall accuracy but improves the model’s ability to predict the minority class (i.e., identifying flights that **will** experience a delay).

Based on these factors, **Logistic Regression with top 10 features and class balancing** was selected as the final model, prioritizing both efficiency and fairness in classification.

## Model Artifacts and Structure

The trained model is stored in pickle format (`.pkl`) and versioned under:

```
models/
└── v1.0/
    ├── lr_model.pkl        # Trained model
    ├── lr_metadata.json    # Model metadata (e.g., train date, version, feature, metrics)
    ├── all_columns.pkl     # Total columns used for validation
```

- `lr_model.pkl`: The serialized model ready for inference.
- `lr_metadata.json`: Contains versioning, hyperparameters, and training details.
- `all_columns.pkl`: Stores the original feature set used during training to ensure valid input data.

## Model Deployment & API Usage

The model is deployed as an API using FastAPI and containerized with Docker.
The deployment pipeline is managed via Google Cloud Platform (GCP), specifically using Cloud Run for serverless execution.

### API Endpoint

The model is available for prediction via a REST API:

```
POST /predict
```

### Request Format (JSON)

```json
{       
    "flights": [
        {
            "OPERA": "Aerolineas Argentinas", 
            "TIPOVUELO": "N",
            "MES": 13
        }
    ]
}
```

### Response Format

```json
{
  "prediction": [0],
}
```

- `prediction`: List of predicted class label.

## CI/CD Pipeline

My CI/CD pipeline automates testing, model validation, API functionality checks, and seamless cloud deployment.

- **Continuous Integration (CI):**  
  - Runs unit tests on the machine learning model to ensure performance and stability.  
  - Executes API tests to verify correct functionality and responses.  

- **Continuous Deployment (CD):**  
  - Builds and packages the application using Docker.  
  - Pushes the container image to **Google Artifact Registry**.  
  - Deploys the application on **Google Cloud Run**, ensuring scalability and availability.  

This automated workflow ensures that every code update is tested and deployed efficiently to maintain a reliable and production-ready system.


## Future Improvements

Several enhancements can be considered for future iterations:

1. **Model Storage Optimization**  
   Instead of storing models in `.pkl` files within the repository, we could use **Google Cloud Storage (GCS) buckets** or **Artifact Registry** for version control and easy retrieval.

2. **Pipeline-based Training and Inference**  
   The current approach keeps preprocessing as a separate process. Instead, we could integrate **sklearn pipelines** to streamline feature processing, making inference more seamless.  
   This would allow us to **load the model as a single object** and handle both preprocessing and predictions in one step.  

3. **Improving CI with Best Practices**  
   The CI pipeline can be enhanced by integrating **code quality and style enforcement tools**, such as:  
   - **Linting** to ensure consistency and avoid common coding errors.  
   - **Black** for automatic code formatting.  
   - **SonarQube** for static code analysis and detecting vulnerabilities.  

4. **Optimizing CD with Artifact Management**  
   In the CD process, we could **store and manage the resulting sklearn pipeline artifact** instead of uploading all model-related files.  
   - This approach would allow us to **deploy only the API**, keeping the pipeline in **Google Artifact Registry** or **GCS**.  
   - The API would then retrieve and load the artifact dynamically, reducing complexity and ensuring version control.  

