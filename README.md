# Mlflow-Codebase
Mlflow-Codebase

Create Environment

```
command: conda create --prefix ./env python=3.6 -y && conda activate ./env
```

Test Runs Available in mlruns folder

For Mlflow UI 
```
command: mlflow ui
```
MLflow Logging Command
```
mlflow.log_param("alpha",alpha)
mlflow.log_param("l1_ratio",l1_ratio)

mlflow.log_metric("rmse",rmse)
mlflow.log_metric("MAE",mae)
mlflow.log_metric("R2_score",r2)
```

MLflow Logging Model Command
```
mlflow.sklearn.log_model(lr, "model")
-> Logging model available for other envs as well such as tensorflow,pytorch etc.
```
