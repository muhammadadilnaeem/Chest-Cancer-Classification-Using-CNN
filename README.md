
------

# **Chest-Cancer-Classification-Using-MLflow-and-DVC**
This repository will contain source code of Chest Cancer Classification Using MLflow and DVC Project.


### **`Workflows`**

1. **Update config.yaml**
2. **Update secrets.yaml [Optional]**
3. **Update params.yaml**
4. **Update the entity**
5. **Update the configuration manager in src config**
6. **Update the components**
7. **Update the pipeline**
8. **Update the main.py**
9. **Update the dvc.yaml**

# **`Dataset`**

- Dataset link : https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images 

- In this project we used mlflow for tracking deep learning experriments and model.

- Also we used DVC( Data Version Control) for project pipeline tracking.

First we use command to initiate project

```python
dvc init
```

Then for pipeline tracking we will use this command

```python
dvc repro
```


-------