[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MqChnODK)

## Run grading tests

To run grading test, https link doesn't work so you have to use this command with SSH link .

```
pytest --github_link git@github.com:CS-433/ml-project-1-overfrites.git .
```

## How to Run

1. Download the dataset and place it in a folder named `data`. (dataset here : https://www.cdc.gov/brfss/annual_data/annual_2015.html)
2. Execute the following notebook `run.ipynb`

## Code Structure

Our codebase consists of the following main files:

- `run.ipynb`  
  This notebook handles the complete workflow, including data loading, training, and prediction steps.

- `implementations.py`  
  Contains all ML methods required in Step 2.

- `preprocessing.py`  
  Handles all preprocessing steps, including:

  - Removing rows and columns with missing values
  - Replacing missing values with mean values
  - Balancing the dataset by creating multiple balanced subsets
  - Adding polynomial features
  - Standardize

- `preprocessing_with_encoding.py`  
  (Optionnal) handle categoricals variables.

- `train.py`  
  Contains functions for making predictions.
- `evaluation.py`  
  Includes functions for calculating F1 score & Accuracy.
- `selection.py`  
  Contains functions which allow us to choose correct parameters.
