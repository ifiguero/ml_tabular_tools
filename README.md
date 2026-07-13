
# MODY Code and Dataset

This repository contains the Python code used to train models and the pre-processed dataset ready to be loaded into the pipeline.

## Prerequisites Installation

You’ll need [Miniconda / `conda`](https://docs.anaconda.com/miniconda/install/) to create the environment with the required dependencies:

```bash
conda create -n mltab_tools -c conda-forge python=3.10 numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn tensorflow keras keras-tuner tensorboard xgboost shap ray-tune enlighten joblib openpyxl xlsxwriter
````

After installing the environment, activate it with:

```bash
conda activate mltab_tools
```

## Running the Pipeline

The file `run.py` contains an example of the pipeline. To run it, simply execute:

```bash
python run.py
```

## Pipeline Structure

```python
study = BasicLoader('Tabular_data.xlsx', target=['outcome'], continuous=['heigth','age'], discrete=['sex','treatment'])
```

The `BasicLoader` reads the `Tabular_data.xlsx` file, which contains raw dataset. Required parameters include `target`, `continuous`, `discrete` that refer to column names on the raw dataset. Target can be a list so the loader keeps separate dataset for each target, all target MUST be binary.


```python
study.save_datasets('datasets_clean.xlsx')
study.save_univariate_analysis('univariate_analysis.xlsx')
study.plot_all_boxplots()
study.plot_all_distributions()
study.plot_all_heatmaps()
study.plot_all_pca()
```

Export the data and figures as specified on each method, providing sane defaults but all are customizable.

```python
df_outcome = study.get_dataset('outcome', remap_smaller_is_zero=True)
```

You can access the dataframe for a target directly by referencing it on the study. This return a specific DataFrame` containing the target column. `remap_smaller_is_zero` maps the binary to [0,1] for downstream analysis, mapping the smaller value to 0 and 1 otherwise, by default is disabled.

Next, we create a `BinaryTuner` object, which provides the abstractions to generate 10 datasets and test 10 different machine learning strategies.
It takes multiple parameters — the first being the target column name, followed by either the number of seeds (`n_seeds`) or a vector of specific seeds (`seeds`), and the test set proportion.

```python
trial = BinaryTuner(df_outcome, 'outcome', drop_ratio=0.2, n_seeds=5)
```

This creates a directory named after the target column, which stores logs, models, and generated images.
**Note:** This introduces a restriction — if you want to run experiments on other datasets with the same target column name, ensure that the directory does **not** already exist before starting a new training run.

```python
trial.fit()
```

Trains all models for all test sets.

```python
trial.explain_model('GaussianNB', 'fulldataset-oversampled-mice', 231964)
```

Generates SHAP plots for the specified model/dataset/seed combination.
A complete list of model names, datasets, and seeds can be found in the files within the directory.

```python
trial.wrap_and_save()
```

This is an auxiliary method that creates a compressed `.zip` file containing the directory’s contents.
The file includes a timestamp to prevent name collisions in subsequent runs.
