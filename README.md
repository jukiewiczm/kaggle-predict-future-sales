# Predict Future Sales

My attempt to solve Kaggle's Predict Future Sales competition.

Contains:

* RNN approach (pytorch)
* Data processing script in Spark (scala)
* Creation of embeddings for item/item category/shop descriptions
* Feature importance check using eli5
* And more!

# Report

For the solution report, check [here](report/solution_report.md).

# Automatic hyperparameter tuning
For RNN-based model, automatic hyperparemeter tuning is available with [Guild AI](https://github.com/guildai/guildai) AutoML tool.

Check [this document](modeling/models/rnn/README.md) for details.

# External resources

* Datasets from https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data
* If you would like to use russian embeddings pretrained from wikipedia, download them from 
[here](https://fasttext.cc/docs/en/pretrained-vectors.html)

The default location for these is the [data](data) folder.

# Dependencies
All you need is included in [environment.yml](environment.yml). Conda is the package manager for this project.