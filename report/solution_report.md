Mateusz Jukiewicz

January 7, 2019

# Kaggle's Predict Future Sales solution report

## Introduction
This report summarizes my attempt to solve the "Predict Future Sales" competition hosted by Kaggle.
 
The report includes the following sections:

* [Competition overview](#competition-overview)
* [Solution description](#solution-description)
    * [Input data](#input-data)
    * [Initial ideas & challenges](#initial-ideas--challenges)
    * [Exploratory data analysis](#exploratory-data-analysis)
    * [Evaluation / Validation](#evaluation--validation)
    * [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
        * [Preprocessing](#preprocessing)
        * [Feature Engineering](#feature-engineering)
            * [Finding out new features](#finding-out-new-features)
    * [Modeling](#modeling)
        * [Initial model](#initial-model)
        * [Model with feature engineering](#model-with-feature-engineering)
        * [Dense embedding of items/shops/categories](#dense-embedding-of-itemsshopscategories)
        * [PyTorch embedding neural net](#pytorch-embedding-neural-net)
        * [PyTorch LSTM net](#pytorch-lstm-net)
* [Results](#results)
* [Performance improvements](#performance-improvements)
* [Summary](#summary)
 
## Competition overview
The task in the competition is to predict monthly sales for a combination of items and shops.
Competitors are provided with daily sales data for certain shops and items. The evaluation metric
for the competition is Root Mean Squared Error (RMSE).

For more information, see <https://www.kaggle.com/c/competitive-data-science-predict-future-sales>.

## Solution description

#### Solution note
The solution described in this report is rather a fast-paced one. Therefore, it is not focused on stuff that is usually heavily explored in
competitions, namely:

* Extensive feature selection & engineering
* Hyperparameter tuning
* Complex ensemble modeling

Instead, my main goal was to firstly obtain a reasonably good result, and then to explore some approaches that
I thought are novel and not explored by other competitors (at least according to competition kernels/discussions).

### Input data
There are 6 files provided by the competition organizer:

* sales_train.csv - daily historical sales data from January 2013 to October 2015.
* test.csv - the test set with item/shop combinations from November 2015, for which sales have to be predicted
* sample_submission.csv - a sample submission file in the correct format
* items.csv - supplementary information about the items
* item_categories.csv - supplementary information about the items categories
* shops.csv- supplementary information about the shops

### Initial ideas & challenges
The first thing to consider is the best way to process the datasets in order to make successful predictions. Since the
training data is a daily based time series, and the test data is monthly-based, several options came to my mind:

1. Data granularity
    1. Daily based training / daily based prediction / aggregation of predictions to monthly format
    1. Monthly based training by training data aggregation / monthly predictions
2. Observations ordering
    1. Classic approach - neglect the order and train as independent observations
    1. Time series approach - sequence based algorithms (ARIMA, RNNs)
    
Every option has some benefits and drawbacks. I decided not to pursue with daily data approach,
because that would require artificially creating days for the test set and then aggregating them back to monthly sales. 
Doesn't seem like a perfectly reasonable approach. Additionally, creating a proper validation setup would also be challenging.

I decided to go with monthly-aggregated non-sequenced approach, so basically the simplest one.

I then experimented more with time-series based approach which yielded pretty good results.

### Exploratory data analysis
I performed some basic EDA to check if the data looks reasonable and there are no obvious errors.

The most important insight I got from EDA was that test set has combinations of all shop/items for a month, 
so the train data has to be processed accordingly in order to obtain proper validation setup.

Some additional interesting facts are the following:

* The sales trend is getting lower every year
* The most expensive item sold was a software licence pack with over 500 licences
* The most frequently sold item is a plastic bag
* The item with most varying price is a credit card payment (or a gift card, I'm not 100% sure)

##### Sales for each year
![Sales for each year](figures/sales-through-year.png)

#### Total sales by shop by year
![Sales by shop by year](figures/sales_by_shop_by_year.png)

The full EDA report is available as a [Jupyter Notebook](../eda/basic_datasets_overview.ipynb)
or [HTML file](../eda/basic_datasets_overview.html) (there are some interactive plots not 
shown here).

### Evaluation / Validation
Since the competition data is a time series, the observations are not independent, as is usually assumed in classic
problems. Therefore, a standard random hold-out/cross-validation setup would not be proper here, as it would yield biased results.
What is needed is a time aware splitting procedure.

For validating the models built, I implemented two time aware validation procedures:

* Simple holdout - the faster method for quick validation. I performed training on `X` months, then I performed
validation on the consecutive month. E.g. training: Jan 2015 - Sep 2015, validation: Oct 2015.
* Time aware cross validation - more extensive validation, where train set width and test set width (in months)  has to
be specified, as well as the number of iterations. The training/validation is then performed similarly to the method above,
but repeated many times for different date ranges. This one reports more detailed information about estimated evaluation metric.

The implementation of the validation procedures is available at [../modeling/model_validation.py](../modeling/model_validation.py).

### Preprocessing & Feature Engineering
I implemented the preprocessing/feature engineering procedure in Spark Scala. The code is available at
[../scala/preprocess_data.scala](../scala/preprocess_data.scala).

#### Preprocessing
The most important preprocessing steps were:

* aggregating the data from the daily basis to monthly basis
* extending the training set to contain all item/shop combinations per month
* filling test data and combining it with training data for feature engineering process

I also performed some standard stuff like year/month extraction, etc. The rest of the processing is contained in Feature
Engineering section.

#### Feature Engineering
When it comes to feature engineering, one must be very careful when working on time series data and watch out for proper
ordering. It is very easy to 'leak' some information from future observations and get biased estimates because of that.

Thankfully, spark has nicely working and pretty intuitive window functions, which made this process pretty much seamless.

I created a number of features, amongst them:

* previous item shop sales (1-2-3 periods back)
* number of times an item was sold previously
* previous mean price for item/shop combination
* previous mean price for item (in general, without relating it to any particular shop)
* previous item's category price
* and more

Please check the script file to see all features implemented.

##### Finding out new features

##### Manually 
One of the interesting approaches I used to find out new features was evaluating the model manually. This is a very
useful technique that I believe is undervalued. It is especially useful when features can be interpreted by a human being. 

The procedure looked as follows:

* Train the model on current dataset
* Perform predictions on validation set
* Take few observations with biggest error and try to find out why the model make a mistake
* In case of any ideas, design and implement new features
* Repeat

During this procedure I found several really nicely working features. I later found many similar features in other
competitors' kernels!

##### Using eli5
Later on, I used eli5 to check the permutation importance of created features. It gave me additional insights regarding
useful features. I even found some design errors in few features so it came out to be really useful.
The code is available at [../modeling/feature_importance.py](../modeling/feature_importance.py).

### Modeling
#### Initial model
Initially, I ran training on very basic features (`date_block_num, month, year, item_id, shop_id`)
without any feature engineering using `xgboost` without any hyperparameter tuning.

This gave me the result of `1.10619` on the public leaderboard and around `1.08` on my quick validation procedure
(which I believe is pretty acceptable difference).
#### Model with feature engineering
After adding the features crafted in Spark/Scala script and using the same model, my public leaderboard score
jumped to `0.94416` on the public leaderboard (again around `0.02` worse than my internal validation).

#### Dense embedding of items/shops/categories
One of the competition challenges was that there are lots of different items
(around 22k). Obviously, adding item information to the dataset is crucial. I was not satisfied in label encoding since
this was a categorical and not an ordinal feature. One hot encoding was also a pretty bad option since it would create very wide
and very sparse dataset, creating both computational and predictive performance problems.

Therefore, I decided that I will try to create dense embeddings of items/shops/item_categories from their descriptions
available in supplementary datasets. This has several benefits:

* dense vectors of much lower dimensionality
* since the features were text based, there was a possibility that the model would gain predictive power to never seen items
utilizing the fact that the description of item/shop was similar to previously seen ones

To implement this idea, I used `Word2Vec` algorithm using `gensim` library, as well as already pre-trained Russian
Wikipedia embeddings.
The implementation of the idea is available at [../categorical_variables_embeddings/generate_dense_datasets.py](../categorical_variables_embeddings/generate_dense_datasets.py).

After adding the dense representations and training the xgboost model only on data from 2015, I managed to get a score of 
`0.92882` on the public leaderboard.

*Note: To be honest, I used only 2015 data due to speed reasons, but it turned out to work better than using the whole dataset.
I still believe that full data could be utilized with more careful hyperparameter tuning.*

#### PyTorch embedding neural net
Another idea of generating dense representations for shops/items/categories was to use embedding layers in neural network.
I though that it might be a good idea to let the model learn the embeddings itself from the existing data. The network
seem to work pretty good, achieving slightly better performance than `xgboost` in my internal validation.
Later on, I decided to combine both dense vector approaches and used the following 'feature set':

* All features crafted in scala script
* Dense features learned by gensim
* Dense features learned by internal embedding layers

I implemented this network in `PyTorch`, which is powerful and very intuitive library for building neural networks.

The full implementation is available at [../modeling/models/torch_embedding_net.py](../modeling/models/torch_embedding_net.py).

The code to implement the whole network architecture was as simple as that:

```pythonstub
class EmbeddingNet(torch.nn.Module):
    def __init__(self, items_path, items_categories_path, shops_path):
        super(EmbeddingNet, self).__init__()

        self.device = torch.device('cuda')
        self.cols_in_order = ['item_id', 'item_category_id', 'shop_id']
        self.cols_rest = None
        self.standardizer = StandardScaler()

        items = pd.read_csv(items_path)
        items_categories = pd.read_csv(items_categories_path)
        shops = pd.read_csv(shops_path)

        embeds_size = 20
        other_features_num = 98
        other_features_out = 196

        items_size = items.shape[0]
        items_categories_size = items_categories.shape[0]
        shops_size = shops.shape[0]

        self.embedding_items = torch.nn.Embedding(items_size, embeds_size)
        self.embedding_categories = torch.nn.Embedding(items_categories_size, embeds_size)
        self.embedding_shops = torch.nn.Embedding(shops_size, embeds_size)
        self.input_rest = torch.nn.Linear(other_features_num, other_features_out)
        self.hidden_concat = torch.nn.Linear(embeds_size * 3 + other_features_out, 50)
        self.final = torch.nn.Linear(50, 1)
```  

After a bit of work with the optimizer (setting proper learning rate and number of epochs),
I managed to get a leaderboard score of `0.90933`, while my internal validation score was around `0.89`.

#### PyTorch RNN net
Finally, I decided to utilize the ordered nature of the data and built a RNN-based neural network.
I treated it more like a challenge since I suspected that the dataset might be 'not sequential enough' to enable RNN to
extract some additional insights.

Nevertheless, it has actually turned out to be the best model. Being fair, though, this model had the biggest number
of parameters and I have put the most effort into it, so I'm still not fully convinced that it's the sequential factor
that made the biggest difference.  

After all, using some simple prediction embeddings I managed to get a leaderboard score of `0.86097` which gave me 6th
place at the moment of submission (now 17th)

The code for RNN network is available at [../modeling/models/rnn](../modeling/models/rnn).

## Results
The summary of the results I obtained look as follows:

| Method        | Public leaderboard score           |
| ------------- |-------------:|
| Initial xgboost      | 1.10619 |
| xgboost + engineered features | 0.94416 |
| xgboost + eng feats + dense feats | 0.92882 |
| PyTorch embedding net | 0.90933 |
| Best xgb and Best PyTorch Mean* | 0.90326 |
| RNN simple embedding | 0.86097 |

## Performance improvements
Although the initial dataset was pretty small, after extending it with all items/shops combinations and new features I
started running into memory/speed problems. I used the following technologies to work around this:

* Spark Scala (also useful for standalone applications!)
* Using parquet file format to move data around
* xgboost trained on GPU
* running neural nets training on GPU (kind of obvious improvement)

## Summary
Working on this competition was pretty fun, I reminded myself/learned some stuff about working with time series datasets.

Finally, my current position in the competition after implementing the RNN is 17th. I'm aware I made lots of
submissions so there's a possibility of leaderboard overfit.

I believe I could improve even more with some competition-typical stuff (hyperparameter tuning, ensembles, etc.)

![Kaggle Place](figures/kaggle.png)