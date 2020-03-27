# Guild AI hyperparameter tuning

Using resources in this directory, you can perform automatic hyperparameter optimization of the RNN-based model
using **Guild AI** - a very nice, intuitive and relatively easy to use AutoML tool.

## Running the experiment

You can perform hyperparameter tuning by running the follwing command:
```shell script
PYTHONHASHSEED=1234 guild run train_rnn \
    embedding_type=["gensim","wiki"] \
    embedding_size=[1:100] \
    embedding_epochs=[5:50] \
    embedding_lr=loguniform[1e-3:1.0] \
    average_dense_sets=[0,1] \
    num_epochs=10 \
    batch_size=512 \
    learning_rate=loguniform[1e-6:1e-3] \
    pre_rnn_layers_num=[1:5] \
    pre_rnn_dim=[20:500] \
    rnn_module=["gru","lstm","rnn"] \
    rnn_layers_num=[1:5] \
    rnn_input_dim=[20:500] \
    rnn_hidden_output_dim=[20:500] \
    rnn_initialize_memory_gate_bias=[0,1] \
    post_rnn_layers_num=[1:5] \
    post_rnn_dim=[20:500] \
    pre_output_dim=[20:500] \
    run_type="evaluate" \
    --max-trials 150 \
    --optimizer gp
```
This command runs 150 iterations of bayesian hyperparameter optimization with the range of paremeters specified above.

The `PYTHONHASHSEED=1234` here is needed to ensure reproducibility of the experiment, which you probably want.

Note: Due to a [bug in current version of Guild AI](https://github.com/guildai/guildai/issues/126), you can't run the
tuning directly from a [guild.yml](guild.yml) file. Once that's fixed, it should be possible to run a tuning in a more 
concise way:
````
guild run tune_rnn_iter_1
````

I defined three setups of hyperparameter tuning in [guild.yml](guild.yml) file as an iterative process, but feel free
to test your own parameter set!

## Checking the results

One few iterations of tuning is done, there are few ways to monitor the results:

* `guild compare` - good-looking text-based presentation
* `guild view` - web version of the above
* `guild tensorboard` - Tensorboard-based presentation of the results for detailed investigation

Check more details at https://guild.ai/docs/.

## Preparing submission with tuned parameters
Once you're happy with the tuning results, it's time to prepare a submission using the best parameters.

Choose the run id you like the most (e.g. by looking at `guild compare`). You have the following possibilities:
* `guild run --restart [run_id] run_type="load_predict"` - load the model trained during the validation and predict on 
the test data using that model
* `guild run --restart [run_id] run_type"train_predict"` - retrain the model on full training data, then predict on the
test set using that model

Check [guild.yml](guild.yml) for all available options.

Note: Normally, you would do `guild --rerun [run_id] [new_parameters]` to train & predict with the tuned 
hyperparameters. For some reason, this doesn't work in the current version of Guild AI (seems it's not yet implemented). 
Using `guild --restart` will overwrite the run instead of creating a new one, so make sure to get a backup of the run
of choice (`guild export`) to avoid loosing hyperparameter tuning history.  