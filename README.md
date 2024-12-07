## Info ##

The python repo used for experiments for my masters thesis in ensemble learning. Contains custom made 'ensembles' where ensemble members co-train themselves on a peer-to-peer basis.  </br>
Uses [sacred](https://github.com/IDSIA/sacred) and [river](https://github.com/online-ml/river).</br>
NOTE: This setup is for local exploration purposes and not 'production' ready. 

## Setup ##

Developed with:
 - python 3.12
 - river 0.21
 - sacred 0.85 

Running the experiments needs a running local Mongo DB to store experiment data. Additionally data is stored under local /runs folder.</br> 
Excluding either the MongoDb or the local storage requires removing the observer at the top of each experiment file, ex. removing the Mongo means removing this line: 
```
 ex.observers.append(MongoObserver(url='mongodb://mongo_user:mongo_password@127.0.0.1:27017/sacred?authSource=admin',
                                  db_name='sacred'))
```
For easy docker setup see [sacred's docker setup] (https://sacred.readthedocs.io/en/stable/examples.html#docker-setup). Sacredboard is optional and can be excluded from the docker file. </br>
Recommended visualisation tool: [Omniboard] (https://github.com/vivekratnavel/omniboard) - is part of sacred's docker compose file. 

## Running the experiments ## 

Experiments can be run from the command line, example commands are given in the file: 

```
run-commands.txt
```
Experiments have a lot in common, but are separated into their own files for better naming & dashboard visualisation. </br>
Command parameters are based on sacred configuration - see correspoding function marked with @ex.automain in the experiment file. </br>
Comparison experiments generate a data set as part of the experiment run, to generate the same data set, pass on the ```seed``` parameter: 
example: 

```
python -m src.comparison_experiments.single_vs_elo_ensemble_sea_label_count with nr_runs_per_config=5 nr_samples_train=500 label_count=10 nr_samples_test=50 nr_samples_validation=50 test_step=20 nr_learners=20 pick_train_pairs_strategy='random_subset' pick_play_pairs_strategy='all' nr_pairs=0 nr_repeats=10 seed=750278473
```
## Exploring experiments ##

The folder ```experiment_database_backup``` contains a mongodump database archive export from a local sacred database containing some indicative experiment runs. </br>
To view them run the docker compose file from the docker setup for sacred and import the database shapshot with the [mongodump tool](https://www.mongodb.com/docs/database-tools/mongodump/#mongodump). 
The locally started Omniboard can be accessed on: http://localhost:9000/sacred


