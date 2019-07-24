## BAWML

Behavioral Authentication with Machine Learning

#### Requirements
1. Need to install [conda](https://www.anaconda.com/distribution/)
2. Recommended IDE is PyCharm

#### Install Package

Run the following commands to install all packages:

`conda env create -f environment.yml`

Run the following commands to update package:

`conda env update -f environment.yml`

#### Run Configurations

In order to run the project we need to go through the following steps:
1. Firstly, we need to set some value for `data_path.yaml`
      - `raw_data:` : This is the raw path folder where all persons raw data 
      (input data) should exists.
      - `output_path` : Specify a directory where all data will be saved. e.g 
      series,spectrogram's,training
      - `test_name` : Any arbitrary name which will be used to create sub directory.
      - `train_test_split` : It's a boolean value If we set it true then TFRecord 
      will generate based on 70/30 split otherwise not.
      
2. Then we need to `/prep/data_prep/run raw_data_processing_main.py` file to 
generate all necessary pre-processing data.

3. Then we need to configure our `experiment.yml` file under experiments 
directory. Here we need to specify some path:
    - `NAME` : Give any arbitrary name where logs will be saved
    - `CHECKPOINT_DIR` : Give a path where it can save checkpoint
    - `DATA_DIR` : Give the full path of the three channel spectrogram 
    data directory.

    There are some default values for some other parameters like Image width,height,
    Normalization,optimizer etc. if we want we can modify those variable too. 
    
4. Run the `training.py`. Currently, this file reads from `experiment_3.yml` 
file configurations. But if you want to use some other experiments then
need to change the `main` function of `training.py` accordingly.  

#### Code Committing

Please try to use commit message according to this [convention](https://gist.github.com/brianclements/841ea7bffdb01346392c).