NOTE: All Data has been removed for confidentiality reasons as it is the property of Belvedere Trading LLC.

Report can be found HERE and poster presentation HERE.

This is a private repository for the Spring semester DSCI 435/535 Belvedere project. This code builds on prior work done by the Fall semester's team, and may include some of their code.
<br /><br />
To run the code, you will need the packages found in requirements.txt.
<br /><br />

Python: Version 3.7

Libraries used:
hmmlearn==0.2.3
pandas==0.25.1
numpy==1.17.2
plotly==4.5.4
matplotlib==3.1.1
scikit_learn==0.21.3
talib==0.4.17
seaborn==0.9.0
isoweek==1.3.3

# Description of the Repo
As of 5/1/2020, there are six files and six directories and found in the root directory. There is also a Symbol_Definitions.txt file found in the root directory, which is a quick reference between future contracts symbols and their common name.
<br /><br />

# Files
## main_test.py
This file is meant to be run in terminal/command line from the main directory, and runs tests for the relevant and important files/functions in the project. 
It is meant verify the functionality of the files. Within the Alt_Test_Files directory there are additional test files to serve as references for functionality.

Note: This file may have a long run time (~1+ hour) depending on your machine. 

## main_driver.py
This file is also meant to be run in terminal/command line from the main directory, and shows three standard workflows for usage of the code
in this repo. The first runs the regime classification models, the second runs the integrated classification and lasso model,
and the third runs the integrated hmm and lasso model. This is not at all exhaustive of the work that was done, but should
serve as a helpful reference for how to use the some of the primary functionality of this repo. 

## test_pre_selected_features.py
Shows the workflow and produces the pre modeling selected feature results.

## test_classification_models.py
Produces the classification model results including pair-only vs. pair-plus tests, lagged features tests and ensemble feature selection.
Note: This takes a VERY long time to run - up to a day or more.

## test_hmm.py
Produces results for the hmm model.

## test_lasso_models.py
Produces test results for Lasso and Lasso combination models (hmm and classification models).

# Directories
## Data_Master 
This directory contains all of the data as well as the functions required to clean, wrangle, and export it. 
The main function to use here is read_data.py - this is the master function to read original data supplied by Belvedere
Trading into a pandas dataframe and correlation data from the fall semester's team. Additionally, the feature_engineering.py script is contained within this directory. It 
is typically called by functions within read_data.py. It will be helpful to reference the test files as examples of how to use it.

The R_code directory contains the R code that was used to obtain correlation data that was produced by the fall semester's 
team using graphical models. 

Note: Back-filling of missing correlation data is done in several places in the read_data.py functions. It was not determined if 
this was the best method for handling missing data, but special attention should be given to this before using results in any live/production system
and for additional work done on this project. 

## Data_Exploration
This directory contains a single script - polar_correlation_generator.py - which 
generates polar visualizations of correlations and correlation differences for a given time interval. It can also export the visualizations to the Visualizations directory.

## MarketFeatures
Contains multiple file to experiment with and plot certain market features. Meant for pre-modeling feature selection. 

## Modeling
Contains many files which encompass all attempts at modeling the data. By the end of the project, only a small subset of 
these files were used to generate results. These include Lasso_classification.py, lasso_hmm.py, HMM_all_futures_features.py, and AltModels.py. 
Other files may be used in parts of the project, but these are the main ones which should likely receive attention moving forward. 

## Visualizations
Contains the polar visualizations of correlations and correlation differences over four time intervals (weekly, biweekly, monthly, and seasonal) for five categories of futures - Agriculture, Energy, Livestock, Metals, and Soybeans. 
Also contains a copy of the visualizations produced by functions in MarketFeatures. 

## Alt_Test_Files
Contains five test files from which all other test files were derived. 
These files will need to be moved to the main directory in order to run them. 
There may be some additional tests, methods, and context found in these files which will be useful to those using this repo. 
