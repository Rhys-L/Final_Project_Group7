# Overview of Code

Run these files in the following order:

1. gather_data.py
2. KNN_Modeling.py
3. Project-6103_lm.py
4. Project-6103-logit.py

# Code Architecture 

## Data Wrangling and Pre-Processing

- gather_data.py downloads and integrates 26 .dat files hosted on a [NOAA and GLERL site](https://coastwatch.glerl.noaa.gov/statistic/statistic.html). This script is prone to connection and timeout errors due to restrictions on their server. To work around these constraints, we also hosted a copy of the full dataset on a [GCP Bucket](https://storage.googleapis.com/great_lakes/LakeIce_PhysicalProps.csv). If gather_data.py causes problems, the following scripts directly download the clean dataset from a cloud storage bucket.  

## Modeling and Evaluation

- KNN_Modeling.py predicts which lake a set of characteristics belongs to. This model's output is more than twice as accurate as a naive baseline.
- Project-6103_lm.py predicts ice concentration based on surface temperature.
- Project-6103-logit.py predicts whether ice concentration will exceed a given threshold based on surface temperature and physical characteristics. 

## GUI

- KNN_Modeling.py also displays a GUI that allows users to input a k-value and displays the error at that value.
- Project-6103-logit.py also displays a GUI that allows users to input a k-value and displays the model accuracy at that value.
