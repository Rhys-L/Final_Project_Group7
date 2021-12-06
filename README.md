# Overview of Final project for dats-6103

Our goal for this project is to utilize data mining techniques and algorithms to analyze the physical properties of the Great Lakes. All data used in this project was collected and published for public use by [NOAA and the Great Lakes Environmental Research Laboratory (GLERL)](https://coastwatch.glerl.noaa.gov/statistic/statistic.html). In this project, we aimed to answer the following questions: 

## Key Questions

1. Based on surface temperature, can we build a model to predict whether each of the Great Lakes reach maximum ice coverage with respect to maximum ice cover average threshold?
    - We achieved this through a logistic regression model. 

2. Based on surface temperature, ice concentration, and physical properties, can we build a model to predict which lake a set of characteristics most likely belongs to?   
    - We achieved this through a KNN model that clusters characterists by lake.  


# Repo Structure

The following repositories contain reports, documents, and presentations that outline our findings and modeling approach.


### Code
    
The code subdirectory contains several .py files additional details can be found in the [code directory's readme](code/README.md). 
- The file gather_data.py wrangles data from multiple NOAA and GLERL files on the Great Lakes. 
- KNN_Modeling.py conducts intial EDA on the full dataset, builds a clustering model, and produces a tkinter based GUI to explore how the model performs with different k-values.  
- Project-6103_lm.py conducts initial EDA on the dataset and establishes a threshold of longterm average ice concentration.
- Project-6103-logit.py predicts whether ice concentration exceeds the longterm average based on surface temperature and displays errors in a tkinter GUI.

### Group Reports & Presentations
- Final-Group-Project-Report:
- Group-Proposal:

### Individual Reports 
- carter-rogers-individual-project: report and code written by Carter.
- rhys-leahy-individual-project: report and code written by Rhys.
- nongjie-individual-project: report and code written by Nongjie.

