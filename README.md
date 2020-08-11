
# Tanzania Well Classification

#### A classification project using the [Tanzania Water Well Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) from Driven Data.
![header](references/pexels-kelly-lacy-3030281.jpg)
     
# Our Goal
Our goal is to discover if data aggregated by [Taarifa](http://taarifa.org/) and the [Tanzanian Ministry of Water](https://www.maji.go.tz/) can be used to predict well functionality. Our ultimate goal is to provide a dashboard capable of mapping well functionality as well as predict failure dates for functioning wells. This project is important because [Tanzania has struggled](https://water.org/our-impact/where-we-work/tanzania/) to provide water to its growing population and if we can predict well functionality, we can provide insight that may be crucial for fixing their infrastructure.

## Objectives
1. Investigate the relationship between water point functionality and these factors:
     * Users, installers, and managers
     * Geographic location
     * Year built
     * Technology used
2. Build a predictive model
3. Report findings for future work

## Table of Contents
1. [Context](https://github.com/sydroth/tanzaniawellclassification#context)
2. [Data Analysis](https://github.com/sydroth/tanzaniawellclassification#data-analysis)
3. [Data Preperation](https://github.com/sydroth/tanzaniawellclassification#data-cleaning)
4. [Modelling](https://github.com/sydroth/tanzaniawellclassification#modelling)
5. [Results](https://github.com/sydroth/tanzaniawellclassification#final-model-results)
6. [Methods](https://github.com/sydroth/tanzaniawellclassification#methods)
7. [How to Contribute](https://github.com/sydroth/tanzaniawellclassification#want-to-contribute)

### Context
According to [WaterAid](https://www.wateraid.org/tz/), only 60% of Tanzanians have access to safe drinking water and only half of all rural residents have access to proper handwashing facilities. This problem is exacerbated by their growing population, which is expected to increase in size by about 30% by 2030. The government is trying to address this with plans to provide universal access to water by 2025, but the infrastructure already in place is failing, making it an uphill battle. If the government has a way to predict when and where waterpoints might fail, they will be better able to respond to failures efficiently, saving time and money.

### Data Analysis
The data provided by these resources was collected by Driven Data for a competition which can be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). The dataset provided has 59400 observations, with each row representing a single waterpoint. The target label in this dataset is 'Status_group' which contains the values 'functional', 'functional needs repair', or 'non-functional'. There are 39 features ranging from technologies used to the group that manages the well. A simple data dictionary can be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/#features_list)

#### Sample of Findings: Regional Factors
There are regional disparities in well functionality, whether that's from inadequate funding or geographical limitations. Here we see the two regions with the highest percentage of functional wells compared to the two regions with the lowest.
<img src="reports/figures/images/image3.png"
style="text-align: center;"
alt="Markdown Monster icon"
style="float: center;"
width="600"/>

Below you can see a map of Tanzania where each region is darker if it has more non-functioning wells. Notice the brightest region just below the center; that's Iringa, the region with the highest rate of well functionality.

<img src="reports/figures/images/image1.png"
style="text-align: center;"
alt="Markdown Monster icon"
style="float: center;"
width="600"/>



### Data Cleaning
Before modelling, all of our data was run through the process_data notebook, which will generate a new csv file in the data folder. This csv is ready for the scaling and transformations used in the modelling notebook. Duplicate columns were removed, similar columns were combined so that categorical variables were combined if they were not identical in both columns. Our goal was to reduce the dimensionality of our data while preserving as much information as possible.

### Modelling
 The Tanzanian government is struggling to provide water to its growing population. Providing insights into the future status of their water infrastructure will enable them to prepare for future water demands. Our hope in making this model is to provide the government a way to predict the statuses of these points and allocate resource to areas in need based on these predicitons.
<p style="text-align:center;">
    <img src="reports/figures/images/image2.png"
     style="text-align: center;"
     alt="Markdown Monster icon"
     height="400"/>
</p>

#### Measure of Success
In order for our model to be effective, we will be seeking to minimize false negatives. In the context of our model, a false negative would be identifying a water-point as functional when it is in fact non-functional. This is a problem because we don't want to write off non-functional wells and leave people without access to water. False positives are not as much of an issue, because falsely identifying a well as non-functional will only result in an unnecessary maintenance call. For this reason, we will seek to maximize our model's recall score.

The best classification rate for this competition is currently 0.8294. We would ideally like our model to have a minimum recall classification score of .80, out of all the wells that are non-functional, our model successfully classifying 80 percent of them as non-functional.

#### Baseline Model
- For our baseline model we chose to go with a Decision Tree Classifier with default parameters. 

#### Model Evaluation
- We chose to evaluate the performance of our model by recall score, minimizing false negatives.

- Our baseline model had a recall score of 59 percent for the class 'non-functional'. This score provided us with plenty of room for improvement.

#### Feature Engineering 
- Performed feature engineering, binning more features, unbinning features, change imputation value.

- Performed hyper parameter tuning by creating a parameter grid dictionary and running the model through all possible combinations of the parameters in the grid dictionary.

#### Model Iterations
- After performing feature engineering and hyper parameter tuning, ran the model again with the optimal parameters.

- Also implemented a variety of classification algorithms on the dataset.
    - Random Forest Classifier
    - XGBoost Classifer
    - Bagging Classifier
    - LinearSVM
    - MultinomialNB
    
#### Final Model
- Our most successful model was the Random Forest Classifier with parameters found by running Random Forest and a parameter dictionary through a GridSearchCV. 

- The features that were most relevant in the classification prediction were:
    
    ##### Categorical Features
    
    - Funder combined with Installer
    - Public meeting
    - Region (binned)
    - Lga coded
    - Scheme management combined with management
    - Permit
    - Extraction type combined with Extraction type group and Extraction type class
    - Payment
    - Quality Group               
    - Quantity 
    - Source type
    - Waterpoint type
    - Waterpoint decade (added)
    
  ##### Continuous Numerical Features
    - GPS height
    - Population
    
    
- The final model had a recall score for the class 'non-functional' of 82 percent and mean accuracy of 82 percent when validated with the test data.

- Out of all of the wells classified as 'non-functional', the model correctly flagged 82 percent of them.

- The accuracy and recall scores below were generated from the model after oversampling the minority class with SMOTE. We chose to resample with smote due to our previous models being overfit as a consequence of class imbalance in the target variable.


<h1 style="text-align:center;"><b>Final Model Results</b></h1>

|             | precision | recall | f1-score | support |                           |
|-------------|-----------|--------|----------|---------|---------------------------|
| **Class 0** |   0.88    |  0.82  |   0.85   |  5678   | _non-functional_          |
| **Class 1** |   0.43    |  0.78  |   0.55   |  1074   | _functional needs repair_ |
| **Class 2** |   0.88    |  0.83  |   0.86   |  8098   | _functional_              | 


|                           |           |        |          |         |
|---------------------------|-----------|--------|----------|---------|
|  **Accuracy**             |           |        |   0.82   |  14850  |
|  **Macro Average**        |   0.73    |  0.81  |   0.75   |  14850  |
|  **Weighted Average**     |   0.85    |  0.82  |   0.83   |  14850  |
 


### Next steps
- Build dashboard
- improve model performance on 'functional needs repair' class

#### Recommendations
- Acquire more data on water points to analyze current usage and longevity of water point pump.
- Increase funding for regions with more non-functional waterpoints by subsidizing private or community built wells.
- Include the community in the planning and building process of new water points to improve overall care.


### Methods
#### Project Workflow

* Initial EDA
* Removing / Imputing Missing Values
* Dataset Scaling and Encoding
* Baseline classification Model
* Model Evaluation 
 - Feature Engineering 
 - Hyper Parameter Tuning
* Model Iterations
* Final Model
* Explore Findings  
 - Results  
 - Business Recommendations
 
<table>
 <tr>
    <td><b style="font-size:20px">Methods Used</b></td>
    <td><b style="font-size:20px">Technologies Used</b></td>
 </tr>
 <tr>
    <td>Inferential Statistics</td>
    <td>Python</td>
 </tr>
 <tr>
    <td>Machine Learning</td>
    <td>Pandas </td>
 </tr>
 <tr>
    <td>Data Visualization</td>
    <td>SKLearn</td>
 </tr>
 <tr>
    <td>Predictive Modeling</td>
    <td>SciPy</td>
 <tr>
    <td>Classification</td>
    <td>NumPy</td>
 </tr>
 <tr>
    <td> </td>
    <td>MatPlotLib</td>
 </tr>
 <tr>
    <td> </td>
    <td>BorutaPy</td>
 </tr>
 <tr>
    <td> </td>
    <td>Jupyter</td>
 </tr>
 <tr>
    <td> </td>
    <td>Anaconda</td>
 </tr>
</table>

### Want to contribute?
Here's how to get started:
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

2. Download data from the [Driven Data Competition Page](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)

    - You need to create a Driven Data account (it's free) and sign up for [this](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition.


2. Explore the data

    - Follow the process_data notebook in this projects 'notebook' folder and explore the processed data.


4. Load the [Conda Environment](https://github.com/sydroth/tanzaniawellclassification/blob/master/environment.yml) used for the project

#### Repository Structure                   

- [Exploratory Notebooks](https://github.com/sydroth/tanzaniawellclassification/tree/master/notebooks)
  - Walkthroughs of all of our Exploratory Data Analysis
- [Reports](https://github.com/sydroth/tanzaniawellclassification/tree/master/reports)
  - Final report slideshow and notebook
- [References](https://github.com/sydroth/tanzaniawellclassification/tree/master/references)
  - Outside material used for business understanding and context
- [Source Code](https://github.com/sydroth/tanzaniawellclassification/tree/master/src)
  - Functions used in notebooks

#### Members:

|         Name             |                  GitHub               | 
|--------------------------|----------------------------------|
|Syd Rothman               | [sydroth](https://github.com/sydroth)|
|Jacob Prebys              | [jprebys](https://github.com/jprebys)|
|Jason Wong                | [jwong853](https://github.com/jwong853)|
|Maximilian Esterhammer-Fic| [mesterhammerfic](https://github.com/mesterhammerfic)|
