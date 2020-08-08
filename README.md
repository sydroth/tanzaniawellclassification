
# Tanzania Well Classification

#### A classification project using the [Tanzania Water Well Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) from Driven Data.
![header](references/pexels-kelly-lacy-3030281.jpg)
     
# Our Goal

Our goal is to discover if data aggregated by [Taarifa](http://taarifa.org/) and the [Tanzanian Ministry of Water](https://www.maji.go.tz/) can be used to predict well functionality. The data provided by these resources was collected by Driven Data for a competition which can be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). This project is important because [Tanzania has struggled](https://water.org/our-impact/where-we-work/tanzania/) to provide water to its growing population and if we can predict well functionality, we can provide insight that may be crucial for fixing their infrastructure.

## Objectives
1. Investigate the relationship between water point functionality and these factors:
 - Users, installers, and managers
 - Geographic location
 - Year built
 - Technology used
2. Build a predictive model
3. Report findings


<img src="reports/figures/images/image1.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" 
     height="600"/>

## Table of Contents
1. Context
2. Data Analysis
3. Data Preperation
4. Modelling
5. Results
6. Methods
7. Contribution

### Context
Furthermore, we are attempting to find out exactly which features best determine this functioning status. Numerous studies have been done on what factors contribute to a wells longevity which have found that the management style, location, and technology are all strong indicators. We want to verify this in our data and also find more factors that are accessible given the format of the data.
<p style="text-align:center;">
    <img src="reports/figures/images/image3.png"
     style="text-align: center;"
     alt="Markdown Monster icon"
     style="float: left;"
     height="600"
     width="1200"/>
</p>


### Data Analysis
We were provided with a trianing dataset, training target dataset, and a test values dataset. After reading in the training dataset and target label, we merged these two dataframes together to have the complete training dataframe with the target variable. The target label in this dataset is 'Status_group' which contains the values 'functional', 'functional needs repair', or 'non-functional'.

### Data Cleaning
#### Removing | Imputting Missing Values
- Using inferential statistics, imputted the missing values with the appropriate values for each specific variable. 

#### Dataset Scaling and Encoding
- Decided on whether to keep the variables as continuous numerical features in the model or utilyze them as categorical features by binning the values. 

- Scaled the numerical features and encoded the categorical features.

### Modelling
 The Tanzanian government is struggling to provide water to its growing population. Providing insights into the future status of their water infrastructure will enable them to prepare for future water demands. Our hope in making this model is to provide the government a way to predict the statuses of these points and allocate resource to areas in need based on these predicitons.
<p style="text-align:center;">
    <img src="reports/figures/images/image2.png"
     style="text-align: center;"
     alt="Markdown Monster icon"
     height="600"
     width="1200"/>
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
 


### Explore Findings 
- The functionality of wells is regionally specific, with wells built in the Iringa region have much higher rates of functionality.
- Commercially managed water points were found
- Our model found that installation by a District Water Engineer is a strong indicator for functionality.

#### Recommendations
- Acquire more data on water points to analyze current usage and longevity of water point pump.
- Increase funding for regions with more non-functional waterpoints by subsidizing private or community built wells.
- Include the community in the planning and building process of new water points to improve overall care.


### Methods
#### Project Workflow
<img align="right" width="600" height="600" src="https://lh3.googleusercontent.com/proxy/qSWooR6X3MdeNgJFd7q4u-VqGU7DzSzyjn6vKWhEbqWYABCtI7yqGlJWOLIuCxTckwy2tGxOftpTkOG2eus3JelvnIkCX0BexnrCflC7KhuQg7TOmXp0BZbE1USx">  

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

    a. Follow the process_data notebook in this projects 'notebook' folder and explore the processed data.


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
