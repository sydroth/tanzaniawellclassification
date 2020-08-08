
# Tanzania Well Classification

#### A classification project using the [Tanzania Water Well Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) from Driven Data.
<img align="right" height="500" src="references/pexels-kelly-lacy-3030281.jpg">  



#### Table of Contents                   


- [Exploratory Notebooks](https://github.com/sydroth/tanzaniawellclassification/tree/master/notebooks)
- [Reports](https://github.com/sydroth/tanzaniawellclassification/tree/master/reports)
- [References](https://github.com/sydroth/tanzaniawellclassification/tree/master/references)
- [Source Code](https://github.com/sydroth/tanzaniawellclassification/tree/master/src)
- [Data Dictionary](https://github.com/sydroth/tanzaniawellclassification/tree/master/references/data_dictionary.txt)


     
## Our Goal

Our goal is to provide a way for water providers in Tanzania to predict the status of water wells. [Tanzania has struggled](https://water.org/our-impact/where-we-work/tanzania/) to provide water to it's citizens and needs to change its approach to water infrastructure. Our tool could provide insights into what needs to be done to overcome this challenge. 
<img src="reports/figures/images/image1.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" 
     height="600"/>



## Objectives
>Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all?  
<p style="text-align:right;"><i>-DRIVENDATA Project Description</i></p>


#### Modelling
 The Tanzanian government is struggling to provide water to its growing population. Providing insights into the future status of their water infrastructure will enable them to prepare for future water demands. Our hope in making this model is to provide the government a way to predict the statuses of these points and allocate resource to areas in need based on these predicitons.
<p style="text-align:center;">
    <img src="reports/figures/images/image2.png"
     style="text-align: center;"
     alt="Markdown Monster icon"
     height="600"
     width="1200"/>
</p>

#### Business Understanding
Furthermore, we are attempting to find out exactly which features best determine this functioning status. Numerous studies have been done on what factors contribute to a wells longevity which have found that the management style, location, and technology are all strong indicators. We want to verify this in our data and also find more factors that are accessible given the format of the data.
<p style="text-align:center;">
    <img src="reports/figures/images/image3.png"
     style="text-align: center;"
     alt="Markdown Monster icon"
     style="float: left;"
     height="600"
     width="1200"/>
</p>



## Measure of Success
In order for our model to be effective, we will be seeking to minimize false negatives. In the context of our model, a false negative would be identifying a water-point as functional when it is in fact non-functional. This is a problem because we don't want to write off non-functional wells and leave people without access to water. False positives are not as much of an issue, because falsely identifying a well as non-functional will only result in an unnecessary maintenance call. For this reason, we will seek to maximize our model's recall score.

The best classification rate for this competition is currently 0.8294. We would ideally like our model to have a minimum recall classification score of .80, out of all the wells that are non-functional, our model successfully classifying 80 percent of them as non-functional.

### Project Outline
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
 

<table border="30">
 <tr>
    <td><b style="font-size:20px">Methods Used</b></td>
    <td><b style="font-size:30px"> </b></td>
    <td><b style="font-size:30px"> </b></td>
    <td><b style="font-size:20px">Technologies Used</b></td>
    <td><b style="font-size:30px"> </b></td>
 </tr>
 <tr>
    <td>Inferential Statistics</td>
    <td> </td>
    <td> </td>
    <td>Python</td>
 </tr>
 <tr>
    <td>Machine Learning</td>
    <td> </td>
    <td> </td>
    <td>Pandas </td>
 </tr>
 <tr>
    <td>Data Visualization</td>
    <td> </td>
    <td> </td>
    <td>SKLearn</td>
 </tr>
 <tr>
    <td>Predictive Modeling</td>
    <td> </td>
    <td> </td>
    <td>SciPy</td>
 <tr>
    <td>Classification</td>
    <td> </td>
    <td> </td>
    <td>NumPy</td>
 </tr>
 <tr>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>MatPlotLib</td>
 </tr>
 <tr>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>BorutaPy</td>
 </tr>
 <tr>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Jupyter</td>
    <tr>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Anaconda</td>
     
 </tr>
</table>




## Getting Started
- A list of the variables and the information they provide can be found here: [Features & Label Descriptions](references/features_&_labels.txt)

#### Initial EDA
We were provided with a trianing dataset, training target dataset, and a test values dataset. After reading in the training dataset and target label, we merged these two dataframes together to have the complete training dataframe with the target variable. The target label in this dataset is 'Status_group' which contains the values 'functional', 'functional needs repair', or 'non-functional'.

Generated a Pandas Profiling Report for an overview of the dataset. The profiling report provides general information on the variables in the dataset. We looked into the interactions, correlations, and missing values of each variable. Variables that were significantly similar to another were either dropped or grouped together. We also dropped variables that had a constant value. You can find the Pandas Profiling Report here: [Profiling Report](references/well_class_report.html)

#### Removing | Imputting Missing Values
- Using inferential statistics, imputted the missing values with the appropriate values for each specific variable. 

#### Dataset Scaling and Encoding
- Decided on whether to keep the variables as continuous numerical features in the model or utilyze them as categorical features by binning the values. 

- Scaled the numerical features and encoded the categorical features.

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

### Business-facing Recommendations
- Acqure more data on water points to analyze current usage and longevity of water point pump.
- Increase funding for regions with more non-functional waterpoints by subsidizing private or community built wells.
- Include the community in the planning and building process of new water points to improve overall care.



### Reproduce Results
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

2. Download data from the [Driven Data Competition Page](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)

    - You need to create a Driven Data account (it's free) and sign up for [this](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition.


2. Explore the data

    a. Follow the process_data notebook in this projects 'notebook' folder and explore the processed data.


4. Load the [Conda Environment](https://github.com/sydroth/tanzaniawellclassification/blob/master/environment.yml) used for the project







#### Members:

|         Name             |                  GitHub               | 
|--------------------------|----------------------------------|
|Syd Rothman               |  [sydroth](https://github.com/sydroth)|
|Jacob Prebys              | [jprebys](https://github.com/jprebys)|
|Jason Wong                | [jwong853](https://github.com/jwong853)|
|Maximilian Esterhammer-Fic| [mesterhammerfic](https://github.com/mesterhammerfic)|
