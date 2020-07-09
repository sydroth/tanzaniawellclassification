# Tanzania Well Classification
A classification project using the [Tanzania Water Well Data](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) from Driven Data.






## Objective
>Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? This is an intermediate-level practice competition. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.   
<p style="text-align:right;"><i>-DRIVENDATA Project Description</i></p>

### Our Goal

We are attempting to make a model that, given information about water points in Tanzania, can predict the functioning status of that water point. Furthermore, we are attempting to find out exactly which features best determine this functioning status. The Tanzanian government is struggling to provide water to its growing population. Providing insights into the future status of their water infrastructure will enable them to prepare for future water demands. Our hope in making this model is to provide the government a way to predict the statuses of these points.

In order for our model to be effective, we will be seeking to minimize false negatives. In the context of our model, a false negative would be identifying a water-point as functional when it is in fact non-functional. This is a problem because we don't want to write off non-functional wells and leave people without access to water. False positives are not as much of an issue, because falsely identifying a well as non-functional is not a problem. For this reason, we will seek to maximize our model's recall score.

### Repository Structure


### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* Classification


### Technologies
- Python
    - Pandas
    - SKLearn
    - NumPy
    - GeoPandas
    - MatPlotLib
- Jupyter
- Anaconda


## Getting Started








1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

2. Download data from the [Driven Data Competition Page](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
    - You need to create a Driven Data account (it's free) and sign up for [this](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition.


3. Use the data processing scripts found in the projects /src folder

4. 


## Data Cleaning

We had a lot of features in our dataset that posed interesting challenges. 

* The locational columns were grouped by how likely their water points were to be non-functional
* For columns that provided names like 'funder', 'installer', and 'management' we left the most common names, and grouped the rest into an 'other' category
* 



## Modeling

For our first model 








#### Members:

|         Name             |                  GitHub               | 
|--------------------------|----------------------------------|
|Syd Rothman               |  [sydroth](https://github.com/sydroth)|
|Jacob Prebys              | [jprebys](https://github.com/jprebys)|
|Jason Wong                | [jwong853](https://github.com/jwong853)|
|Maximilian Esterhammer-Fic| [mesterhammerfic](https://github.com/mesterhammerfic)|
