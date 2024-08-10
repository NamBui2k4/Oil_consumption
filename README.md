# Project: Analyzing and Predicting Oil consumption in many countries

- Author: NamBui
- Prefrence style: [Vaibhav3M](https://github.com/Vaibhav3M/Chicago-crime-analysis)

# ➢ Abstract

The project is a report conducted with the insights of a student. Oil consumption trend is one of most traditional time series problem.

This project analyzes global oil consumption trends from oil consumption dataset, which spanning from 1995 to 2022. Also, a technique related to RNN is used to predict the trend on the next year (2023).

# ➢ dataset

The dataset includes data points for various countries or entities, detailing both the quantities of proven oil reserves and annual oil consumption measured in terawatt-hours (TWh) or equivalent units.

Link: [kaggle](https://www.kaggle.com/datasets/muhammadroshaanriaz/oil-reserves-and-consumption-from-1995-to-2022)

Key info:
- Num of column: 104 - country names
- Num of index: 59 - years from 1965 to 2023
- memory usage: 48.4+ KB

# ➢ I. Analyzing

The trend of oil consumption from 1965 to 2023 shows significant growth, with some fluctuations due to economic and geopolitical events. Understanding the trends helps governments and businesses forecast future energy needs. 
 
**1. Explore a sample**

It takes too many effort to keep tracking all of 104 countries so lets pick random sample to for observation. 

We could start with first 5 countries below:

| <br/>Entity | count<br/> | unique<br/> | top<br/> | freq<br/> |
| :--- | :--- | :--- | :--- | :--- |
| Africa | 59.0 | 59.0 | 342.133700 | 1.0 |
| Africa \(EI\) | 59.0 | 59.0 | 342.133700 | 1.0 |
| Algeria | 59.0 | 59.0 | 15.405252 | 1.0 |
| Argentina | 59.0 | 59.0 | 275.215900 | 1.0 |
| Asia | 59.0 | 59.0 | 2249.215000 | 1.0 |

Visualization:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/945f1772-a8d7-4062-99f5-60a7c8e0a2d0" width="600" margin="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/50ee82e8-c269-4675-9de0-5f34734361f5" width="600"/></td>
  </tr>
</table>

Now, going to the bottom of dataset, lets take a look at the last sample:

| <br/>Entity | count<br/> | unique<br/> | top<br/> | freq<br/> |
| :--- | :--- | :--- | :--- | :--- |
| Uzbekistan | 39.0 | 39.0 | 121.316246 | 1.0 |
| Venezuela | 59.0 | 59.0 | 112.023760 | 1.0 |
| Vietnam | 59.0 | 59.0 | 18.011540 | 1.0 |
| Western Africa \(EI\) | 59.0 | 59.0 | 49.667866 | 1.0 |


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/55b07131-ca53-412d-9b51-a6147ac9c776" width="600" margin="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/ac83b885-dbac-4714-a782-7736b0d36086" width="600"/></td>
  </tr>
</table>

It seems that Asia has the most numerous consumption in the sample

**2. Check missing value**

 <img src="https://github.com/user-attachments/assets/c6c14827-be17-42b6-8476-55230df3e4f5" width="1000" position="relative" align-items="center" display="flex"/>



3. 

  4. Type of locations where crimes happen the most 
  5. Timelapse of crimes hotspots over the years (2010 - 2019)
  6. A brief literal sense about those crimes
  
 **Predictive Analysis:**
 
  1. Predicting the type of crime(s) and probability of crimes based on location.
  2. Predicting the type of crime(s) based on Time and also on other parameters.



  

# ➢ II. Materials and Methods

**Dataset**

The dataset chosen for this project consists of incidents of crime reported in the city of Chicago from 2001 to 2019. Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. It is one of the richest data sources in the area of crime. 
The dataset includes enough information about Date, Type, Description, location etc about the crime for our analysis.
 
**Approach and corresponding technologies**

The dataset contains 7 million records of the crime. Data of this size needs fast and efficient data processing. We have used Spark framework as its in-memory processing capability makes it easy to deal with data of this volume. We have implemented some techniques such as: k-fold cross validation, KNN, Random Forest, ensemble method and feature selection.

Below is the pipeline we followed:

  1. **Data Pre-processing:** In this step we chose data from year 2010-2019 as the accuracy stabilized for this time period. 
      -  Dropped missing/null values as it accounted for <1% of data. 
      -  Filtered out irrelevant features from the dataset. 
      -  Reduced/merged number of crime types from 32 to 16.
      -  Used Random Over sampling/Under sampling techniques to balance the data.
      
     *Technologies*: Apache Spark, pyspark Dataframe.


2. **Exploration Analysis:** In this step, we inferred useful information and analyzed important trends for crime detection and   prevention. The analysis will also help identify useful features for building predictive models.

   *Methods*: Bar graph, line graph, pie-chart, heatmaps, querying data.
  
   *Technologies*:  pyspark DataFrame, pyspark SQL, pyspark RDD, Matplotlib, Folium, Tableau. 


3. **Predictive Analysis:** Below predictions were tried on both KNN and Random Forest and the results were compared with each other.  Below are the steps involved: 
    -  We used random split, k-Fold Cross-Validation technique while training.
    -  We further trained the model with additional features such as Location Description, Arrest etc. to achieve better accuracy.
    -  We transformed categorical data to binary vectors using One Hot Vector/ Label Encoding.
    -  Used ExtraTreesClassifier, Correlation Matrix/HeatMap, Principal Component Analysis (PCA) as feature selection techniques.
    -  Tuned the hyperparameters such as no of neighbors in KNN and no of trees in Random Forest.
    -  Used an ensemble of different classification models and used soft voting for output.
    
    3.1. **Predicting the type of crime (probabilities) based on its location**: We used latitude, longitude as location to predict the type of crime. We used vector assembler to transform two columns into a vector. 

    3.2. **Predicting the crime based on time(week)**: We used week as a feature to predict the crime based on time. 

     
 
# ➢ III. Results

 <h3><pre>1. Important Preprocessing Steps</pre></h3>

- **Dataset Analysis** - Our dataset was quite imbalanced and had a lot of features. Therefore, we tried making it balanced by merging similar types or dropping insignificant ones. 

<img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/pie-comparison.png" height="400"/> 

- **Feature Extraction** - 
    1. Feature importance in Extra Tree Classifier
    2. Principal Component Analysis 
    3. Correlation Matrix/HeatMap.
   
 **Correlation Matrix/HeatMap -** The heatmap and matrix help us decide features which are in high correlation with Primary Type crime.

<img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/heatmap.png" width="350" height="210"/>  <img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/corelation.png" width="210" height="350"/> 


<h3><pre>2. Exploratory Analysis: </pre></h3>

<img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/image1.gif" width="700" height="360"/>
		<p> Crime hotstops across the past decade</p>

<img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/image8.png"> 
		Trend of crime types across the past decade



<h3><pre>3. Predictive Analysis: </pre></h3>

**Predicting the type of crime(s) and probability of crimes based on location and time data:**

| Prediction Model | Measure  | Location Data Based | Time Data Based |
|------------------|----------|---------------------|-----------------|
| Random Forest    | Accuracy | 26.33%              | 22.65%          |
| Random Forest    | F1 Score | 17.58 %             | 8.37%           |
| KNN              | Accuracy | 29.62%              | 27.7%           |
| KNN              | F1 Score | 25.33%              | 21.2%           |


We concluded that location or time data alone donot provide sufficient details.

**Predicting the type of crime(s) and probability of crimes based on both location and time data:**

**Random Forest Classifier**
 
 Grid-search and k-fold cross validation provided the best params for RF.
 
 <img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/RF-grid.png">
 <img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/RF-parameters.png">
 

 Results: 
 - Accuracy = 36.86%
 - F1 score = 25.42%
          
 
 Additionally providing crime probabilities.
 
 <img src="https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/RF-prob.png">
 
 
 **KNN Classifier:**
 
 Finding optimal K, using the elbow method.

![](https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/knn-elbow.png)



Optimum K = 25
Parameter tuning using Random Search and K-Fold cross-validation:
'weights' =  ‘uniform
‘metric'  = ‘manhattan' (Haversine - in case of Latitude and Longitude)

**Accuracy comparison before and after training model with additonal features**

![](https://github.com/Vaibhav3M/Project-SOEN691-BigData-/blob/master/Exploratory%20Analysis/images/KNN-params.png)


**Impact of sampling on the KNN model:**

Model with no sampling: 
- Accuracy - 33.5%
- F1 Score - 29.6%

**UnderSampling** 
| Sampling Technique | Accuracy | F1 Score | 
|--------------------|----------|----------|
| Cluster Centroids  | 24.92%   | 19.55%   |
| Random             | 24.93%   | 19.55%   |  
			                   

**OverSampling**
| Sampling Technique | Accuracy | F1 Score |
|--------------------|----------|----------|
| SMOTE              | 32.96%   | 29.62%   |
| Random             | 41%      | 31.60%   |   

Random oversampling of minority classes improved the prediction of the model. This could be as the model now better fits the minority data due to availability of a higher number of instances

**Ensemble models - Voting Classifier**

An ensemble of KNeighborsClassifier, RandomForestClassifier, and SVC. We have used soft voting for output. 
Individual accuracy: 
- KNN -> 28.63%  
- RF -> 33.65%
- SVC -> 22.81% 
  
- Overall Ensemble - 35.21%

<h3><pre>4. Comparison of best models from each category:</h3></pre>

| Measures      | Random Forest | KNN (K = 25) | KNN (OverSampling) | Ensemble (KNN, RF, SVM) |
|---------------|---------------|--------------|--------------------|-------------------------|
| Accuracy      | 36.8%         | 33.5%        | 41%                | 35%                     |
| F1-Score      | 25.4%         | 29.6%        | 31.6%              | 26.7%                   |
| Time(Approx.) | 5 mins        | 25 mins      | 30 mins            | 1 hour                  |

KNN (OverSampling) provides the best results. 


# ➢ IV. Discussion

**Relevance of solution**:
- Machine Learning models are as good or as bad as the data you have. Correlation between features is important for predictions. In our case, we experienced low correlation features with our predicting variable. We experimented with different features in order to get better predictions such as using week/month/year to predict crime type based on time and using additional features such as location description, arrest. The results became better, however, not significant enough. 

- The original dataset was highly imbalanced. Even after dropping/merging related some crime types we still had an imbalance of 100:3. Then, we tried sampling techniques for balancing. Random Oversampling gave best results in comparison to other sampling techniques. However, the increase was comparably small. Applying combination of both undersampling and oversampling might result in better overall performance.

- Ensembling various classification models also seemed useful, particularly ‘soft voting’ technique provided better results

- We used Google Colab as development environment.
	- GPU acceleration for scklearn models
	- 25GB RAM 
	- Easy collaboration between team


**Limitations**:
-  Predicting crime patterns have complicated factors, some of them are related to sociology, economics, even history, and geography. The tasks can be further extended to include information about the victims and the offenders are made available.
-  Good predictions are based on two factors: Good Model, but more importantly, good data. Even though we have a big dataset, the features it provides are not good to predict where and when a crime may happen.
-  Not all crimes had a good correlation with parameters such as latitude and longitude. 

**Future work**:
-  Adding data: More data such as economic, demographic and weather data can help make better predictions. 
-  Using models such as XGboost and Neural Network to identify patterns between data.
-  Focus on specific crime types can provide better intuition. 
-  Using a combination of oversampling and undersampling techniques.

## License 


[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://github.com/vaibhav3m)
