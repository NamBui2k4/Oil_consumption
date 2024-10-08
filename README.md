# Machine learning: Analyzing and Predicting Oil consumption in many countries

- Author: NamBui
- Prefrence style: [Vaibhav3M](https://github.com/Vaibhav3M/Chicago-crime-analysis)

## ➢ I. Abstract

The project is a report conducted with the insights of a student. Oil consumption trend is one of most traditional time series problem.

This project analyzes global oil consumption trends from oil consumption dataset, which spanning from 1995 to 2022. Also, a technique related to RNN is used to predict the trend on the next year (2024).

## ➢ II. Dataset

The dataset chosen for this project includes data points for various countries or entities, detailing both the quantities of proven oil reserves and annual oil consumption measured in terawatt-hours (TWh) or equivalent units. Data is extracted from National Energy Departments and International Organizations such as the International Energy Agency (IEA), Organization of the Petroleum Exporting Countries (OPEC), and the United Nations.


Link: [kaggle](https://www.kaggle.com/datasets/muhammadroshaanriaz/oil-reserves-and-consumption-from-1995-to-2022)

Key features:
- Entity: Names of countries or regions included in the dataset.
- Year: Time period ranging from 1995 to 2022, capturing annual data points.
- Oil Reserves: Quantities of proven oil reserves, typically measured in barrels or metric tons, reflecting the estimated amount of economically recoverable oil.
- Oil Consumption (TWh): Annual oil consumption represented in terawatt-hours (TWh) or equivalent units, indicating the amount of oil utilized for various energy needs including transportation, industrial processes, and residential use.

More info:
- Num of column: 104 - country names
- Num of index: 59 - years from 1965 to 2023
- memory usage: 48.4+ KB

  A Sample of data:

  ![image](https://github.com/user-attachments/assets/0518bd85-0c7e-4f20-b9b6-0e7abcad597f)


## ➢ III. Exploration Analysis

The trend of oil consumption from 1965 to 2023 shows significant growth, with some fluctuations due to economic and geopolitical events. Understanding the trends helps governments and businesses forecast future energy needs. 

In this step, we inferred useful information and analyzed important trends for crime detection and   prevention. The analysis will also help identify useful features for building predictive models.

   - *Methods*: Bar graph, line graph, pie-chart, area graph, statistic, sampling.
  
   - *Technologies*:  pandas, Matplotlib, numpy. 

**1. Statistic**

It takes too many effort to keep tracking all of 104 countries so lets pick random sample to for observation. 

We could start with first 5 countries below: <br /><br />
<div align="center">
	
| <br/>Entity | count<br/> | unique<br/> | top<br/> | freq<br/> |
| :--- | :--- | :--- | :--- | :--- |
| Africa | 59.0 | 59.0 | 342.133700 | 1.0 |
| Africa \(EI\) | 59.0 | 59.0 | 342.133700 | 1.0 |
| Algeria | 59.0 | 59.0 | 15.405252 | 1.0 |
| Argentina | 59.0 | 59.0 | 275.215900 | 1.0 |
| Asia | 59.0 | 59.0 | 2249.215000 | 1.0 |
	
</div><br />

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/945f1772-a8d7-4062-99f5-60a7c8e0a2d0" width="600" margin="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/50ee82e8-c269-4675-9de0-5f34734361f5" width="600"/></td>
  </tr>
</table><br />

Now, going to the bottom of dataset, lets take a look at the last 5 countries:<br /><br />

<div align="center">
	
| <br/>Entity | count<br/> | unique<br/> | top<br/> | freq<br/> |
| :--- | :--- | :--- | :--- | :--- |
| Uzbekistan | 39.0 | 39.0 | 121.316246 | 1.0 |
| Venezuela | 59.0 | 59.0 | 112.023760 | 1.0 |
| Vietnam | 59.0 | 59.0 | 18.011540 | 1.0 |
| Western Africa \(EI\) | 59.0 | 59.0 | 49.667866 | 1.0 |
</div><br />

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/55b07131-ca53-412d-9b51-a6147ac9c776" width="600" margin="120"/></td>
    <td><img src="https://github.com/user-attachments/assets/ac83b885-dbac-4714-a782-7736b0d36086" width="600"/></td>
  </tr>
</table><br />

It seems that Asia has the most numerous consumption in the sample<br />

**2. Check missing value**<br />

<table>
  <div>
    <td><img src="https://github.com/user-attachments/assets/c6c14827-be17-42b6-8476-55230df3e4f5"/></td>
  </div>
</table>

As the above figure shown, almost missing values appear from 1965 to 1985. This could be due to several historical and contextual factors such as Economic and Political Stability, Technological Limitations, or Data Infrastructure.

**3. Analyze top nations**

After investigation nations which possess highest oil consumption values, fifteen typical familiar nations among of them were chosen to analyze:

```python
[
 'United States',
 'China',
 'India',
 'Russia',
 'Japan',
 'South Korea',
 'Brazil',
 'Canada',
 'Singapore',
 'France',
 'United Kingdom',
 'Spain',
'Italy',
 'Thailand',
]
```

Here is what we see:<br />

<table>
  <div>
    <td><img src="https://github.com/user-attachments/assets/324fddd6-0e1e-42d5-96a1-8a95811e597f"/></td>
  </div>
</table><br />

The period from 1965 to 2022 witness a significant increase of USA's consumption following by China. This can be obviously undertanded because United State consumes about 19 millions barrels of oil per day while this number of China is 14 millions. Those number are extremely high for development countries<br />

Although there were few fluctuation, that is still a such numerous number for USA, we can prove that by checking the proportion of USA's consumption compared to the World.<br />

**4. Proportion of USA**<br /><br />

<table>
  <div>
    <td><img src="https://github.com/user-attachments/assets/82c8c4b9-1749-4fb6-97f4-b2c54780743c"/></td>
  </div>
</table>


As we see, the USA'consumption take part in almost 20% of the World. The trends was recorded until 2023. 

This makes USA become leader of eight countries cosuming most oil in the world.

6. Analyze data group by Continents

<table>
  <div>
    <td><img src="https://github.com/user-attachments/assets/b56bbabe-e45e-440b-b233-01412b5ec9d7"/></td>
  </div>
</table>


The above figure shows that Asia reach to highest point of 25000. With rapid economic development and a large population, the demand for oil in this area increase day by day. According to a report by the International Energy Agency (IEA), the Asia-Pacific region accounts for about 34% of total global oil demand 

In the opposite direction, Africa's consumption seems to not over 2500. Africa is not the world's least oil-consuming region, but its oil consumption is much lower than other regions such as Asia and North America.

## ➢ IV. Predicting

 
**Approach and corresponding technologies**

This is timne series problem. Therefor, we consider each country as an input data to put into machine learning model. The output is value of oil consumption in 2024. 

Below is the pipeline we followed:

1. **Data Pre-processing:**: 
      -  Normalize the data
      -  Create recursive data for a country
      -  Dropped missing/null values of recursive data.

In the second stage, We need to decide how many timesteps that an input has. If we have `n` timesteps, the recursive data will have `n + 1` features.

 2. **Predicting**

   *Methods*: k-Fold Cross-Validation, using linear models.
   *Model:*
   - Linear regression
   - Linear regression with Stochastic griadient descent
   - Decision Tree
   - Random forest
   *Technologies*:  GridSearchCV, scikit-learn. 
****

**For example**

Lets take Uzbekistan into account.The timesteps to split data is 5. 

Here is a result of preprocessing:
<br>

```
Entity  Uzbekistan Uzbekistan 2 Uzbekistan 3 Uzbekistan 4 Uzbekistan 5 Uzbekistan 6
1985    121.316246     123.9132    125.95903    130.77919    132.45718    130.16013
1986      123.9132    125.95903    130.77919    132.45718    130.16013    123.54609
1987     125.95903    130.77919    132.45718    130.16013    123.54609     98.57593
1988     130.77919    132.45718    130.16013    123.54609     98.57593     87.02526
1989     132.45718    130.16013    123.54609     98.57593     87.02526       75.798
....	..........    .........	   .........	........      .......	     ......

- train-size: 0.8
- test-size: 0.2
- shape: (34, 6)
```
<h3><pre>1. Linear Regression </pre></h3><br>

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/057b416c-c962-4e08-a69d-092d2bf0875d" height="300"/>
    </td>
    <td>

	Actual: 49.598866, Predicted: 53.888488217568664 

	Actual: 56.866768, Predicted: 53.822199973679346

	Actual: 58.118244, Predicted: 52.79624932173364

	Actual: 62.331253, Predicted: 61.7337395190528

	Actual: 61.463074, Predicted: 59.8863215843685

	Actual: 61.494442, Predicted: 60.97227165464203

	Actual: 61.651016, Predicted: 58.23722435648358
 
</td>
  </tr>
</table>
<br><br>

 <h3><pre>2. SGDRegressor </pre></h3><br>
 
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f82606ec-3371-4cb5-8087-5c9eef2bb80c" height="300"/>
    </td>
    <td>
	    
         Actual: 49.598866, Predicted: 46.722316536756836 

         Actual: 56.866768, Predicted: 47.30630238087184

         Actual: 58.118244, Predicted: 50.477292594681494

         Actual: 62.331253, Predicted: 53.10352718552769

         Actual: 61.463074, Predicted: 56.476213710381785

         Actual: 61.494442, Predicted: 57.37724028267572

         Actual: 61.651016, Predicted: 57.79154287956481


</td>
  </tr>
</table>
<br>

<h3><pre>3. Decision Tree </pre></h3><br>

<br>
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/2f2436bf-9281-4c8f-a22a-b826b8514d60" height="300"/>
    </td>
    <td>
	    
         Actual: 49.598866, Predicted: 44.268963 

         Actual: 56.866768, Predicted: 44.268963

         Actual: 58.118244, Predicted: 45.62668

         Actual: 62.331253, Predicted: 44.268963

         Actual: 61.463074, Predicted: 49.220936

         Actual: 61.494442, Predicted: 49.220936

         Actual: 61.651016, Predicted: 49.220936

</td>
  </tr>
</table><br>

<h3><pre>4. Random forest </pre></h3>

<br><br>
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/ce357e35-a074-47df-a537-6027af6dd6e8" height="300"/>
    </td>
    <td>

         Actual: 49.598866, Predicted: 48.74836252820513 

         Actual: 56.866768, Predicted: 48.74836252820513

         Actual: 58.118244, Predicted: 51.3976141948718

         Actual: 62.331253, Predicted: 51.3976141948718

         Actual: 61.463074, Predicted: 56.13808456666667

         Actual: 61.494442, Predicted: 56.13808456666667

         Actual: 61.651016, Predicted: 56.13808456666667
 
</td>
  </tr>
</table>

 <h3><pre> Metrics </pre></h3><br>

|   Model        	|   MSE  |   MAE  |   R2  | Max error | variance  | 
|----------------	|--------|--------|-------|-----------|-----------| 
|LinearRegression	| 10.109 |  2.681 | 0.428 |   5.322   |    0.548  |
|Random Forest Regressor|  45.540|   6.117| -1.575|   10.934  |    0.541  |
|Decision Tree Regressor| 160.631|  12.204| -8.082|   18.062  |    0.339  |
|SGDRegressor		|  42.847|   6.038| -1.422|   9.560   |    0.639  |

<h3><pre> Choose best model to predict all country </pre></h3><br>


As We saw the result above, seems that Linear Regression performed most effectiveness to predict consumption. We used it to predict all country's figure and here are result:

| Country | 2024 |
|---------|------|
| Algeria | 240.416 |
| Argentina | 377.664 |
| Austria | 139.714 |
| Azerbaijan | 67.340 |
| Bangladesh | 144.969 |
| Belarus | 85.252 |
| Belgium | 317.670 |
| Brazil | 1474.026 |
| Bulgaria | 59.261 |
| Canada | 1279.289 |
| Chile | 235.189 |
| China | 9616.744 |
| Colombia | 280.300 |
| Croatia | 42.596 |
| Cyprus | 29.737 |
| Czechia | 121.880 |
| Denmark | 91.515 |
| Ecuador | 172.077 |
| Egypt | 415.033 |
| Estonia | 15.854 |
| Finland | 97.613 |
| France | 841.030 |
| Germany | 1179.281 |
| Greece | 186.482 |
| Hong Kong | 171.988 |
| Hungary | 96.071 |
| Iceland | 9.602 |
| India | 3094.698 |
| Indonesia | 847.625 |
| Iran | 987.965 |
| Iraq | 473.692 |
| Ireland | 83.055 |
| Israel | 118.800 |
| Italy | 776.741 |
| Japan | 2076.771 |
| Kazakhstan | 192.936 |
| Kuwait | 213.369 |
| Latvia | 20.122 |
| Lithuania | 36.407 |
| Luxembourg | 28.761 |
| Malaysia | 534.970 |
| Mexico | 1047.749 |
| Morocco | 173.059 |
| Netherlands | 481.498 |
| New Zealand | 89.700 |
| North Macedonia | 11.528 |
| Norway | 110.208 |
| Oman | 141.420 |
| Pakistan | 196.419 |
| Peru | 146.844 |
| Philippines | 262.852 |
| Poland | 395.125 |
| Portugal | 130.070 |
| Qatar | 213.733 |
| Romania | 130.757 |
| Russia | 2026.020 |
| Saudi Arabia | 1992.708 |
| Singapore | 879.496 |
| Slovakia | 51.680 |
| Slovenia | 28.377 |
| South Korea | 1514.119 |
| Spain | 725.086 |
| Sri Lanka | 65.410 |
| Sweden | 137.846 |
| Switzerland | 135.801 |
| Taiwan | 454.942 |
| Thailand | 670.948 |
| Trinidad and Tobago | 18.482 |
| Turkey | 627.602 |
| Turkmenistan | 95.117 |
| Ukraine | 120.096 |
| United Arab Emirates | 635.299 |
| United Kingdom | 780.173 |
| United States | 10049.574 |
| Uzbekistan | 60.733 |
| Venezuela | 262.248 |
| Vietnam | 358.461 |

<!--
## ➢ IV. Discussion

- Machine Learning models are good or bad depending on characteristic of data. If the data was not sufficiently hughe enough, using esemble model could not perform high effecient. Instead, using liear regression are more suitable

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

-->
