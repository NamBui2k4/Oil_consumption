# Machine learning: Analyzing and Creating model for Oil consumption in many countries

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
| Africa | 2398.7463560851425 |
| Africa (EI) | 2398.7464600490175 |
| Algeria | 240.41600024091667 |
| Argentina | 377.6642660179845 |
| Asia | 25136.70390328896 |
| Asia Pacific (EI) | 21835.190430384846 |
| Australia | 618.9428703169297 |
| Austria | 139.7138888406289 |
| Azerbaijan | 67.34001802296316 |
| Bangladesh | 144.9693869165015 |
| Belarus | 85.25233377219489 |
| Belgium | 317.67047141783166 |
| Brazil | 1474.0264857912546 |
| Bulgaria | 59.261321826096825 |
| CIS (EI) | 2618.280447848326 |
| Canada | 1279.2888801489319 |
| Central America (EI) | 313.3561746337289 |
| Chile | 235.18898478713254 |
| China | 9616.744371537556 |
| Colombia | 280.29976921049104 |
| Croatia | 42.596097435107886 |
| Cyprus | 29.73746710464424 |
| Czechia | 121.88016107869666 |
| Denmark | 91.51470913387803 |
| Eastern Africa (EI) | 353.79890368456734 |
| Ecuador | 172.07675801108184 |
| Egypt | 415.03318949911113 |
| Estonia | 15.853866090746848 |
| Europe | 9644.989645772277 |
| Europe (EI) | 8677.351810730723 |
| European Union (27) | 6524.641014260711 |
| Finland | 97.6127469292008 |
| France | 841.0302828660411 |
| Germany | 1179.2807318438822 |
| Greece | 186.48171606632516 |
| High-income countries | 26417.972286959455 |
| Hong Kong | 171.98767858361327 |
| Hungary | 96.07070046908187 |
| Iceland | 9.601569679013611 |
| India | 3094.6975750351776 |
| Indonesia | 847.6254542518564 |
| Iran | 987.9652107590662 |
| Iraq | 473.69166156791664 |
| Ireland | 83.05535955187702 |
| Israel | 118.80021598398336 |
| Italy | 776.7410850200179 |
| Japan | 2076.771213386999 |
| Kazakhstan | 192.93610284596568 |
| Kuwait | 213.36894047820843 |
| Latvia | 20.121568328879977 |
| Lithuania | 36.406914166789534 |
| Lower-middle-income countries | 6043.129925616946 |
| Luxembourg | 28.760865117160854 |
| Malaysia | 534.9698129768909 |
| Mexico | 1047.748876618457 |
| Middle Africa (EI) | 183.3607998434819 |
| Middle East (EI) | 5248.328812662961 |
| Morocco | 173.0585313851887 |
| Netherlands | 481.4975481670581 |
| New Zealand | 89.69993687528495 |
| Non-OECD (EI) | 31446.47709287158 |
| North America | 12786.656753860587 |
| North America (EI) | 12436.449125370553 |
| North Macedonia | 11.528464657585923 |
| Norway | 110.20767257986701 |
| OECD (EI) | 25282.139300335555 |
| Oceania | 721.5591026705856 |
| Oman | 141.42044818016467 |
| Pakistan | 196.41870056173656 |
| Peru | 146.84350142803692 |
| Philippines | 262.8518890423694 |
| Poland | 395.125343435129 |
| Portugal | 130.06985592447433 |
| Qatar | 213.7334857047328 |
| Romania | 130.7568550973533 |
| Russia | 2026.0203892202198 |
| Saudi Arabia | 1992.708335201248 |
| Singapore | 879.496103375622 |
| Slovakia | 51.67963456625902 |
| Slovenia | 28.377282949072317 |
| South Africa | 332.03980003158915 |
| South America | 3216.6078779397544 |
| South Korea | 1514.1188008308889 |
| South and Central America (EI) | 3834.8162774799116 |
| Spain | 725.0858988126645 |
| Sri Lanka | 65.41049047970135 |
| Sweden | 137.84564550309258 |
| Switzerland | 135.80148267182366 |
| Taiwan | 454.94164193266346 |
| Thailand | 670.9475893245553 |
| Trinidad and Tobago | 18.481960415682998 |
| Turkey | 627.6015470658106 |
| Turkmenistan | 95.11651196503126 |
| USSR | 5080.9801487480145 |
| Ukraine | 120.09577827438329 |
| United Arab Emirates | 635.298803999659 |
| United Kingdom | 780.1725779165615 |
| United States | 10049.573813116005 |
| Upper-middle-income countries | 19203.21702425719 |
| Uzbekistan | 60.73344260093528 |
| Venezuela | 262.24838197118396 |
| Vietnam | 358.46136086904875 |
| Western Africa (EI) | 500.9180172159837 |


## ➢ IV. Discussion

- Machine Learning models are good or bad depending on characteristic of data. Correlation between features is important for predictions. In our case, we experienced low correlation features with our predicting variable. We experimented with different features in order to get better predictions such as using week/month/year to predict crime type based on time and using additional features such as location description, arrest. The results became better, however, not significant enough. 

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
