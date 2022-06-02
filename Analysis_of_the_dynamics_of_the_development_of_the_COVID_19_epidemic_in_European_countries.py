#!/usr/bin/env python
# coding: utf-8

# # **Bachelor's thesis - Analysis of the dynamics of the development of the COVID-19 epidemic in European countries**
# University of Warsaw
# Author: Tomasz Karolewski
# 
# Promoter: Ph.D. Krzysztof Gogolewski 
# 
# Data source: COVID-19 - Johns Hopkins University
# 

# #1. Importing libraries and loading data#
# result: europe_df - DataFrame with only European countries 

# In[ ]:


pip install yellowbrick


# In[ ]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.signal import find_peaks, peak_widths
from yellowbrick.cluster import KElbowVisualizer
from datetime import date
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor


# In[ ]:


get_ipython().run_line_magic('config', 'InlineBackend.print_figure_kwargs={\'facecolor\' : "w"}')


# In[ ]:


url1 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/new_cases_per_million.csv"
url2 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/new_deaths_per_million.csv"

df = pd.read_csv(url1, sep=',')
df = df.set_index("date")
df


# In[ ]:


countries_in_europe = ['Albania','Andorra','Armenia','Austria','Azerbaijan','Belarus','Belgium',
                       'Bosnia and Herzegovina','Bulgaria','Croatia','Cyprus','Czechia',
                       'Denmark','Estonia','Finland','France','Georgia','Germany','Greece','Hungary',
                       'Iceland','Ireland','Italy','Kazakhstan','Latvia','Liechtenstein','Lithuania',
                       'Luxembourg','North Macedonia','Malta','Moldova','Monaco','Montenegro','Netherlands',
                       'Norway','Poland','Portugal','Romania','Russia','San Marino','Serbia','Slovakia',
                       'Slovenia','Spain','Sweden','Switzerland','Turkey','Ukraine','United Kingdom','Vatican']

len(countries_in_europe)


# In[ ]:


#we select only European countries
europe_df = df[list(countries_in_europe)]
europe_df


# #2. Data cleaning and preprocessing#
# Our motivation is to improve the quality of the data before we make any changes to it. Note that the data has blank values and a negative number of new infections.
# 
# The first step will be to remove "small countries", or that one with strange reporting system as they may influence our model: Andorra, Kazakhstan, Liechtenstein, Luxembourg, Malta, Monaco, Montenegro, San Marino, Vatican.
# 
# The second step is to remove the blank rows and replace all negative values with 0.
# 
# The third step is to make rolling average on data from 01.24.2020 to 12.31.2021.
# 
# 
# result: europe_df_rolled

# In[ ]:


#removing countries (and rows with Nans) that doesn't suit our criterias
europe_df_cleaned = europe_df.drop(['Andorra', 'Kazakhstan', 'Liechtenstein', 'Luxembourg', 'Malta', 'Monaco', 'Montenegro', 'San Marino', 'Vatican'], axis=1).dropna(axis=0, how='all').fillna(0)

#changing negative values to 0
for country, value in (europe_df_cleaned < 0).any(axis=0).items():
  if value == True:
    europe_df_cleaned.loc[europe_df_cleaned[country] < 0, country] = 0
europe_df_cleaned


# In[ ]:


def rolling_avg(dataframe, window_size):
    return dataframe.rolling(window_size, center=True, min_periods=int(window_size/2)).mean()


# In[ ]:


#rolling average
europe_df_rolled = rolling_avg(europe_df_cleaned, 14)

d0 = date(2021, 12, 31)
d1 = date.today()
delta = d0 - d1

europe_df_cleaned = europe_df_cleaned[:delta.days+1]

#europe_df_rolled = europe_df_rolled[203:]
europe_df_rolled = europe_df_rolled[:delta.days+1]
europe_df_rolled


# In[ ]:


country = 'Poland'

title_dict = dict(
    text= 'Poland - Covid-19 new cases',
    y=0.9,
    x=0.5,
    xanchor= 'center',
    yanchor= 'top',
    font_size=23, 
    font_family='Arial'
)

legend_dict = dict(
    orientation='h',
    y=-0.35,
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=europe_df_cleaned.index, y=europe_df_cleaned[country], name='Before rolling average'))
fig.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='After rolling average'))
fig.update_layout(title=title_dict, height=400, width=700, legend=legend_dict)
fig.update_xaxes(title='Date')
fig.update_yaxes(title='New cases per milion')
fig.show()
  


# In[ ]:


#plotting line charts before and after moving average
for country in europe_df_cleaned.columns:
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=europe_df_cleaned.index, y=europe_df_cleaned[country], name=f"{country} before rolling average"))
  fig.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=f"{country} after rolling average"))
  fig.show()


# In[ ]:


#plotting all traces
title_dict = dict(
    text= 'Covid-19 new cases',
    y=0.9,
    x=0.5,
    xanchor= 'center',
    yanchor= 'top',
    font_size=23, 
    font_family='Arial'
)

legend_dict = dict(
    orientation='h',
    y=-0.35,
)

countries = ['Belgium', 'Poland', 'Ireland', 'Italy']

fig = go.Figure(layout={"xaxis_title":"Date", "yaxis_title":"New cases per million"})
for country in countries:
  fig.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country))
fig.update_layout(title=title_dict, height=400, width=700, legend=legend_dict)
fig.show()


# #3. Dimension reduction - PCA#
# 
# We select key components that describe the space in 95%
# 
# result: pca_df

# In[ ]:


#implementation of pca with 95% coverage
pca = PCA(n_components=0.95)
data = europe_df_rolled.dropna().T

scaled_data = scale(data, axis=0, with_mean=True, with_std=False)
pca_model = pca.fit(scaled_data)

#plotting bar chart that show what percentage of the space describes the component
per_var = np.round_(pca.explained_variance_ratio_*100, decimals=2)
lab = [f'PC{i}' for i in range(1, len(per_var)+1)]

px.bar(x=lab, y=per_var, labels={"x":"Components", "y":"%"})


# In[ ]:


#transforming data
pca_data = pca.transform(scaled_data)
pca_df = pd.DataFrame(pca_data, columns=lab, index=data.index)
pca_df


# In[ ]:


fig = go.Figure(layout={"xaxis_title":"Date", "yaxis_title":"New cases per million"})
tmp = pd.DataFrame(scaled_data, index=data.index)
country = 'Poland' 
fig.add_trace(go.Scatter(x=tmp.T.index, y=tmp.T[country], name=country))
fig.show()


# In[ ]:


fig = go.Figure(layout={"xaxis_title":"Date", "yaxis_title":"New cases per million"})
tmp = pd.DataFrame(pca_data, index=data.index)
for country in data.index:
  fig.add_trace(go.Scatter(x=tmp.T.index, y=tmp.T[country], name=country))
fig.show()


# In[ ]:


fig = go.Figure(layout={"xaxis_title":"Date", "yaxis_title":"New cases per million"})
country='Albania'
fig.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country))
fig.show()


# In[ ]:


fig = pd.plotting.scatter_matrix(pca_df.iloc[:,:4], alpha=1, figsize=(6, 6), diagonal='hist')
plt.tight_layout()


# In[ ]:


u, s, vt = np.linalg.svd(scaled_data, full_matrices=True)
U = pd.DataFrame(vt)
px.line(U.T[0])


# In[ ]:


fig = go.Figure(layout={"xaxis_title":"Date", "yaxis_title":"New cases per million"})
tmp = pd.DataFrame(vt)
for country in range(4):
  fig.add_trace(go.Scatter(x=tmp.T.index, y=tmp.T[country], name=country))
fig.show()


# #4. pca_df clustering
# Division of the pandemic in countries into clusters

# ##a) KMeans##
# 
# Overview of divisions depending on the number of clusters

# In[ ]:


def kmeans_clustering(data):
  #loop responsible for generating 3-7 figures
  for n in range(2, 7):
    print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
    kmeans = KMeans(n_clusters=n+1, random_state=100).fit(data)
    kmeans.labels_

    clusters_list = []
    #loop responsible for creating list of lists of countries splited by clustering and printing them
    for iterator in range(n+1):
      print(f"Cluster {iterator+1}: ")
      cluster_list = []
      for iterator2, country in enumerate(data.index):
        if kmeans.labels_[iterator2] == iterator:
          cluster_list.append(country)
      print(cluster_list, end="\n")
      clusters_list.append(cluster_list)


    #setting choropleth parameters
    config = dict(
      type = 'choropleth',
      locations = data.index.values,
      locationmode='country names',
      z=np.append(kmeans.labels_+1, 7),
      colorscale='sunset',
      marker_line_color='black',
      marker_line_width=0.5,
      colorbar_title = 'Klastry')

    #plotting first part of the figure
    fig = go.Figure(data=[config])
    fig.update_geos(scope="world", lataxis_showgrid=True, lonaxis_showgrid=True, projection_type="mercator", lataxis_range=[40,75], lonaxis_range=[-30, 70])
    fig.update_layout(height=300, margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()

    #plotting second part of the figure
    fig2 = subplots.make_subplots(rows=int(n/3)+1, cols=3, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))])
    for iterator, cluster in enumerate(clusters_list):
      #calculating mean for each cluster
      frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                          columns=['Mean'], index=europe_df_rolled.index)
      for country in cluster:
        fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country), row=int(iterator/3)+1, col=iterator%3 +1)
        fig2.update_xaxes(title_text="Date", row=int(iterator/3)+1, col=iterator%3 +1)
        fig2.update_yaxes(title_text="New cases per milion", row=int(iterator/3)+1, col=iterator%3 +1, range=[0,2700])
        
      #adding trace of mean
      fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', line=dict(color='black', width=3, dash='dash')), row=int(iterator/3)+1, col=iterator%3 +1)

        
    fig2.update_layout(height=(int(n/3)+1)*300) 
    fig2.show()

    print("\n")

kmeans_clustering(pca_df)


# ##b) KHierarchy##
# 
# Overview of divisions depending on the number of clusters
# **works better than KMeans**

# In[ ]:


def khierarchy_clustering(data):
  #loop responsible for generating 3-7 figures
  for n in range(2, 7):
    print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
    hierarchy = AgglomerativeClustering(n_clusters=n+1, linkage='ward', affinity='euclidean').fit(data)
    hierarchy.labels_

    clusters_list = []
    #loop responsible for creating list of lists of countries splited by clustering and printing them
    for iterator in range(n+1):
      print(f"Cluster {iterator+1}: ")
      cluster_list = []
      for iterator2, country in enumerate(data.index):
        if hierarchy.labels_[iterator2] == iterator:
          cluster_list.append(country)
      print(cluster_list, end="\n")
      clusters_list.append(cluster_list)


    #setting choropleth parameters
    config = dict(
      type = 'choropleth',
      locations = data.index.values,
      locationmode='country names',
      z=np.append(hierarchy.labels_+1, 7),
      colorscale='sunset',
      marker_line_color='black',
      marker_line_width=0.5,
      colorbar_title = 'Klastry')

    #plotting first part of the figure
    fig = go.Figure(data=[config])
    fig.update_geos(scope="world", lataxis_showgrid=True, lonaxis_showgrid=True,
                    projection_type="mercator", lataxis_range=[40,75], lonaxis_range=[-30, 70])
    fig.update_layout(height=300, margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()

    #plotting second part of the figure
    fig2 = subplots.make_subplots(rows=int(n/3)+1, cols=3, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))])
    for iterator, cluster in enumerate(clusters_list):
      #calculating mean for each cluster
      frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), columns=['Mean'], index=europe_df_rolled.index)
      for country in cluster:
        fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country), 
                      row=int(iterator/3)+1, col=iterator%3 +1)
        fig2.update_xaxes(title_text="Date", row=int(iterator/3)+1, col=iterator%3 +1)
        fig2.update_yaxes(title_text="New cases per milion", row=int(iterator/3)+1, col=iterator%3 +1, range=[0,2700])

      #adding trace of mean
      fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', line=dict(color='black', width=3, dash='dash')), row=int(iterator/3)+1, col=iterator%3 +1)

    fig2.update_layout(height=(int(n/3)+1)*300) 
    fig2.show()

    print("\n")
khierarchy_clustering(data)


# ##c) Selection of the number of clusters
# The most optimal number of clusters for KMeans is 3, while for Hclust it is 4

# In[ ]:


#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(pca_df)
visualizer.show()


# In[ ]:


#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(pca_df)        
visualizer.show()


# #5. Waves extraction#
# The motivation is to isolate the waves of the pandemic and re-clusters on them
# 
# result: waves_df, train_df, predict_df, validate_df

# In[ ]:


#function to help me save data about waves
def create_wave_df_column(country, real_height, left_side_base, right_side_base, left_x, left_y):

  wave_data = pd.DataFrame(columns=[country])
  train_data = pd.DataFrame(columns=[country])
  predict_data = pd.DataFrame(columns=[country])
  validate_data = pd.DataFrame(columns=[country])

  #adding height rows
  for iterator, item in enumerate(real_height):
    new_row = pd.Series(data={country:item}, name=f"height{iterator}")
    wave_data = wave_data.append(new_row, ignore_index=False)
    if iterator + 1 == len(real_height):
      new_row = pd.Series(data={country:item}, name=f"height")
      predict_data = predict_data.append(new_row, ignore_index=False)
    else: 
      train_data = train_data.append(new_row, ignore_index=False)
    if iterator + 2 == len(real_height):
      new_row = pd.Series(data={country:item}, name=f"height")
      validate_data = validate_data.append(new_row, ignore_index=False)
  
  #adding left base rows
  for iterator, item in enumerate(left_side_base):
    new_row = pd.Series(data={country:item}, name=f"left_base{iterator}")
    wave_data = wave_data.append(new_row, ignore_index=False)
    if iterator + 1 == len(left_side_base):
      new_row = pd.Series(data={country:item}, name=f"left_base")
      predict_data = predict_data.append(new_row, ignore_index=False)
    else: 
      train_data = train_data.append(new_row, ignore_index=False)
    if iterator + 2 == len(left_side_base):
      new_row = pd.Series(data={country:item}, name=f"left_base")
      validate_data = validate_data.append(new_row, ignore_index=False)

  #adding right base rows
  for iterator, item in enumerate(right_side_base):
    new_row = pd.Series(data={country:item}, name=f"right_base{iterator}")
    wave_data = wave_data.append(new_row, ignore_index=False)
    train_data = train_data.append(new_row, ignore_index=False)

  #adding left point x
  for iterator, item in enumerate(left_x):
    new_row = pd.Series(data={country:item}, name=f"left_x{iterator}")
    wave_data = wave_data.append(new_row, ignore_index=False)
    if iterator + 1 == len(left_x):
      new_row = pd.Series(data={country:item}, name=f"left_x")
      predict_data = predict_data.append(new_row, ignore_index=False)
    else: 
      train_data = train_data.append(new_row, ignore_index=False)
    if iterator + 2 == len(left_x):
      new_row = pd.Series(data={country:item}, name=f"left_x")
      validate_data = validate_data.append(new_row, ignore_index=False)

  #adding left point y
  for iterator, item in enumerate(left_y):
    new_row = pd.Series(data={country:item}, name=f"left_y{iterator}")
    wave_data = wave_data.append(new_row, ignore_index=False)
    if iterator + 1 == len(left_y):
      new_row = pd.Series(data={country:item}, name=f"left_y")
      predict_data = predict_data.append(new_row, ignore_index=False)
    else: 
      train_data = train_data.append(new_row, ignore_index=False)
    if iterator + 2 == len(left_y):
      new_row = pd.Series(data={country:item}, name=f"left_y")
      validate_data = validate_data.append(new_row, ignore_index=False)

  return wave_data, train_data, predict_data, validate_data


# In[ ]:


#test
tmp1, train_tmp, pred_tmp, val_tmp = create_wave_df_column('Poland', ['20', '30'], [17, 19, 30], [27, 67], [4], [5])
#tmp2 = create_wave_df_column('Holand', ['20', '30', '46'], [76, 5], [56, 78], [6], [7])
#pd.merge(left=tmp1, right=tmp2, how="outer", left_index=True, right_index=True, sort=False)
pred_tmp


# In[ ]:


waves_df = pd.DataFrame()
train_df = pd.DataFrame()
predict_df = pd.DataFrame()
validate_df = pd.DataFrame()

for country in europe_df_rolled.columns:
  tmp_df = europe_df_rolled[country].copy().reset_index(drop=True)

  #temporary changing first and last row for better peaks extraction
  first_day = float(tmp_df.iloc[:1])
  last_day = float(tmp_df.iloc[-1:])

  tmp_df[0] = 0
  tmp_df[len(tmp_df)-1] = 0

  peaks, _ = find_peaks(tmp_df, prominence=35, distance=60)

  results_full = peak_widths(tmp_df, peaks, rel_height=1, wlen=250)
  results_full[0]

  print("\n")
  print("\033[1m peaks before removing subpeaks: \033[0m", peaks, sep=' ')

  #removing subpeaks
  if len(peaks) > 1:
    index_to_del = []
    #if first peak is inside right part of next peak  
    if results_full[2][0] >= results_full[2][1] and results_full[3][0] <= results_full[3][1]:
      index_to_del.append(0)

    for iterator in range(1, len(peaks)-1):
      #if peak is inside left part of previous peak
      if results_full[2][iterator] >= results_full[2][iterator-1] and results_full[3][iterator] <= results_full[3][iterator-1]:
        index_to_del.append(iterator)

      #if peak is inside right part of next peak  
      if results_full[2][iterator] >= results_full[2][iterator+1] and results_full[3][iterator] <= results_full[3][iterator+1]:
        index_to_del.append(iterator)

    iterator = len(peaks)-1
    #if last peak is inside left part of previous peak
    if results_full[2][iterator] >= results_full[2][iterator-1] and results_full[3][iterator] <= results_full[3][iterator-1]:
      index_to_del.append(iterator)
    
    index_to_del = list(set(index_to_del))
    print("\033[1m subpeaks indexes: \033[0m", index_to_del, sep=' ')

    peaks = np.delete(peaks, index_to_del)
    results_full = np.delete(np.array(results_full), index_to_del, axis=1)

    print("\033[1m peaks: \033[0m", peaks, sep=' ')



  #calculating triangle points
  base_left_point = dict()
  base_left_point['x'] = results_full[2]
  base_left_point['y'] = tmp_df[base_left_point['x'].astype(int)].values

  base_right_point = dict()
  base_right_point['x'] = results_full[3]
  base_right_point['y'] = base_left_point['y']

  peak_point = dict()
  peak_point['x'] = peaks
  peak_point['y'] = tmp_df[peaks].values

  #calculating the length of the sides of the triangle
  real_height = peak_point['y'] - base_left_point['y']
  left_side_base = peak_point['x'] - base_left_point['x']
  right_side_base = base_right_point['x'] - peak_point['x']

  #calculating angles
  left_angle = np.degrees(np.arctan(real_height/left_side_base))
  right_angle = np.degrees(np.arctan(real_height/right_side_base))

  #removing last right sides because this is unknown
  right_side_base = right_side_base[:-1]
  right_angle = right_angle[:-1]
  
  print("\033[1m left angles: \033[0m", left_angle, sep=' ')
  print("\033[1m right angles: \033[0m", right_angle, sep=' ')

  #saving wave data
  wave_column, train_column, predict_column, validate_column = create_wave_df_column(country, real_height, left_side_base, right_side_base, base_left_point['x'], base_left_point['y'])
  waves_df = pd.merge(left=waves_df, right=wave_column, how="outer", left_index=True, right_index=True, sort=False)
  train_df = pd.merge(left=train_df, right=train_column, how="outer", left_index=True, right_index=True, sort=False)
  predict_df = pd.merge(left=predict_df, right=predict_column, how="outer", left_index=True, right_index=True, sort=False)
  validate_df = pd.merge(left=validate_df, right=validate_column, how="outer", left_index=True, right_index=True, sort=False)


  #returing original data for plotting
  tmp_df[0] = first_day
  tmp_df[len(tmp_df)-1] = last_day
  
  #ploting
  plt.plot(tmp_df)
  plt.title(country)

  for iterator in range(len(peaks)-1):
    plt.plot([base_left_point['x'][iterator], peak_point['x'][iterator]], 
             [base_left_point['y'][iterator], peak_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.plot([peak_point['x'][iterator], base_right_point['x'][iterator]], 
             [peak_point['y'][iterator], base_right_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.plot([peak_point['x'][iterator], peak_point['x'][iterator]], 
             [base_left_point['y'][iterator], peak_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.hlines(base_left_point['y'][iterator], 
               base_left_point['x'][iterator], 
               base_right_point['x'][iterator], 
               color='black', linestyle='--')
  

  #last peak only have left side because right is unknown
  iterator = len(peaks)-1
  plt.plot([base_left_point['x'][iterator], peak_point['x'][iterator]], 
           [base_left_point['y'][iterator], peak_point['y'][iterator]], 
           color='black', linestyle='--')
  
  plt.plot([peak_point['x'][iterator], peak_point['x'][iterator]], 
           [base_left_point['y'][iterator], peak_point['y'][iterator]], 
           color='black', linestyle='--')
  
  plt.hlines(base_left_point['y'][iterator], 
             base_left_point['x'][iterator], 
             peak_point['x'][iterator], 
             color='black', linestyle='--')
  
  plt.show()


# In[ ]:


waves_df


# #6. waves_df clustering
# Division of the pandemic in countries into clusters

# In[ ]:


waves_df.fillna(0)


# ##a) KMeans

# In[ ]:


kmeans_clustering(waves_df.fillna(0).T)


# ##b) KHierarchy

# In[ ]:


khierarchy_clustering(waves_df.fillna(0).T)


# ##c) Selection of the number of clusters

# In[ ]:


#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(waves_df.fillna(0).T)
visualizer.show()


# In[ ]:


#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(waves_df.fillna(0).T)        
visualizer.show()


# # 7. pca_df + waves_df clustering
# There is no difference between pca_df and pca_df + waves_df

# In[ ]:


pca_waves_df = pd.merge(left=pca_df, right=waves_df.fillna(0).T, how="outer", left_index=True, right_index=True, sort=False)
pca_waves_df


# ##a) KMeans

# In[ ]:


kmeans_clustering(pca_waves_df)


# ##b) KHierarchy

# In[ ]:


khierarchy_clustering(pca_waves_df)


# ##c) Wybór liczby klastrów

# In[ ]:


#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(pca_waves_df)
visualizer.show()


# In[ ]:


#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True)
visualizer.fit(pca_waves_df)        
visualizer.show()


# #8. Prediction of the end of the last wave and the rate of its decay
# Overview of the different models

# In[ ]:


train_df_prepared = pd.DataFrame()

for iterator in range(int(len(train_df.T.columns)/5)):
  tmp_df = train_df.fillna(0).T[[f"height{iterator}", f"left_base{iterator}", f"left_x{iterator}", f"left_y{iterator}", f"right_base{iterator}"]]

  train_df_prepared = pd.concat([train_df_prepared, pd.DataFrame(tmp_df.values)], axis=0)

train_df_prepared.columns=['height', 'left_base', 'left_x', 'left_y', 'right_base']
train_df_prepared.drop_duplicates(inplace=True)
train_df_prepared = train_df_prepared.reset_index(drop=True)
X_train_df, y_train_df = train_df_prepared.iloc[:,:4], train_df_prepared.iloc[:,4:]


# In[ ]:


y_train_df


# In[ ]:


models = [
          AdaBoostRegressor(),
          LGBMRegressor(),
          ExtraTreesRegressor(),
          BaggingRegressor(),
          GradientBoostingRegressor(),
          RandomForestRegressor(),
          HistGradientBoostingRegressor(),
          SVR(),
          LinearRegression(),
          XGBRegressor(),
          
          RandomForestRegressor(n_estimators=500, random_state=100), #ten wygrywa w model.score
          RandomForestRegressor(n_estimators=500, criterion='absolute_error', min_samples_leaf=5, oob_score=True, random_state=100), #najlepsze parametry z gridCV       
          RandomForestRegressor(n_estimators=200, criterion='absolute_error', min_samples_leaf=5, oob_score=True, random_state=100), #najlepsze parametry z gridCV

          SVR(C=0.25, epsilon=0.2, kernel='linear'), #najlepsze parametry z gridCV
          AdaBoostRegressor(n_estimators=500, learning_rate=0.5, loss='exponential', random_state=100), #najlepsze parametry z gridCV
          XGBRegressor(n_estimators=500, booster='gblinear', learning_rate=0.33, max_depth=1, min_child_weight=0, n_jobs=-1, objective='reg:squarederror', random_state=100) #najlepsze parametry z gridCV
]

for model in models:
  cv_scores = cross_val_score(model, X_train_df, y_train_df.values.ravel(), cv=LeaveOneOut(), scoring='neg_mean_absolute_error', n_jobs=-1)
  cv_scores = np.absolute(cv_scores)
  #best is score 0
  print('Model:', model, ', MAE: \033[91m %.3f (%.3f) \033[0m' % (cv_scores.mean(), cv_scores.std()))
  model.fit(X_train_df, y_train_df.values.ravel())
  #best is score 1
  print('R^2 on training data: \033[91m %.3f \033[0m' % (model.score(X_train_df, y_train_df.values.ravel())))


# In[ ]:


parameters = dict(
criterion=['absolute_error'],
min_samples_split=[2, 5, 10, 20, 30, 40, 50],
min_samples_leaf=[1, 5, 10, 20, 30, 40 ,50, 60, 70],
oob_score=[True],
random_state=[100]
)

grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters, cv=LeaveOneOut(), n_jobs=-1, scoring='neg_mean_absolute_error', verbose=3)
grid.fit(X_train_df, y_train_df.values.ravel())

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)

pd.DataFrame(grid.cv_results_)


# In[ ]:


parameters = dict(
booster=['gbtree', 'gblinear', 'dart'],
learning_rate=[0.33, 0.66, 1],
gamma=[0, 0.25, 0.5, 0.75],
max_depth=[1, 4, 8],
min_child_weight=[0, 1, 3],
n_jobs=[-1],
objective =['reg:squarederror'],
random_state=[100]
)

grid = GridSearchCV(estimator=XGBRegressor(), param_grid=parameters, cv=LeaveOneOut(), n_jobs=-1, scoring='neg_mean_absolute_error', verbose=3)
grid.fit(X_train_df, y_train_df.values.ravel())

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)

pd.DataFrame(grid.cv_results_)


# In[ ]:


parameters = dict(
kernel=['linear', 'rbf', 'poly'],
C=[0.25, 0.5, 1, 1.5, 3],
epsilon=[ 0.025, 0.05, 0.1, 0.2],
max_iter=[-1]
)

grid = GridSearchCV(estimator=SVR(), param_grid=parameters, cv=LeaveOneOut(), n_jobs=-1, scoring='neg_mean_absolute_error')
grid.fit(X_train_df, y_train_df.values.ravel())

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)

pd.DataFrame(grid.cv_results_)


# In[ ]:


parameters = dict( 
learning_rate=[0.5, 1, 2, 3, 4, 5, 6], 
loss=['linear', 'square', 'exponential'],
random_state=[100]
)

grid = GridSearchCV(estimator=AdaBoostRegressor(), param_grid=parameters, cv=LeaveOneOut(), n_jobs=-1, scoring='neg_mean_absolute_error', verbose=3)
grid.fit(X_train_df, y_train_df.values.ravel())

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)

pd.DataFrame(grid.cv_results_)


# #9. Results submit

# In[ ]:


predict_df


# Kmeans, Hclust for pca

# In[ ]:


#@title
#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(pca_df)
visualizer.show()

#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(pca_df)
visualizer.show()


# In[ ]:


#@title
#n+1 number of clusters
n=3
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = KMeans(n_clusters=n+1, random_state=100).fit(data)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)

clusters_list_km_pca = clusters_list.copy()


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.25, "rgb(255, 196, 51)"),
              (0.25, "rgb(255, 51, 119)"), (0.5, "rgb(255, 51, 119)"),
              (0.5, "rgb(219, 51, 255)"),  (0.75, "rgb(219, 51, 255)"),
              (0.75, "rgb(51, 189, 255)"),  (1.00, "rgb(51, 189, 255)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=4, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to k-means clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.06)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1200, width=800, margin=config_margin, legend_tracegroupgap=55, title=config_title)
fig2.show()

print("\n")


# In[ ]:


#@title
#n+1 number of clusters
n=4
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = AgglomerativeClustering(n_clusters=n+1, linkage='ward', affinity='euclidean').fit(data)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.2, "rgb(255, 196, 51)"),
              (0.2, "rgb(255, 51, 119)"), (0.4, "rgb(255, 51, 119)"),
              (0.4, "rgb(219, 51, 255)"),  (0.6, "rgb(219, 51, 255)"),
              (0.6, "rgb(51, 189, 255)"),  (0.8, "rgb(51, 189, 255)"),
              (0.8, "rgb(51, 255, 53)"),  (1.0, "rgb(51, 255, 53)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=5, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to hierarchical clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.06)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1400, width=800, margin=config_margin, legend_tracegroupgap=55, title=config_title)
fig2.show()

print("\n")


# Prediction

# In[ ]:


model = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, n_estimators=500, oob_score=True, random_state=100)
#model = XGBRegressor(booster='gblinear', learning_rate=0.33, max_depth=1, min_child_weight=0, n_estimators=500, n_jobs=-1, objective='reg:squarederror', random_state=100)


model.fit(X_train_df, y_train_df.values.ravel())

predictions = pd.DataFrame(model.predict(predict_df.T), index=predict_df.T.index, columns=['predicted_right_base'])
validation = pd.DataFrame(model.predict(validate_df.T), index=validate_df.T.index, columns=['predicted_right_base'])

pd.concat([predict_df.T, predictions], axis=1)


# In[ ]:


#@title
waves_df = pd.DataFrame()
train_df = pd.DataFrame()
predict_df = pd.DataFrame()
validate_df = pd.DataFrame()

for country in europe_df_rolled.columns:
  tmp_df = europe_df_rolled[country].copy().reset_index(drop=True)

  #temporary changing first and last row for better peaks extraction
  first_day = float(tmp_df.iloc[:1])
  last_day = float(tmp_df.iloc[-1:])

  tmp_df[0] = 0
  tmp_df[len(tmp_df)-1] = 0

  peaks, _ = find_peaks(tmp_df, prominence=35, distance=60)

  results_full = peak_widths(tmp_df, peaks, rel_height=1, wlen=250)
  results_full[0]

  print("\n")
  print("\033[1m peaks before removing subpeaks: \033[0m", peaks, sep=' ')

  #removing subpeaks
  if len(peaks) > 1:
    index_to_del = []
    #if first peak is inside right part of next peak  
    if results_full[2][0] >= results_full[2][1] and results_full[3][0] <= results_full[3][1]:
      index_to_del.append(0)

    for iterator in range(1, len(peaks)-1):
      #if peak is inside left part of previous peak
      if results_full[2][iterator] >= results_full[2][iterator-1] and results_full[3][iterator] <= results_full[3][iterator-1]:
        index_to_del.append(iterator)

      #if peak is inside right part of next peak  
      if results_full[2][iterator] >= results_full[2][iterator+1] and results_full[3][iterator] <= results_full[3][iterator+1]:
        index_to_del.append(iterator)

    iterator = len(peaks)-1
    #if last peak is inside left part of previous peak
    if results_full[2][iterator] >= results_full[2][iterator-1] and results_full[3][iterator] <= results_full[3][iterator-1]:
      index_to_del.append(iterator)
    
    index_to_del = list(set(index_to_del))
    print("\033[1m subpeaks indexes: \033[0m", index_to_del, sep=' ')

    peaks = np.delete(peaks, index_to_del)
    results_full = np.delete(np.array(results_full), index_to_del, axis=1)

    print("\033[1m peaks: \033[0m", peaks, sep=' ')



  #calculating triangle points
  base_left_point = dict()
  base_left_point['x'] = results_full[2]
  base_left_point['y'] = tmp_df[base_left_point['x'].astype(int)].values

  base_right_point = dict()
  base_right_point['x'] = results_full[3]
  base_right_point['y'] = base_left_point['y']

  peak_point = dict()
  peak_point['x'] = peaks
  peak_point['y'] = tmp_df[peaks].values

  #calculating the length of the sides of the triangle
  real_height = peak_point['y'] - base_left_point['y']
  left_side_base = peak_point['x'] - base_left_point['x']
  right_side_base = base_right_point['x'] - peak_point['x']

  #calculating angles
  left_angle = np.degrees(np.arctan(real_height/left_side_base))
  right_angle = np.degrees(np.arctan(real_height/right_side_base))

  #removing last right sides because this is unknown
  right_side_base = right_side_base[:-1]
  right_angle = right_angle[:-1]
  
  print("\033[1m left angles: \033[0m", left_angle, sep=' ')
  print("\033[1m right angles: \033[0m", right_angle, sep=' ')

  #saving wave data
  wave_column, train_column, predict_column, validate_column = create_wave_df_column(country, real_height, left_side_base, right_side_base, base_left_point['x'], base_left_point['y'])
  waves_df = pd.merge(left=waves_df, right=wave_column, how="outer", left_index=True, right_index=True, sort=False)
  train_df = pd.merge(left=train_df, right=train_column, how="outer", left_index=True, right_index=True, sort=False)
  predict_df = pd.merge(left=predict_df, right=predict_column, how="outer", left_index=True, right_index=True, sort=False)
  validate_df = pd.merge(left=validate_df, right=validate_column, how="outer", left_index=True, right_index=True, sort=False)


  #returing original data for plotting
  tmp_df[0] = first_day
  tmp_df[len(tmp_df)-1] = last_day
  
  #ploting
  plt.plot(tmp_df, label='New cases')
  plt.title(f"{country} - peaks and prediction", fontsize=16)
  plt.ylim([0,3000])
  plt.xlim([0,850])
  plt.xlabel('Day', fontsize=13)
  plt.ylabel('New cases per milion', fontsize=13)

  for iterator in range(len(peaks)-1):
    plt.plot([base_left_point['x'][iterator], peak_point['x'][iterator]], 
             [base_left_point['y'][iterator], peak_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.plot([peak_point['x'][iterator], base_right_point['x'][iterator]], 
             [peak_point['y'][iterator], base_right_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.plot([peak_point['x'][iterator], peak_point['x'][iterator]], 
             [base_left_point['y'][iterator], peak_point['y'][iterator]], 
             color='black', linestyle='--')
    
    plt.hlines(base_left_point['y'][iterator], 
               base_left_point['x'][iterator], 
               base_right_point['x'][iterator], 
               color='black', linestyle='--')
  

  #last peak only have left side because right is unknown
  iterator = len(peaks)-1
  plt.plot([base_left_point['x'][iterator], peak_point['x'][iterator]], 
           [base_left_point['y'][iterator], peak_point['y'][iterator]], 
           color='black', linestyle='--', label='Wave')
  
  plt.plot([peak_point['x'][iterator], peak_point['x'][iterator]], 
           [base_left_point['y'][iterator], peak_point['y'][iterator]], 
           color='black', linestyle='--')
  
  plt.hlines(base_left_point['y'][iterator], 
             base_left_point['x'][iterator], 
             peak_point['x'][iterator], 
             color='black', linestyle='--')
  
  #predictions
  plt.plot([peak_point['x'][iterator], peak_point['x'][iterator] + predictions.T[country]], 
           [peak_point['y'][iterator], base_left_point['y'][iterator]], 
           color='red', linestyle=(0, (5, 7)), label='Predicition')

  plt.hlines(base_left_point['y'][iterator], 
             peak_point['x'][iterator], 
             peak_point['x'][iterator] + predictions.T[country],
             color='red', linestyle=(0, (5, 7)))
  
  #validation
  iterator = len(peaks)-2
  plt.plot([peak_point['x'][iterator], peak_point['x'][iterator] + validation.T[country]], 
           [peak_point['y'][iterator], base_left_point['y'][iterator]], 
           color='red', linestyle=(0, (5, 6)))

  plt.hlines(base_left_point['y'][iterator], 
             peak_point['x'][iterator], 
             peak_point['x'][iterator] + validation.T[country],
             color='red', linestyle=(0, (5, 6)))
  
  plt.legend(frameon=True, framealpha=1, edgecolor='black')
  plt.show()


# Kmeans, hclust for triangles

# In[ ]:


#@title
#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(waves_df.fillna(0).T)
visualizer.show()

#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(waves_df.fillna(0).T)        
visualizer.show()


# In[ ]:


#@title
#n+1 number of clusters
n=2
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = KMeans(n_clusters=n+1, random_state=100).fit(waves_df.fillna(0).T)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)

clusters_list_km_triangle = clusters_list.copy()


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.33, "rgb(255, 196, 51)"),
              (0.33, "rgb(255, 51, 119)"), (0.66, "rgb(255, 51, 119)"),
              (0.66, "rgb(219, 51, 255)"),  (1.00, "rgb(219, 51, 255)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=3, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to k-means clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.08)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1000, width=800, margin=config_margin, legend_tracegroupgap=30, title=config_title)
fig2.show()

print("\n")


# In[ ]:


#@title
#n+1 number of clusters
n=2
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = AgglomerativeClustering(n_clusters=n+1, linkage='ward', affinity='euclidean').fit(waves_df.fillna(0).T)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.33, "rgb(255, 196, 51)"),
              (0.33, "rgb(255, 51, 119)"), (0.66, "rgb(255, 51, 119)"),
              (0.66, "rgb(219, 51, 255)"),  (1.00, "rgb(219, 51, 255)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=3, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to hierarchical clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.08)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1000, width=800, margin=config_margin, legend_tracegroupgap=30, title=config_title)
fig2.show()

print("\n")


# Kmeans, hclust for pca+triangles

# In[ ]:


#@title
#Silhouette Score for K means
model = KMeans(random_state=100)

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(pca_waves_df)
visualizer.show()

#Silhouette Score for Hierarchical clustering
model = AgglomerativeClustering(linkage='ward', affinity='euclidean')

visualizer = KElbowVisualizer(model, k=(3,8),metric='silhouette', timings= True, size=(500, 300))
visualizer.fit(pca_waves_df)        
visualizer.show()


# In[ ]:


#@title
#n+1 number of clusters
n=4
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = KMeans(n_clusters=n+1, random_state=100).fit(pca_waves_df)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)

clusters_list_km_pca_triangle = clusters_list.copy()


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.2, "rgb(255, 196, 51)"),
              (0.2, "rgb(255, 51, 119)"), (0.4, "rgb(255, 51, 119)"),
              (0.4, "rgb(219, 51, 255)"),  (0.6, "rgb(219, 51, 255)"),
              (0.6, "rgb(51, 189, 255)"),  (0.8, "rgb(51, 189, 255)"),
              (0.8, "rgb(51, 255, 53)"),  (1.0, "rgb(51, 255, 53)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=5, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to k-means clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.06)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1400, width=800, margin=config_margin, legend_tracegroupgap=55, title=config_title)
fig2.show()

print("\n")


# In[ ]:


#@title
#n+1 number of clusters
n=4
print("\033[1m" + f"{n+1} clusters: " + "\033[0m")
kmeans = AgglomerativeClustering(n_clusters=n+1, linkage='ward', affinity='euclidean').fit(pca_waves_df)
kmeans.labels_

clusters_list = []
#loop responsible for creating list of lists of countries splited by clustering and printing them
for iterator in range(n+1):
  print(f"Cluster {iterator+1}: ")
  cluster_list = []
  for iterator2, country in enumerate(data.index):
    if kmeans.labels_[iterator2] == iterator:
      cluster_list.append(country)
  print(cluster_list, end="\n")
  clusters_list.append(cluster_list)


#setting choropleth parameters
config = dict(
  type = 'choropleth',
  locations = data.index.values,
  locationmode='country names',
  z=np.append(kmeans.labels_+1, n+1).astype(int),
  colorscale=[(0.00, "rgb(255, 196, 51)"),   (0.2, "rgb(255, 196, 51)"),
              (0.2, "rgb(255, 51, 119)"), (0.4, "rgb(255, 51, 119)"),
              (0.4, "rgb(219, 51, 255)"),  (0.6, "rgb(219, 51, 255)"),
              (0.6, "rgb(51, 189, 255)"),  (0.8, "rgb(51, 189, 255)"),
              (0.8, "rgb(51, 255, 53)"),  (1.0, "rgb(51, 255, 53)")],
  marker_line_color='black',
  marker_line_width=0.5,
  colorbar=dict(nticks=5, tickprefix='Cluster ')
  )

config_coloraxis=dict(
  tickvals=[el for el in range(1,n+2)], 
  title='Clusters', 
  ticks='outside'
  )

config_margin=dict(
  r=25, 
  t=25, 
  l=25,
  b=25
  )

#plotting first part of the figure
fig = go.Figure(data=[config])
fig.update_geos(scope='world', lataxis_showgrid=True, lonaxis_showgrid=True, 
                projection_type='mercator', lataxis_range=[40,75], lonaxis_range=[-30, 70], 
                lataxis_dtick=10, lonaxis_dtick=10, resolution=50)
fig.update_layout(height=350, width=500, margin=config_margin, 
                  coloraxis_colorbar=config_coloraxis, title='Division of map according to hierarchical clustering', title_y=0.97)
fig.show()

#setting legend title parameters(broken method)
config_title=dict(
  text='Countries', 
  x=0.875, 
  y=0.99,
  font_size=13, 
  font_family='Arial'
)

#plotting second part of the figure
fig2 = subplots.make_subplots(rows=int(n+1), cols=1, subplot_titles=[f"Cluster {iterator + 1}" for iterator in range(len(clusters_list))], vertical_spacing=0.06)
for iterator, cluster in enumerate(clusters_list):
  #calculating mean for each cluster
  frame = pd.DataFrame(np.mean(europe_df_rolled[cluster], axis=1), 
                      columns=['Mean'], index=europe_df_rolled.index)
  for country in cluster:
    #breaking long country names
    if country == 'Bosnia and Herzegovina':
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name='Bosnia and<br> Herzegovina', legendgroup=iterator), row=int(iterator)+1, col=1)
    else:
      fig2.add_trace(go.Scatter(x=europe_df_rolled.index, y=europe_df_rolled[country], name=country, legendgroup=iterator), row=int(iterator)+1, col=1)
    fig2.update_xaxes(title_text='Date', row=int(iterator)+1, col=1)
    fig2.update_yaxes(title_text='New cases per milion', row=int(iterator)+1, col=1, range=[0,2900])
    
  #adding trace of mean
  fig2.add_trace(go.Scatter(x=frame.index, y=frame['Mean'], name='Mean', 
                            line=dict(color='black', width=3, dash='dash'), legendgroup=iterator, showlegend=False), 
                 row=int(iterator)+1, col=1)

    
fig2.update_layout(height=1400, width=800, margin=config_margin, legend_tracegroupgap=55, title=config_title)
fig2.show()

print("\n")


# *Comparison* of clusters

# In[ ]:


##we can use it
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Use the venn2 function
title = ['First cluster', 'Second cluster', 'Third cluster']
for iterator in range(3):
  venn3(subsets = (set(clusters_list_km_pca[iterator]), 
                  set(clusters_list_km_triangle[iterator]),
                  set(clusters_list_km_pca_triangle[iterator])), set_labels = ('KMeans on pca', 'KMeans on triangles', 'KMeans on pca+triangles'))
  plt.title(title[iterator], fontsize=16)
  plt.show()

