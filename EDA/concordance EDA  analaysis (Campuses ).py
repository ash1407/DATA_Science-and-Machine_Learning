import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#sparkDF = spark.read.csv("dbfs:/FileStore/tables/IPSS_OBJ_GRADES_pbidatasetdb_test.csv", header="true",inferSchema="true")
data = pd.read_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\IPSS_OBJ_GRADES_pbidatasetdb_test.csv")


#%%
#getting unique data
df=data
# Create an empty list to store the unique values
unique_values = []

# Iterate through each column in the DataFrame
for column in df.columns:
    # Get all unique values for the current column
    column_unique_values = df[column].unique()
    # Add the unique values to the list
    unique_values.append(column_unique_values)

# Convert the list of unique values to a DataFrame
unique_values_df = pd.DataFrame(unique_values)

unique_values_df = unique_values_df.transpose()
# Write the DataFrame to a CSV file
#unique_values_df.to_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\unique_values.csv")



#%%
data.shape
data.columns
data.describe()
data.info()
#%%
# Getting Duplicate data 


df=data
duplicate_count = df.duplicated(keep='first').sum() #5509

#Print the number of duplicate rows
print("Number of duplicate rows:", duplicate_count)

#Show the duplicate rows
duplicate_rows = df[df.duplicated(keep='first')]

duplicate_rows.to_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\duplicate_data.csv")

#dropping duplicates
data.drop_duplicates(inplace = True) #50000 to 44500 ,5500 data is droped

#%%
#droping coulumns
#Model
data.drop(['Model','EVENT_START_DT', 'Training_Program_Start_Date',
       'Training_Program_End_Date','StudentCcid',
       'StudentUserID','EventID'],1,inplace = True)
#%%
#Arranging the columns
data=data[['InstructorName', 'InstructorBEMSID','LocationName','Score','StudentName', 'Position','Customer', 
           'CourseType','InstructorAirlineName','StudentAirlineName','Variant', 
       'CompletionStatus', 'ObjectiveNumber', 'ObjectiveOrder',
       'ObjectiveSkill', 'ObjectiveName', 'Repeats', 'Reason',
       'LastAttempt', 'Comments', 'Regulatory', 'DeviceName',
       'EventName','TrainingProgramID']]
#%%
dataty=data.dtypes


#%%
#numerical and object features selection
obj_feature=list( data.select_dtypes(include=[object]).columns) # 23
num_featur_int=list( data.select_dtypes(include='int64').columns) #0
num_featur_float=list( data.select_dtypes(include=['float64']).columns) #5

num_features=list( data.select_dtypes(include=['float64','int64']).columns)

#%%
#data cleaning is started

#checking null values within a column 
datanull=data.isnull().sum()
#rows where data is null in any of clms
#----datanull=data.loc[:,data.isnull().any()]
#%%
# replacing nan values unknown
for clm in obj_feature:
 data[clm].fillna('None', inplace = True)
 
for clm in num_featur_int: 
 data[clm].fillna(0, inplace = True)

for clm in num_featur_float: 
 data[clm].fillna(0, inplace = True) 

#cleaning is complete now.
 
#%% 
#duplicate data
#print(data.duplicated())
#%%

#                  Univariate anlaysis 
 
#getting unique value counts for location,airline,instructor,student,score
campuse_unique=data['LocationName'].value_counts() 
airline_student_unique=data['StudentAirlineName'].value_counts() 
airline_instructor_unique=data['InstructorAirlineName'].value_counts() 
objective_unique=data['ObjectiveName'].value_counts() 
instructor_unique=data['InstructorName'].value_counts() 
StudentName_unique=data['StudentName'].value_counts() 
score_unique=data['Score'].value_counts() 
df=pd.concat([campuse_unique,airline_student_unique,airline_instructor_unique ,objective_unique ,instructor_unique,score_unique],axis=1)
#df.to_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\unique_value.csv")
#univaraite analysis
plt.figure(figsize=(60, 100))
data1=data.groupby(['LocationName','TrainingProgramID'])['Score'].mean().plot(title='Average score on locations')
#%%
#Given coursetype in each location analysis is done to get a score mean ,stddev , total unique instructor and
#   students  taken part in course module ,displaying  grades given by instructor with in campus

temp = data.groupby(['CourseType','LocationName']).agg({'Score':['mean','std'],
                                                      'InstructorBEMSID':'nunique',
                                                      'StudentUserID':'nunique',
                                                      'StudentName':'count'}).reset_index()
#rename columns
temp.columns = ['CourseType','LocationName','Score Mean','Score Std Dev','Total Instructors','Total Students','StudentName']


temp=temp[['CourseType','LocationName','Score Mean','Score Std Dev','Total Instructors','Total Students']]

temp.to_csv('C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\temp.csv', index=False)
#%%
#plotting the above data
tempc1=temp[temp['LocationName']=='Miami Campus']
sns.barplot(x=tempc1['CourseType'],y=tempc1['Score Mean'])
plt.xticks(rotation=90)
plt.title('Coursetypes with in Miamai Campus')
plt.show()

tempc1=temp[temp['LocationName']=='Singapore Campus']
sns.barplot(x=tempc1['CourseType'],y=tempc1['Score Mean'])
plt.xticks(rotation=90)
plt.title('Coursetypes with in Singapore Campus')
plt.show()

#%%
#course type rated globally and with in Miami_campus 

miami_data=data[data['LocationName']=='Miami Campus']
not_miamidata=data[data['LocationName']!='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped_miami = miami_data.groupby(['CourseType'])['Score'].mean().reset_index()

grouped_all = not_miamidata.groupby(['CourseType'])['Score'].mean().reset_index()

merged_coursetype=pd.merge(grouped_miami, grouped_all, on='CourseType',how='left')

merged_coursetype.columns=['CourseType','Score_Miami','Score_other_regions']

merged_coursetype['Score_other_regions'].fillna(0, inplace=True)

melted_df = pd.melt(merged_coursetype, id_vars='CourseType', value_vars=['Score_Miami','Score_other_regions'])
sns.barplot(data=melted_df, x='CourseType', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()

#%%and with 
#for course type in miamai campus , how other course type rated globally and within miamai

# Group by CourseType and locationname to get the average score for each location
grouped_by_CourseType = data.groupby(["LocationName","CourseType"]).mean()["Score"]

# Reset the index to make the columns available to use
grouped_by_CourseType = grouped_by_CourseType.reset_index()


# Renaming the columns
grouped_by_CourseType.columns = ['LocationName','CourseType', "Score"]

#Calculating average score of all other CourseType present in other locations
unique_location=list(grouped_by_CourseType.LocationName.unique())
locname=[]
other_scores=[]
sameloc_score=[]
for loc in unique_location:
    
   var1=grouped_by_CourseType[ grouped_by_CourseType['LocationName']!=loc]["Score"].mean()
   var2=grouped_by_CourseType[ grouped_by_CourseType['LocationName']==loc]["Score"].mean()
   other_scores.append(var1)
   sameloc_score.append(var2)
   locname.append(loc)
   
var2 = pd.DataFrame(
    {'LocationName': locname,
     'avg_score_same_loc': sameloc_score,
     'avg_score_other_loc': other_scores
    }) 
merged_df=pd.merge(grouped_by_CourseType,var2,on='LocationName',how='left')    

#ploteed for only top 10 rows there are lot of rows which will conjust the plot
melted_df = pd.melt(merged_df[merged_df['LocationName']=='Miami Campus'], id_vars='CourseType', value_vars=['Score','avg_score_same_loc','avg_score_other_loc'])
sns.barplot(data=melted_df, x='CourseType', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()
#%%
##Analysis on each location how the score is distributed,total instructors,total students,total course types, 
#   total objectives covered in different locations

temp1 = data.groupby(['LocationName']).agg({'Score':['mean','std'],
                                                      'InstructorBEMSID':'nunique',
                                                      'StudentUserID':'nunique',
                                                      'TrainingProgramID':'nunique',
                                                      'CourseType':'nunique',
                                                      'ObjectiveName':'nunique'}).reset_index()
#rename columns
temp1.columns = ['LocationName','Score Mean','Score Std Dev','Total Instructors','Total Students','Total Trainings','Total course types','Total objectives']

temp1.to_csv('C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\temp1.csv', index=False)


#%%
## Normal Distribution plot of score with in a campus

plt.figure(figsize=(15,10))
df1=data[data['LocationName']=='Miami Campus']
ax=sns.distplot(df1['Score'],color='purple')
ax.set(xlabel='Score', ylabel='percentage of Distribution',title='Distribution plot for Miami Campus ')
plt.show() 

plt.figure(figsize=(15,10))
df1=data[data['LocationName']=='Singapore Campus']
ax=sns.distplot(df1['Score'],color='purple')
ax.set(xlabel='Score', ylabel='percentage of Distribution',title='Distribution plot for Singapore Campus ')
plt.show() 

plt.figure(figsize=(15,10))
df1=data[data['LocationName']=='Gatwick Campus']
ax=sns.distplot(df1['Score'],color='purple',hist=True, kde=True)
ax.set(xlabel='Score', ylabel='percentage of Distribution',title='Distribution plot for Gatwick Campus')
plt.show() 

#%%

#%%

#      Bivariate analaysis

#converted all string to mumericals
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df=data
for features in df:
    df[features] = le.fit_transform(df[features].astype(str))

#converting back to orignal form     
#le.inverse_transform(df[['LocationName']]) 
#%%    

#box plot between campuses and score 

#selecting y axix feature
df_n=['Score']
# Box_plot
print("********************************* Box Plot ***********************************************")
plt.figure(figsize=(20,5))
j=1
for features in df_n:  
    plt.subplot(1,3,j); j+=1 
    sns.boxplot(x= 'LocationName',y= features,data=data)
plt.grid()    
plt.show()

#students passed and failed in traing 

df_n=['CompletionStatus']
# Box_plot
print("********************************* Box Plot ***********************************************")
plt.figure(figsize=(20,5))
j=1
for features in df_n:  
    plt.subplot(1,3,j); j+=1 
    sns.boxplot(x= 'LocationName',y= features,data=df)
plt.grid()    
plt.show()

# 2:pass ,1 fail
print("*********************************** Violin Plot ******************************************")
# violin_plot
plt.figure(figsize=(20,5))
k=1
for features in df_n:  
    plt.subplot(1,3,k); k+=1 
    sns.violinplot(x= 'LocationName',y= features,data=data)
plt.grid()
plt.show()

plt.figure(figsize=(20,5))
j=1
for features in df_n:  
    plt.subplot(1,3,j); j+=1 
    sns.violinplot(x= 'LocationName',y= features,data=df)
plt.grid()    
plt.show()

#%%
#bar plot between location name and no of trainings happened in which instructor rated students on score 1,2,3,4,5

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped = data.groupby(['LocationName', 'Score']).size().reset_index(name='count')

# Create the bar plot
sns.barplot(x='LocationName', y='count', hue='Score', data=grouped)

# Add labels and titles to the plot
plt.xlabel('Location')
plt.ylabel('Count of Score')
plt.title('Score Count by Location')

# Set y-axis ticks
plt.yticks(range(0, max(grouped['count'])+1,1000))

#fig size
plt.figure(figsize=(10,50))

# Show the plot
plt.show()

#%%
#bar plot between location name and score given by instructors with in region

grouped_df = data.groupby('LocationName').mean().reset_index()
# Creating the bar plot
plt.bar(grouped_df['LocationName'], grouped_df['Score'])
plt.xlabel('Location')
plt.ylabel('Average Score')
plt.title('Average Score by Location')
plt.show()
#%%
#bar plot between locations and score given by instructors with in region and globallly

# Group by  locationname to get the average score for each location
grouped_by_location = data.groupby(["LocationName"]).mean()["Score"]

# Reset the index to make the columns available to use
grouped_by_location = grouped_by_location.reset_index()

# Renaming the columns
grouped_by_location.columns = ['LocationName', "avg_score_location"]


#Calculating average score of all other InstructorBEMSID present in other locations
#merged_df["avg_score_other_location"] = A[~A.LocationName.isin(merged_df.LocationName)].groupby(["LocationName"]).mean()["Score"]
#merged_df["avg_score_other_location"] = data[(merged_df2['LocationName'] != merged_df['LocationName'])].groupby(["LocationName"]).mean()["Score"]
unique_location=list(grouped_by_location.LocationName.unique())
locname=[]
other_scores=[]

for loc in unique_location:
    
   var1=grouped_by_location[ grouped_by_location['LocationName']!=loc]["avg_score_location"].mean()
   other_scores.append(var1)
   locname.append(loc)
   
var2 = pd.DataFrame(
    {'LocationName': locname,
     'avg_score_other_loc': other_scores
    }) 
grouped_by_location=pd.merge(grouped_by_location,var2,on='LocationName',how='left')    

#ploteed for only top 10 rows there are lot of rows which will conjust the plot
melted_df = pd.melt(grouped_by_location, id_vars='LocationName', value_vars=['avg_score_location','avg_score_other_loc'])
sns.barplot(data=melted_df, x='LocationName', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()
#%% 
#Bar plot b/w score count given by instructors and with in campus
miamai_data=data[data['LocationName']=='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped = miamai_data.groupby(['InstructorBEMSID', 'Score']).size().reset_index(name='count')

# Create the bar plot
sns.barplot(x='InstructorBEMSID', y='count', hue='Score', data=grouped)

# Add labels and titles to the plot
plt.xlabel('Instructor with in Miamai Campus')
plt.ylabel('Count of Score')
plt.title('Score Count by Location')

# Set y-axis ticks
plt.yticks(range(0, max(grouped['count'])+1,500))
plt.xticks(rotation=90)
 
#fig size
plt.figure(figsize=(10,100))

# Show the plot
plt.show()

#%%
#Bar plot b/w average score given by instructors and with in campus
miamai_data=data[data['LocationName']=='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped = miamai_data.groupby(['InstructorBEMSID'])['Score'].mean().reset_index()

# Create the bar plot
sns.barplot(x='InstructorBEMSID', y='Score', data=grouped)

# Add labels and titles to the plot
plt.xlabel('Instructor with in Miamai Campus')
plt.ylabel('Average Score')
plt.title('Average score  within miami campus')

# Set y-axis ticks
plt.xticks(rotation=90) 

#fig size
plt.figure(figsize=(10,100))

# Show the plot
plt.show()
#%%
#Bar plot b/w score given by instructor and with in campus and same instructor globally
miami_data=data[data['LocationName']=='Miami Campus']
not_miamidata=data[data['LocationName']!='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped_miami = miami_data.groupby(['InstructorBEMSID'])['Score'].mean().reset_index()

grouped_all = not_miamidata.groupby(['InstructorBEMSID'])['Score'].mean().reset_index()

merged_coursetype=pd.merge(grouped_miami, grouped_all, on='InstructorBEMSID',how='left')

merged_coursetype.columns=['InstructorBEMSID','Score_Miami','Score_other_regions']

merged_coursetype['Score_other_regions'].fillna(0, inplace=True)

melted_df = pd.melt(merged_coursetype, id_vars='InstructorBEMSID', value_vars=['Score_Miami','Score_other_regions'])
sns.barplot(data=melted_df, x='InstructorBEMSID', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()

#%%

#Bar plot b/w score given by instructor and with in campus ,in same campuses and globally

# Group by CourseType and locationname to get the average score for each location
grouped_by_InstructorBEMSID = data.groupby(["LocationName","InstructorBEMSID"]).mean()["Score"]

# Reset the index to make the columns available to use
grouped_by_InstructorBEMSID = grouped_by_InstructorBEMSID.reset_index()


# Renaming the columns
grouped_by_InstructorBEMSID.columns = ['LocationName','InstructorBEMSID', "Score"]

#Calculating average score of all other CourseType present in other locations
unique_location=list(grouped_by_InstructorBEMSID.LocationName.unique())
locname=[]
other_scores=[]
sameloc_score=[]
for loc in unique_location:
    
   var1=grouped_by_InstructorBEMSID[ grouped_by_InstructorBEMSID['LocationName']!=loc]["Score"].mean()
   var2=grouped_by_InstructorBEMSID[ grouped_by_InstructorBEMSID['LocationName']==loc]["Score"].mean()
   other_scores.append(var1)
   sameloc_score.append(var2)
   locname.append(loc)
   
var2 = pd.DataFrame(
    {'LocationName': locname,
     'avg_score_same_loc': sameloc_score,
     'avg_score_other_loc': other_scores
    }) 
merged_df=pd.merge(grouped_by_InstructorBEMSID,var2,on='LocationName',how='left')    

#ploteed for only top 10 rows there are lot of rows which will conjust the plot
melted_df = pd.melt(merged_df[merged_df['LocationName']=='Miami Campus'].head(10), id_vars='InstructorBEMSID', value_vars=['Score','avg_score_same_loc','avg_score_other_loc'])
sns.barplot(data=melted_df, x='InstructorBEMSID', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()
#%%

# Bar plot b/w competencys with in region

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped = miamai_data.groupby(['ObjectiveName', 'Score']).mean().reset_index()

# Create the bar plot
sns.barplot(x='ObjectiveName', y='Score', hue='Score', data=grouped)

# Add labels and titles to the plot
plt.xlabel('objectives with in Miamai Campus')
plt.ylabel('Count of Score')
plt.title('Score Count by objectives within Miamai Campus')

plt.xticks(rotation=90) 

#fig size
plt.figure(figsize=(10,10))

# Show the plot
plt.show()

#%%
#Bar plot b/w competencys with in region and globally
miami_data=data[data['LocationName']=='Miami Campus']
not_miamidata=data[data['LocationName']!='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped_miami = miami_data.groupby(['ObjectiveSkill'])['Score'].mean().reset_index()

grouped_all = not_miamidata.groupby(['ObjectiveSkill'])['Score'].mean().reset_index()

merged_coursetype=pd.merge(grouped_miami, grouped_all, on='ObjectiveSkill',how='left')

merged_coursetype.columns=['ObjectiveSkill','Score_Miami','Score_other_regions']

merged_coursetype['Score_other_regions'].fillna(0, inplace=True)

melted_df = pd.melt(merged_coursetype, id_vars='ObjectiveSkill', value_vars=['Score_Miami','Score_other_regions'])
sns.barplot(data=melted_df, x='ObjectiveSkill', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()

#%%

#for objective type in miamai campus , how other objectives rated globally and within miamai

# Group by CourseType and locationname to get the average score for each location
grouped_by_ObjectiveSkill = data.groupby(["LocationName","ObjectiveSkill"]).mean()["Score"]

# Reset the index to make the columns available to use
grouped_by_ObjectiveSkill = grouped_by_ObjectiveSkill.reset_index()


# Renaming the columns
grouped_by_ObjectiveSkill.columns = ['LocationName','ObjectiveSkill', "Score"]

#Calculating average score of all other CourseType present in other locations
unique_location=list(grouped_by_ObjectiveSkill.LocationName.unique())
locname=[]
other_scores=[]
sameloc_score=[]
for loc in unique_location:
    
   var1=grouped_by_ObjectiveSkill[ grouped_by_ObjectiveSkill['LocationName']!=loc]["Score"].mean()
   var2=grouped_by_ObjectiveSkill[ grouped_by_ObjectiveSkill['LocationName']==loc]["Score"].mean()
   other_scores.append(var1)
   sameloc_score.append(var2)
   locname.append(loc)
   
var2 = pd.DataFrame(
    {'LocationName': locname,
     'avg_score_same_loc': sameloc_score,
     'avg_score_other_loc': other_scores
    }) 
merged_df=pd.merge(grouped_by_ObjectiveSkill,var2,on='LocationName',how='left')    

#ploteed for only top 10 rows there are lot of rows which will conjust the plot
melted_df = pd.melt(merged_df[merged_df['LocationName']=='Miami Campus'], id_vars='ObjectiveSkill', value_vars=['Score','avg_score_same_loc','avg_score_other_loc'])
sns.barplot(data=melted_df, x='ObjectiveSkill', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()
#%%
# Bar plot b/w instructorairline ratings with in region

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped = miamai_data.groupby('InstructorAirlineName')['Score'].mean().reset_index()

# Create the bar plot
sns.barplot(x='InstructorAirlineName', y='Score', data=grouped)

# Add labels and titles to the plot
plt.xlabel('objectives with in Miamai Campus')
plt.ylabel('Average Count of Score')
plt.title('Score Count by objectiveskills within Miamai Campus')

plt.xticks(rotation=90) 
#fig size
plt.figure(figsize=(10,10))

# Show the plot
plt.show()

#%%
# Bar plot b/w instructorairline ratings with in region and globally

miami_data=data[data['LocationName']=='Miami Campus']
not_miamidata=data[data['LocationName']!='Miami Campus']

#Use the groupby function to group the data by 'location' and count the number of occurrences of each 'score'
grouped_miami = miami_data.groupby(['InstructorAirlineName'])['Score'].mean().reset_index()

grouped_all = not_miamidata.groupby(['InstructorAirlineName'])['Score'].mean().reset_index()

merged_coursetype=pd.merge(grouped_miami, grouped_all, on='InstructorAirlineName',how='left')

merged_coursetype.columns=['InstructorAirlineName','Score_Miami','Score_other_regions']

merged_coursetype['Score_other_regions'].fillna(0, inplace=True)

melted_df = pd.melt(merged_coursetype, id_vars='InstructorAirlineName', value_vars=['Score_Miami','Score_other_regions'])
sns.barplot(data=melted_df, x='InstructorAirlineName', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()

#%%

#for airlines in miamai campus , how other airlines rated globally and within miami

# Group by CourseType and locationname to get the average score for each location
grouped_by_InstructorAirlineName = data.groupby(["LocationName","InstructorAirlineName"]).mean()["Score"]

# Reset the index to make the columns available to use
grouped_by_InstructorAirlineName = grouped_by_InstructorAirlineName.reset_index()


# Renaming the columns
grouped_by_InstructorAirlineName.columns = ['LocationName','InstructorAirlineName', "Score"]

#Calculating average score of all other CourseType present in other locations
unique_location=list(grouped_by_ObjectiveSkill.LocationName.unique())
locname=[]
other_scores=[]
sameloc_score=[]
for loc in unique_location:
    
   var1=grouped_by_InstructorAirlineName[ grouped_by_InstructorAirlineName['LocationName']!=loc]["Score"].mean()
   var2=grouped_by_InstructorAirlineName[ grouped_by_InstructorAirlineName['LocationName']==loc]["Score"].mean()
   other_scores.append(var1)
   sameloc_score.append(var2)
   locname.append(loc)
   
var2 = pd.DataFrame(
    {'LocationName': locname,
     'avg_score_same_loc': sameloc_score,
     'avg_score_other_loc': other_scores
    }) 
merged_df=pd.merge(grouped_by_InstructorAirlineName,var2,on='LocationName',how='left')    

#ploteed for only top 10 rows there are lot of rows which will conjust the plot
melted_df = pd.melt(merged_df[merged_df['LocationName']=='Miami Campus'], id_vars='InstructorAirlineName', value_vars=['Score','avg_score_same_loc','avg_score_other_loc'])
sns.barplot(data=melted_df, x='InstructorAirlineName', y='value', hue='variable')

plt.xticks(rotation=90)

plt.show()
#%%    
 #Pair Plot
data_nume= data[num_features]
sns.pairplot(data_nume)

#%%

#%%

#heat map
import seaborn as sns
df_g123_corr=df.corr()    # df_g123 is data frame=[]
plt.figure(figsize=(50,50))
sns.heatmap(df_g123_corr,annot=True,cmap=plt.cm.CMRmap_r)
plt.show()   
    

#%%    
df_n=data[['LocationName','Score']]
plt.figure(figsize=(10,30),facecolor='white')
plotnumber=1
for c in df_n:
    ax=plt.subplot(8,1,plotnumber)
    sns.boxplot(df[c],color='green')
    plotnumber=plotnumber + 1 
plt.show()

#%%


#%%

#
campuse_unique=list(data['LocationName'].unique())
airline_instructor_unique=list(data['InstructorAirlineName'].unique() )

df=data
plt.figure(figsize=(30, 30))
for clm1 in campuse_unique:
    
    for clm2 in airline_instructor_unique:
       
       campus_data = df[df['InstructorAirlineName'] == clm2]
       grouped_data = campus_data.groupby(['LocationName','TrainingProgramID'])
       average_scores = grouped_data['Score'].mean()
       
       # Plot the average scores for each objective
       average_scores.plot.bar(stacked=True, color='blue')
    
       # Add labels and a title to the chart
       plt.xlabel(clm2)
       plt.ylabel('Average Score')
       plt.title('Average Scores for Each Objective')
       plt.savefig('C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\'+clm1+'_'+clm2+'.png')
    
       # Show the chart
       plt.show()

#%%

df=data
plt.figure(figsize=(30, 30))
for clm1 in campuse_unique:
    
    for clm2 in airline_instructor_unique:
       
       campus_data = df
       grouped_data = campus_data.groupby(['LocationName','TrainingProgramID'])
       average_scores = grouped_data['Score'].mean()
       
       # Plot the average scores for each objective
       average_scores.plot.bar(stacked=True, color='blue')
    
       # Add labels and a title to the chart
       plt.xlabel(clm2)
       plt.ylabel('Average Score')
       plt.title('Average Scores for Each Objective')
       plt.savefig('C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Plots\\'+clm1+'_'+clm2+'.png')
    
       # Show the chart
       plt.show()
       

#%%
