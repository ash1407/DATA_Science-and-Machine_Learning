#Skeness and kurtosis script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis

#sparkDF = spark.read.csv("dbfs:/FileStore/tables/IPSS_OBJ_GRADES_pbidatasetdb_test.csv", header="true",inferSchema="true")
data = pd.read_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\IPSS_OBJ_GRADES_pbidatasetdb_test.csv")


#%%

data1_grades = data[["LocationName", "Score"]].dropna()
#data1_grades["Score"] = data1_grades["Score"].astype(float)
data1_grades["Score"] = data1_grades["Score"].astype(int)

loc_data = data1_grades.drop_duplicates(subset="LocationName")
loc_list = loc_data["LocationName"].unique().tolist()

skew_kurt_df = pd.DataFrame(columns=["Location", "Skewness", "Kurtosis"])

for loc in loc_list:
    location = data1_grades[data1_grades["LocationName"] == loc]
    skewness = skew(location["Score"])
    kurtosis_value = kurtosis(location["Score"])
    skew_kurt_df = skew_kurt_df.append(
        {"Location": loc, "Skewness": skewness, "Kurtosis": kurtosis_value}, ignore_index=True

    )
    
#skew_kurt_df.to_csv("C:\\Users\\qj771f\\Desktop\\PTA\\Task\\data dictionary and concordance\\Skewness and kurtosis\\campus_skew_kurt.csv")    
print(skew_kurt_df)

#%%

#skewness and kurtosis value for each instructor which is taken from above

# Create a scatter plot of skewness and kurtosis values
plt.scatter(skew_kurt_df["Location"], skew_kurt_df["Skewness"], color='red',label="Skewness")
plt.scatter(skew_kurt_df["Location"], skew_kurt_df["Kurtosis"], color='blue',label="Kurtosis")

# Add axis labels and a title
plt.xlabel("Location")
plt.ylabel("Skewness and Kurtosis")
plt.title("Skewness and Kurtosis plot for Location")

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%

#show metrics like kurtosis, skewness on distribution plot for Miami Campus using seaborn in python

data1_grades = data[["LocationName", "Score"]].dropna()
data1_grades["Score"] = data1_grades["Score"].astype(int)

data1_grades=data1_grades[data1_grades["LocationName"]=='Miami Campus']['Score']

# Calculate kurtosis and skewness for the "Score" column
kurtosis_value = kurtosis(data1_grades)
skewness = skew(data1_grades)

# Create a distribution plot for skewness
sns.distplot(data1_grades, kde=False, fit=stats.norm)


# Calculate mean, median, and mode
mean = data1_grades.mean()
median = data1_grades.median()
mode = data1_grades.mode().values[0]

# Add vertical lines for mean, median, and mode
plt.axvline(mean, color='red', linestyle='dashed', label='Mean')
plt.axvline(median, color='green', linestyle='dashed', label='Median')
plt.axvline(mode, color='yellow', linestyle='dashed', label='Mode')


# Add kurtosis and skewness values to the plot
plt.text(0.3, 0.8, "Kurtosis: {:.2f}".format(kurtosis_value), transform=plt.gca().transAxes)
plt.text(0.3, 0.7, "Skewness: {:.2f}".format(skewness), transform=plt.gca().transAxes)

plt.text(0.3, 0.6, "Mean: {:.2f}".format(mean), transform=plt.gca().transAxes)
plt.text(0.3, 0.5, "Median: {:.2f}".format(median), transform=plt.gca().transAxes)
plt.text(0.3, 0.4, "Mode: {:.2f}".format(mode), transform=plt.gca().transAxes)

# Add a title
plt.title("Distribution of Scores for Miami Campus")

# Show the plot
plt.show()


print('mean:', mean)
print('Median:', median)
print('Mode:', mode)
print('Skewness:', skewness)
print('Kurtosis:', kurtosis_value)
#%%

#show metrics like kurtosis, skewness on distribution plot for Singapore Campus using seaborn in python

data1_grades = data[["LocationName", "Score"]].dropna()
data1_grades["Score"] = data1_grades["Score"].astype(int)

data1_grades=data1_grades[data1_grades["LocationName"]=='Singapore Campus']['Score']

# Calculate kurtosis and skewness for the "Score" column
kurtosis_value = kurtosis(data1_grades)
skewness = skew(data1_grades)

# Create a distribution plot for skewness
sns.distplot(data1_grades, kde=False, fit=stats.norm)


# Calculate mean, median, and mode
mean = data1_grades.mean()
median = data1_grades.median()
mode = data1_grades.mode().values[0]

# Add vertical lines for mean, median, and mode
plt.axvline(mean, color='red', linestyle='dashed', label='Mean')
plt.axvline(median, color='green', linestyle='dashed', label='Median')
plt.axvline(mode, color='yellow', linestyle='dashed', label='Mode')



# Add kurtosis and skewness values to the plot
plt.text(0.6, 0.8, "Kurtosis: {:.2f}".format(kurtosis_value), transform=plt.gca().transAxes)
plt.text(0.6, 0.7, "Skewness: {:.2f}".format(skewness), transform=plt.gca().transAxes)

plt.text(0.6, 0.6, "Mean: {:.2f}".format(mean), transform=plt.gca().transAxes)
plt.text(0.6, 0.5, "Median: {:.2f}".format(median), transform=plt.gca().transAxes)
plt.text(0.6, 0.4, "Mode: {:.2f}".format(mode), transform=plt.gca().transAxes)

# Add a title
plt.title("Distribution of Scores for Singapore Campus")

# Show the plot
plt.show()

print('mean:', mean)
print('Median:', median)
print('Mode:', mode)
print('Skewness:', skewness)
print('Kurtosis:', kurtosis_value)

#%%

#show metrics like kurtosis, skewness on distribution plot for Gatwick Campus using seaborn in python

data1_grades = data[["LocationName", "Score"]].dropna()
data1_grades["Score"] = data1_grades["Score"].astype(int)

data1_grades=data1_grades[data1_grades["LocationName"]=='Gatwick Campus']['Score']

# Calculate kurtosis and skewness for the "Score" column
kurtosis_value = kurtosis(data1_grades)
skewness = skew(data1_grades)

# Create a distribution plot for skewness
sns.distplot(data1_grades, kde=False, fit=stats.norm)


# Calculate mean, median, and mode
mean = data1_grades.mean()
median = data1_grades.median()
mode = data1_grades.mode().values[0]

# Add vertical lines for mean, median, and mode
plt.axvline(mean, color='red', linestyle='dashed', label='Mean')
plt.axvline(median, color='green', linestyle='dashed', label='Median')
plt.axvline(mode, color='yellow', linestyle='dashed', label='Mode')


# Add kurtosis and skewness values to the plot
plt.text(0.3, 0.8, "Kurtosis: {:.2f}".format(kurtosis_value), transform=plt.gca().transAxes)
plt.text(0.3, 0.7, "Skewness: {:.2f}".format(skewness), transform=plt.gca().transAxes)

plt.text(0.3, 0.6, "Mean: {:.2f}".format(mean), transform=plt.gca().transAxes)
plt.text(0.3, 0.5, "Median: {:.2f}".format(median), transform=plt.gca().transAxes)
plt.text(0.3, 0.4, "Mode: {:.2f}".format(mode), transform=plt.gca().transAxes)

# Add a title
plt.title("Distribution of Scores for Gatwick Campus")

# Show the plot
plt.show()

print('mean:', mean)
print('Median:', median)
print('Mode:', mode)
print('Skewness:', skewness)
print('Kurtosis:', kurtosis_value)

#%%
#---------------------------------------------------------------------------------
#kurtosis Plot
