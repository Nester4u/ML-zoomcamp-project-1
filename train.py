
import numpy as np
import pandas as pd
import seaborn as sns


# # Data Preparation, Cleaning and EDA


# to read the dataset into a given dataframe
data = pd.read_csv("student_performance.csv")
data


pd.set_option('display.max_columns', None)


data.shape #Displaying the dimension of the data set


# In the dataset provided we have **34 columns and 544 rows**.


data.head() #To print out the first 5 columns



data.tail() #to print the last 5 columns



data.info() #To Display information about the DataFrame



data.describe() #Looking at the EDD



data.isna().sum() #To Check for missing values in the data set


# ## Observation
# 	- The dataset provided 544 entries
# ## Categorical columns
# 	- school
# 	- sex
# 	- address    
#     - famsize     
#     - Pstatus     
#     - Medu        
#     - Fedu       
#     - Mjob        
#     - Fjob        
#     - reason     
#     - guardian
# 	- schoolsup
# 	- famsup
# 	- paid
# 	- activities
# 	- nursery
# 	- higher
# 	- internet
# 	- romantic
# 	- health
# ## Missing values
# 	- famsize
# 	- paid
# 	- G1
# 	- G2	
# ## Abnormalities
# 	- StudentID(drop row)
# 	- age
# 	- goout(has a negative value)(sure the person meant 5)
# 	- absences
# 	- G2

# ## Plotting The Graphs For The Categorical Column


sns.countplot(x = "school", data = data) #to check for the categorical data



sns.countplot(x = "sex", data = data) #to check for the categorical data


sns.countplot(x = "address", data = data) #to check for the categorical data


sns.countplot(x = "famsize", data = data) #to check for the categorical data


sns.countplot(x = "Pstatus", data = data)


sns.countplot(x = "Medu", data = data)


sns.countplot(x = "Fedu", data = data)


sns.countplot(x = "Mjob", data = data)


sns.countplot(x = "Fjob", data = data)


sns.countplot(x = "reason", data = data)


sns.countplot(x = "guardian", data = data)



sns.countplot(x = "schoolsup", data = data)


sns.countplot(x = "famsup", data = data)


sns.countplot(x = "paid", data = data)


sns.countplot(x = "activities", data = data)


sns.countplot(x = "nursery", data = data)



sns.countplot(x = "higher", data = data)


sns.countplot(x = "internet", data = data)


sns.countplot(x = "romantic", data = data)


sns.countplot(x = "health", data = data)


# ## Handling The Missing Values


data["sex"].replace({"Male": "M", "female": "F", "Female": "F"}, inplace=True)


sns.countplot(x="sex", data=data)


data["famsize"].unique()


data["famsize"].isna().sum() #to check for the amount of missing values


data["famsize"].fillna("ET3", inplace= True)


data["famsize"].isna().sum()


sns.countplot(x = "famsize", data = data)



data.info()


data["paid"].unique()



data["paid"].isna().sum() #to check for the amount of missing values


data.paid.describe()


data["paid"].mode()[0]



data["paid"].fillna(data["paid"].mode()[0], inplace= True)



data["paid"].isna().sum()


sns.countplot(x = "paid", data = data)


data.info()



data["G1"].isna().sum()


data["G1"].mean()


data["G1"] = data["G1"].fillna(data["G1"].mean())


data["G1"].isna().sum()


data.describe()


data["G2"].isna().sum()


data["G2"].median()



data["G2"] = data["G2"].fillna(data["G2"].median())


data["G1"].isna().sum()


data.describe()


data.info()


# Here i have sorrted all my missing values


data.isna().sum()


# here i dropped the StudentID column because it wasn't a necessary data in our dataset and so that the machine would have some accuracy in the model 


data.drop("StudentID", axis = 1, inplace=True) #delete a column


data.describe()


# ## Outlier
# 	- age
# 	- goout
# 	- G2
# ## skewness
# 	- absences
# 

# ## Handling The Outliers


sns.scatterplot(x ="age", y ="Grade", data = data )


sns.scatterplot(x ="goout", y ="Grade", data = data )


sns.jointplot(x ="absences", y ="Grade", data = data )



sns.scatterplot(x ="G2", y ="Grade", data = data )



np.percentile(data.age,[99])



np.percentile(data.age,[99])[0]



uv = np.percentile(data.age,[99])[0]



data[(data.age>uv)]



data.age[(data.age>uv)] = uv



sns.scatterplot(x ="age", y ="Grade", data = data )



np.percentile(data.age,[1])



np.percentile(data.age,[1])[0]



lv = np.percentile(data.age,[1])[0]


data[(data.age<lv)]


data.age[(data.age<lv)] = lv


sns.scatterplot(x ="age", y ="Grade", data = data )


data["goout"].describe()



data['goout'] = data['goout'].abs()



data["goout"].describe()



sns.scatterplot(x ="goout", y ="Grade", data = data )



data["G2"].describe()



np.percentile(data.G2,[99])

np.percentile(data.G2,[99])[0]


uppervalue = np.percentile(data.G2,[99])[0]


data[(data.G2>uppervalue)]


data.G2[(data.G2>uppervalue)] = uppervalue


sns.scatterplot(x ="G2", y ="Grade", data = data )


sns.distplot(x =data.absences,  hist=True, kde=True)


data.absences.describe()


np.log(data.absences).describe()


np.sqrt(data.absences).describe()


np.exp(data.absences).describe()


np.log(1+ data.absences).describe()


data.absences = np.log(1+ data.absences)


# here i cleaned all my outliers and plotted a good displot of the column absences


sns.distplot(x =data.absences,  hist=True, kde=True)


#  ## Handling The Categorical Column


encoded_df = pd.get_dummies(data)
encoded_df.head()


df= pd.get_dummies(data["school"])


data.head()


df.head()


data["school"] = df["GP"] 


data.head()



df= pd.get_dummies(data["sex"])


df.head()


data["sex"] = df["F"] 


data.head()


df= pd.get_dummies(data["address"])



df.head()



data["address"] = df["R"] 



data.head()



data["famsize"].replace({"GT3":1, "LE3":-1, "ET3" : 0}, inplace=True)


data["famsize"].describe()


data.head()


df= pd.get_dummies(data["Pstatus"])


df.head()



data["Pstatus"] = df["A"] 



data.head()


data["Medu"].unique()


['Tertiary education', 'Secondary education','Postgraduate education', 'Primary education', 'No education']


from sklearn.preprocessing import OrdinalEncoder

order_mappings = ['No education', 'Primary education','Secondary education', 'Tertiary education', 'Postgraduate education']

encoder = OrdinalEncoder(categories=[order_mappings])

data["Medu_Encoded"] = encoder.fit_transform(data[["Medu"]])


data.head()



data["Fedu"].unique()



['Secondary education', 'Postgraduate education','Tertiary education', 'Primary education', 'No education']



from sklearn.preprocessing import OrdinalEncoder

order_mappings = ['No education', 'Primary education','Secondary education', 'Tertiary education', 'Postgraduate education']
encoder = OrdinalEncoder(categories=[order_mappings])

data["Fedu_Encoded"] = encoder.fit_transform(data[["Fedu"]])



data.head()



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data['Mjob'])
data['Mjob'] = encoded_data



data.head()



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data['Fjob'])
data['Fjob'] = encoded_data


data.head()



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data['reason'])
data['reason'] = encoded_data



data.head()


data.head()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data['guardian'])
data['guardian'] = encoded_data


data.head()


df= pd.get_dummies(data["schoolsup"])


df.head()


data["schoolsup"] = df['no'] 


data["schoolsup"]= data["schoolsup"].astype(float)


data.head()


df= pd.get_dummies(data["famsup"])
df.head()


data["famsup"] = df['no'] 
data["famsup"] = data["famsup"].astype(float)
data.head()


df= pd.get_dummies(data["paid"])
df.head()


data["paid"] = df['no'] 
data["paid"] = data["paid"].astype(float)
data.head()


df= pd.get_dummies(data["activities"])
df.head()


data["activities"] = df['no'] 
data["activities"] = data["activities"].astype(float)
data.head()



df= pd.get_dummies(data["nursery"])
df.head()
data["nursery"] = df['no'] 
data["nursery"] = data["nursery"].astype(float)
data.head()



df= pd.get_dummies(data["higher"])
df.head()
data["higher"] = df['no'] 
data["higher"] = data["higher"].astype(float)
data.head()


df= pd.get_dummies(data["internet"])
df.head()
data["internet"] = df['no'] 
data["internet"] = data["internet"].astype(float)
data.head()



df= pd.get_dummies(data["romantic"])
df.head()
data["romantic"] = df['no'] 
data["romantic"] = data["romantic"].astype(float)
data.head()


data["health"].unique()


['very good', 'very bad', 'good', 'average', 'bad']


from sklearn.preprocessing import OrdinalEncoder

order_mappings = ['very bad', 'bad','average', 'good', 'very good']

encoder = OrdinalEncoder(categories=[order_mappings])

data["health_Encoded"] = encoder.fit_transform(data[["health"]])



data.head()


data.drop(["Medu", "Fedu","health"], axis=1, inplace=True) #delete the column of Medu,fedu,health


data.head()


data = data.astype(float)


data.head()


# from the table above it can be seen that i have properly handle all the categorical columns

# ## Multicollinearity


data.corr()


import matplotlib.pyplot as plt
plt.figure(figsize = (21, 21))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)


# ## Important Features
#     - G1
#     - G2


data['avg_G1_G2'] = (data.G1+data.G2)/2



data.head()



import matplotlib.pyplot as plt
plt.figure(figsize = (21, 21))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)


data.drop(["G1", "G2"], axis=1, inplace=True) #delete the column of G1, G2


import matplotlib.pyplot as plt
plt.figure(figsize = (21, 21))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', center=0, square=True)


# # Model Selection Process

# ### Linear Regression


y = data["Grade"]
x = data.drop("Grade",axis=1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(x,y)


lr.intercept_, lr.coef_


y_pred = lr.predict(x_train)
y_pred_sel = lr.predict(x_test)


# Here i used one of the Machine learning Evaluation Matrics to check for the performance of my module


from sklearn.metrics import r2_score
r2_score(y_train, y_pred)



y_pred_sel = lr.predict(x_test)



r2_score(y_test, y_pred_sel)



df.to_csv("student_performance.csv", index = False)


import ipywidgets as widgets
from IPython.display import display
from sklearn.linear_model import LinearRegression  # Replace with your actual model

model =lr

X_train = x

# Create a dictionary to store the input widgets
input_widgets = {}

# Create text input widgets for each feature in X_train
for feature in X_train.columns:
    input_widgets[feature] = widgets.Text(description=feature + ':')
prediction = 0
# Create a prediction function
def make_prediction(b):
    input_features = {}
    
    # Retrieve the input values from the widgets
    for feature, widget in input_widgets.items():
        input_features[feature] = widget.value

    # Prepare the input data as a dictionary
    input_data = {feature: [value] for feature, value in input_features.items()}

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Make a prediction
    prediction = model.predict(input_df)

    # Display the prediction
    with output:
        print(f'Predicted Grade: {prediction[0]}')

# Create a button for prediction
predict_button = widgets.Button(description='Predict', button_style= "danger")
predict_button.on_click(make_prediction)
output = widgets.Output()

# Display the input widgets and prediction button
input_widgets_list = list(input_widgets.values())
input_widgets_list.append(predict_button)
input_widgets_list.append(output)
display(*input_widgets_list)



data.describe()


# ### Random Forest


from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(max_depth=2, random_state=42)



rf.fit(x_train, y_train)



y_pred = rf.predict(x_train)



y_pred_sel = rf.predict(x_test)


r2_score(y_train, y_pred)


r2_score(y_test, y_pred_sel)


# ### Decision Tree


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# function to perform training with Entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)


# function to make prediction
y_pred_en = clf_entropy.predict(x_test)
y_pred_en


# checking accuracy
print("Accuracy is ", accuracy_score(y_test, y_pred_en))


# ### Evaluation of the Model Selection Process

# - The linear regression, random forest and decision tree processes were carried out on the model and the linear regression was found to have the best performance

# # Parameter Tuning and Training the Final Model 

data.head()


import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(x_train, y_train)



ridge_reg.score(x_train, y_train)



ridge_reg.score(x_test, y_test)


print(f"the model's accuracy is 0.7864261243414166")


# # Saving the Model


import pickle


filename = 'stud_performance.pkl'
pickle.dump(ridge_reg,open(filename,'wb'))

