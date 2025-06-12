#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Loading the spreadsheet to inspect its contents
df = pd.read_csv('//Datafilepath')
# Displaying the first few rows of the dataframe to understand its structure
df.head()


# In[2]:


#Want to find out what evento column contains 
df['evento'].unique()


# # Filtering out only flood related event

# In[3]:


# Filter the data to extract only the rows related to flood events
flood_df = df[df['evento'].isin(['RAINS','FLOOD'])]
#Displaying the raws of the data
flood_df.head()


# In[4]:


flood_df.columns


# In[5]:


flood_df.info


# # Translating the name of the columns form Spanish to English

# In[6]:


#As data sets header are in the another language then we need 
#So we need to translate the data 
translated_columns = {
    'serial': 'serial',
    'level0': 'level_0',
    'level1': 'level_1',
    'level2': 'level_2',
    'name0': 'name_0',
    'name1': 'name_1',
    'name2': 'name_2',
    'evento': 'event',
    'lugar': 'place',
    'fechano': 'year',
    'fechames': 'Month',
    'fechadia': 'Day',
    'muertos': 'Deaths',
    'heridos': 'Injured',
    'desaparece': 'Missing',
    'afectados': 'Affected',
    'vivdest': 'Destroyed Homes',
    'vivafec': 'Affected Homes',
    'otros': 'Others',
    'fuentes': 'Sources',
    'valorloc': 'Local Value',
    'valorus': 'National Value',
    'fechapor': 'Reported Date',
    'fechafec': 'Date of Occurrence',
    'hay_muertos': 'Deaths Reported',
    'hay_heridos': 'Injuries Reported',
    'hay_deasparece': 'Missing Reported',
    'hay_afectados': 'Affected Reported',
    'hay_vivdest': 'Destroyed Homes Reported',
    'hay_vivafec': 'Affected Homes Reported',
    'hay_otros': 'Other Reported',
    'socorro': 'Relief',
    'salud': 'Health',
    'educacion': 'Education',
    'agropecuario': 'Agricultural',
    'industrias': 'Industries',
    'acueducto': 'Water Supply',
    'alcantarillado': 'Sewage',
    'energia': 'Energy',
    'comunicaciones': 'Communications',
    'causa': 'Cause',
    'descausa': 'Cause Description',
    'transporte': 'Transport',
    'magnitud2': 'Magnitude',
    'nhospitales': 'Hospitals',
    'nescuelas': 'Schools',
    'nhectareas': 'Hectares',
    'cabezas': 'Heads of Livestock',
    'kmvias': 'Roads (km)',
    'duracion': 'Duration',
    'damnificados': 'Affected Persons',
    'evacuados': 'Evacuated Persons',
    'Damaged': 'Damaged Property',
    'Evacuated': 'Evacuated Persons',
    'Relocated': 'Relocated Persons',
    'Relocated Count': 'Relocated Count',
    'glide': 'GLIDE Code',
    'UUID': 'UUID'}
# Renaming the columns of the filtered dataframe
flood_df_translated = flood_df.rename(columns=translated_columns)
# Display the filtered data
flood_df_translated.head()


# In[7]:


flood_df_translated.columns


# # Data Preprocessing

# In[8]:


#Cleaning columns which have only numeric values 
numeric_cols = flood_df_translated.select_dtypes(include ='number').columns
#Indentifying the columns which all values are 0 or negative which are not relevnt for this
cols_to_drop =[col for col in numeric_cols if(flood_df_translated[col] <=0).all()]
#Drop the columns which are not needed 
cleaned_df = flood_df_translated.drop(columns = cols_to_drop)
cleaned_df.head(10)


# In[9]:


# Selecting only the non-object columns
non_text_columns = cleaned_df.select_dtypes(exclude='object')
# Create a new DataFrame with only those columns
numeric_only_df = cleaned_df[non_text_columns.columns]
numeric_only_df.head()


# # Cleaning the data set

# In[10]:


# 1. Make a copy to work on
clean_df = numeric_only_df.copy()

# 2. Check and drop duplicate rows
clean_df = clean_df.drop_duplicates()

# 3. Check for missing values
missing_values = clean_df.isnull().sum()

# 4. Drop columns where all values are missing
clean_df = clean_df.dropna(axis=1, how='all')

# 5. Fill or drop remaining missing values (if any)
# For simplicity, fill numeric columns with median (robust to outliers)
clean_df = clean_df.fillna(clean_df.median(numeric_only=True))

# 6. Ensure all columns are numeric
clean_df = clean_df.apply(pd.to_numeric, errors='coerce')

# Re-check for any new NaNs after conversion and fill them again
clean_df = clean_df.fillna(clean_df.median(numeric_only=True))
clean_df.head()


# In[11]:


# Remove unnecessary columns
columns_to_drop = ['serial','level_0', 'Other Reported', 'Relief', 'Relief', 'Health',
       'Education', 'Agricultural', 'Industries', 'Water Supply', 'Sewage',
       'Energy', 'Communications', 'Transport', 'Hospitals', 'Schools',
       'Hectares', 'Roads (km)', 
        'reubicados', 'clave','latitude',
       'longitude'] 
df_cleaned = clean_df.drop(columns=columns_to_drop)
df_cleaned.head(10)


# # Conversion of date and time

# In[12]:


# Convert date-related columns to datetime format
date_columns = ['year']

for col in date_columns:
    if col in df_cleaned:
        # Convert 'date_year' to just the year format
        if col == 'year':
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], format='%Y', errors='coerce').dt.year
        else:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the trend of flood occurrences over the years
plt.figure(figsize=(10, 5))
sns.histplot(df_cleaned['year'].dropna(), bins=20, kde=True)
plt.title("Flood Occurrences Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Flood Events")
plt.show()


# In[14]:


# Counting flood occurrences per year
yearly_flood_counts = df_cleaned['year'].value_counts().sort_index()

# Ploting flood occurrences by year
plt.figure(figsize=(12, 6))
sns.barplot(x=yearly_flood_counts.index, y=yearly_flood_counts.values, palette="Blues")
plt.title("Yearly Trend of Flood Events Pakistan")
plt.xlabel("Year")
plt.ylabel("Number of Flood Events")
plt.xticks(rotation=45)
plt.show()


# In[15]:


# Total flood incidents
total_floods = yearly_flood_counts.sum()
print(f"Total number of flood incidents in Pakistan 1980-2014 : {total_floods}")


# In[16]:


df_cleaned.columns


# # Correlation Matrix Between Flood Year and Imapct Factors

# In[17]:


# Relevant columns for correlation analysis
impact_columns = ['year','Deaths', 'Injured', 'Missing', 'Affected',
       'Destroyed Homes', 'Affected Homes', 'Heads of Livestock', 'Duration',
       'Affected Persons', 'Evacuated Persons', 'approved'] 
# columns to numeric format
df_cleaned[impact_columns] = df_cleaned[impact_columns].apply(pd.to_numeric, errors='coerce')
# Calculate correlation matrix
correlation_matrix = df_cleaned[impact_columns].corr()
# Heatmap to visualize correlation between year and impact factors
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Flood Year and Impact Factors")
plt.show()


# # Logistic Regression model

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Features and compute severity_score (same as before)
features = [
    'Deaths',
    'Injured',
    'Affected',
    'Missing',
    'Destroyed Homes',
    'Affected Homes',
    'Evacuated Persons']
LogisticRegression(class_weight='balanced')
# Severity score
df_cleaned['severity_score'] = (
    df_cleaned['Deaths'] +
    df_cleaned['Injured'] +
    df_cleaned['Affected'] +
    df_cleaned['Destroyed Homes'] +
    df_cleaned['Evacuated Persons'])

# Severity class from severity_score
# Adjusting bins for data distribution
df_cleaned['severity_class'] = pd.cut(df_cleaned['severity_score'],
                                      bins=[-1, 1000, 5000, float('inf')],
                                      labels=[0, 1, 2])  # 0=Low, 1=Medium, 2=High

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned['severity_class'], test_size=0.2, random_state=42)

# Train a Logistic Regression Model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance (Coefficients)
coeff_df = pd.DataFrame(model.coef_.T, index=features, columns=['Class 0', 'Class 1', 'Class 2'])
coeff_df['Mean Importance'] = coeff_df.abs().mean(axis=1)
coeff_df = coeff_df.sort_values(by='Mean Importance', ascending=False)

# Plot Coefficients as Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Mean Importance', y=coeff_df.index, data=coeff_df, palette='coolwarm')
plt.title("Logistic Regression Feature Importance for Flood Severity Classification")
plt.xlabel("Coefficient Magnitude (Mean)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# # Random Forest Regressor Model

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Features and target variable
features = ['year', 'Deaths', 'Injured', 'Missing', 'Affected',
       'Destroyed Homes', 'Affected Homes', 'Heads of Livestock', 'Duration',
       'Affected Persons', 'Evacuated Persons', 'approved']
target = 'Destroyed Homes' # Target variable (affected persons)
df_cleaned['severity_score'] = (
    df_cleaned['Deaths'] * 5 +
    df_cleaned['Injured'] * 2 +
    df_cleaned['Affected'] * 1 +
    df_cleaned['Destroyed Homes'] * 3 +
    df_cleaned['Evacuated Persons'] * 2
)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display Results
print(f"Random Forest Model Performance:\nR² Score: {r2:.2f}\nMean Absolute Error: {mae:.2f}")

# Feature Importance Analysis
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance for Flood Severity Prediction")
plt.show()


# # Random Forest model Prediction and Evaluation with confusion matrix with increasing severity score

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Define severity score
df_cleaned['severity_score'] = (
    df_cleaned['Affected'] * 1 +
    df_cleaned['Evacuated Persons'] * 2 +
    df_cleaned['Destroyed Homes'] * 3 +
    df_cleaned['Deaths'] * 5
)

# STEP 2: Create severity category (Low, Medium, High)
def categorize_severity(score):
    if score < 100:
        return 'Low'
    elif score < 1000:
        return 'Medium'
    else:
        return 'High'

df_cleaned['severity_level'] = df_cleaned['severity_score'].apply(categorize_severity)

# STEP 3: Define features and target
features = [
    'year', 'Month', 'Day', 'Deaths', 'Injured', 'Missing', 'Affected',
       'Destroyed Homes', 'Affected Homes', 'Heads of Livestock', 'Duration',
       'Affected Persons', 'Evacuated Persons', 'approved']
target = 'severity_level'

# Drop rows with missing values
df_model = df_cleaned[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# STEP 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# STEP 6: Predictions & Evaluation
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix: Severity Level Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 7: Feature Importance
importances = clf.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance for Severity Level Classification")
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Define features and target variable without the distroyed home 
features = [
    'Deaths',
    'Injured',
    'Affected',
    'Missing',
    'Affected Homes', 
    'Evacuated Persons'
]
target = 'severity_score' # Target variable (affected persons)
df_cleaned['severity_score'] = (
    df_cleaned['Deaths'] +
    df_cleaned['Injured'] +
    df_cleaned['Affected']  +
    df_cleaned['Evacuated Persons'] 
)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)

# Step 2: Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Make Predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate Model Performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display Results
print(f"Random Forest Model Performance:\nR² Score: {r2:.2f}\nMean Absolute Error: {mae:.2f}")

# Step 5: Feature Importance Analysis
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance for Flood Severity Prediction")
plt.show()


# # Feature Importance for Flood Severity Prediction 

# In[22]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming `model` trained RandomForestRegressor and `X_train` your feature set
importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='cornflowerblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Flood Severity Prediction")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', va='center')

plt.tight_layout()
plt.show()


# # 5 Flod Cross Validation

# In[23]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation (5-fold)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R² Score: {cv_scores.mean():.2f}")


# # XG Booster Regressor Model Performance

# In[24]:


from xgboost import XGBRegressor

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.7, random_state=42)
xgb_model.fit(X_train, y_train)

# Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate Model
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"XGBoost Model Performance:\nR² Score: {r2_xgb:.2f}\nMean Absolute Error: {mae_xgb:.2f}")


# # Cross validation R2 score

# In[25]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R² Score: {cv_scores.mean():.2f}")


# # Prediction and Evaluation with confusion matrix with increasing severity score

# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Define severity score
df_cleaned['severity_score'] = (
    df_cleaned['Affected'] * 1 +
    df_cleaned['Evacuated Persons'] * 2 +
    df_cleaned['Destroyed Homes'] * 3 +
    df_cleaned['Deaths'] * 5
)

# STEP 2: Create severity category (Low, Medium, High)
def categorize_severity(score):
    if score < 100:
        return 'Low'
    elif score < 1000:
        return 'Medium'
    else:
        return 'High'

df_cleaned['severity_level'] = df_cleaned['severity_score'].apply(categorize_severity)

# STEP 3: Define features and target
features = [
    'Affected', 'Evacuated Persons', 'Destroyed Homes',
    'Affected Homes', 'Deaths', 'Injured', 'Missing', 
    'Duration', 'Heads of Livestock'
]
target = 'severity_level'

# Drop rows with missing values
df_model = df_cleaned[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# STEP 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# STEP 6: Predictions & Evaluation
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title("Confusion Matrix: Severity Level Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 7: Feature Importance
importances = clf.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance for Severity Level Classification")
plt.show()


# In[27]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train model with balanced data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_resampled, y_resampled)


# # F1 Score 

# In[28]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
print("F1 Scores:", scores)
print("Mean F1:", scores.mean())


# In[29]:


# Map categories to numeric values explicitly (if justified)
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_train_numeric = y_train.map(mapping)


# In[ ]:





# In[30]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(X, y)

print("Best Params:", grid_search.best_params_)


# # Kfold R2 score for XGBregressor

# In[31]:


from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np

model = XGBRegressor()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train_numeric.iloc[train_index], y_train_numeric.iloc[test_index]
    
    model.fit(X_fold_train, y_fold_train)
    y_pred = model.predict(X_fold_test)
    
    score = r2_score(y_fold_test, y_pred)
    fold_scores.append(score)
    
    print(f"Fold {i+1} R² Score: {score:.2f}")

worst_fold = np.argmin(fold_scores)
print(f"\nWorst fold is Fold {worst_fold + 1} with R² = {fold_scores[worst_fold]:.2f}")


# # Random Forest After Expanding the Features

# In[32]:


#Expanded features and target
features = ['year', 'Month', 'Day', 'Deaths', 'Injured', 'Missing', 'Affected',
       'Destroyed Homes', 'Affected Homes', 'Heads of Livestock', 'Duration',
       'Affected Persons', 'Evacuated Persons', 'approved']
target = 'severity_score'  # Use the custom severity score

# Recalculate severity_score with available columns
df_cleaned['severity_score'] = (
    df_cleaned['Deaths'] * 5 +
    df_cleaned['Injured'] * 2 +
    df_cleaned['Affected'] * 1 +
    df_cleaned['Destroyed Homes'] * 3 +
    df_cleaned['Evacuated Persons'] * 2 +
    df_cleaned['Duration'] * 1.5
)

# Handling missing values if any
df_cleaned = df_cleaned[features + [target]].dropna()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df_cleaned[features], df_cleaned[target], test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest Model Performance:\nR² Score: {r2:.2f}\nMean Absolute Error: {mae:.2f}")

# Feature importance plot
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance for Flood Severity Prediction (Expanded Model)")
plt.show()


# # Displacement and Damge Ratio of Pakistan

# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Define improved displacement and damage indicators
df_cleaned['total_displaced'] = df_cleaned['Evacuated Persons'] + df_cleaned['Affected']

df_cleaned['damage_score'] = (
    df_cleaned['Affected Homes'] * 3 +
    df_cleaned['Deaths'] * 4 +
    df_cleaned['Injured'] * 2
)

# Step 2: Correlation between improved variables
correlation = df_cleaned[['total_displaced', 'damage_score']].corr()
print("Correlation Matrix:\n", correlation)

# Step 3: Scatter Plot
sns.scatterplot(data=df_cleaned, x='damage_score', y='total_displaced')
plt.title("Total Displacement vs. Flood Damage in Pakistan")
plt.xlabel("Damage Score")
plt.ylabel("Total Displaced (Evacuated + Affected)")
plt.tight_layout()
plt.show()

# Step 4: Displacement-to-Damage Ratio
df_cleaned['displacement_ratio'] = df_cleaned['total_displaced'] / (df_cleaned['damage_score'] + 1)
print("\nAverage Displacement-to-Damage Ratio:", df_cleaned['displacement_ratio'].mean())

