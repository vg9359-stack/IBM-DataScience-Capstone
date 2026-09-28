# ==============================================================================
# IBM DATA SCIENCE CAPSTONE PROJECT: COMPLETE END-TO-END CODE
# ==============================================================================

import sqlite3
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import dash
from dash import dcc, html
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# STAGE 1 & 2: DATA COLLECTION & WRANGLING
# ==============================================================================
print("--- STAGE 1 & 2: Loading & Cleaning Dataset ---")

# Load standard IBM Capstone Dataset directly
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
df = pd.read_csv(dataset_url)

# Define 'Class' target variable (1 for successful landing, 0 otherwise)
df['Class'] = df['Outcome'].apply(lambda x: 1 if 'True' in str(x) else 0)

# Feature Selection and One-Hot Encoding
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
features_one_hot = features_one_hot.astype('float64')

print(f"Dataset Loaded Successfully! Feature Shape: {features_one_hot.shape} (83 expected columns)")


# ==============================================================================
# STAGE 3A: EXPLORATORY DATA ANALYSIS (EDA WITH SEABORN)
# ==============================================================================
print("\n--- STAGE 3A: Generating EDA Seaborn Plots ---")

plt.figure(figsize=(10, 6))
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect=2, kind="strip")
plt.ylabel("Launch Site", fontsize=15)
plt.xlabel("Flight Number", fontsize=15)
plt.title("Flight Number vs. Launch Site", fontsize=16)
plt.savefig("flight_vs_launchsite.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect=2, kind="strip")
plt.ylabel("Launch Site", fontsize=15)
plt.xlabel("Payload Mass (kg)", fontsize=15)
plt.title("Payload Mass vs. Launch Site", fontsize=16)
plt.savefig("payload_vs_launchsite.png")
plt.close()

orbit_success = df.groupby('Orbit')['Class'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x="Orbit", y="Class", data=orbit_success, palette="Blues_d")
plt.ylabel("Success Rate", fontsize=12)
plt.xlabel("Orbit Type", fontsize=12)
plt.title("Success Rate by Orbit Type", fontsize=14)
plt.savefig("orbit_success_rate.png")
plt.close()

print("Plots saved: 'flight_vs_launchsite.png', 'payload_vs_launchsite.png', 'orbit_success_rate.png'")


# ==============================================================================
# STAGE 3B: EDA WITH SQL
# ==============================================================================
print("\n--- STAGE 3B: Executing SQL Queries ---")

conn = sqlite3.connect(':memory:')
df.to_sql('SPACEXTBL', conn, index=False, if_exists='replace')

q1 = pd.read_sql_query("SELECT * FROM SPACEXTBL LIMIT 20", conn)
q2 = pd.read_sql_query("SELECT sum(PayloadMass) AS Total_Payload_Mass FROM SPACEXTBL", conn)
q3 = pd.read_sql_query("SELECT min(PayloadMass) AS Min_Payload_Mass FROM SPACEXTBL", conn)
q4 = pd.read_sql_query('SELECT count(Outcome) AS MISSION_OUTCOME_COUNT, LaunchSite FROM SPACEXTBL GROUP BY LaunchSite', conn)

print("SQL Outcome Count by Launch Site:")
print(q4)


# ==============================================================================
# STAGE 4A: INTERACTIVE VISUAL ANALYTICS (FOLIUM MAP)
# ==============================================================================
print("\n--- STAGE 4A: Generating Folium Interactive Map ---")

# Use Esri World Imagery (satellite view) which requires NO API key
site_map = folium.Map(
    location=[28.562302, -80.577356], 
    zoom_start=5,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri'
)

marker_cluster = MarkerCluster()
site_map.add_child(marker_cluster)

for index, row in df.iterrows():
    coordinate = [28.562302, -80.577356]
    icon_color = 'green' if row['Class'] == 1 else 'red'
    marker = folium.Marker(
        location=coordinate,
        popup=f"Site: {row['LaunchSite']} | Class: {row['Class']}",
        icon=folium.Icon(color='white', icon_color=icon_color)
    )
    marker_cluster.add_child(marker)

site_map.save("spacex_launch_map.html")
print("Folium map saved as 'spacex_launch_map.html'")
# ==============================================================================
# STAGE 4B: DASHBOARD APP SETUP (PLOTLY DASH)
# ==============================================================================
print("\n--- STAGE 4B: Initializing Plotly Dash App Structure ---")

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    dcc.Dropdown(
        id='site-dropdown',
        options=[
            {'label': 'All Sites', 'value': 'ALL'},
            {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
            {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
            {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
            {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}
        ],
        value='ALL',
        placeholder="Select a Launch Site",
        searchable=True
    ),
    html.Br(),
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(
        id='payload-slider',
        min=0, max=10000, step=1000,
        marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500', 10000: '10000'},
        value=[0, 10000]
    )
])


# ==============================================================================
# STAGE 5: MACHINE LEARNING PREDICTIVE ANALYSIS
# ==============================================================================
print("\n--- STAGE 5: Machine Learning Model Training & Evaluation ---")

X = features_one_hot.copy()
Y = df['Class'].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(f"Test Set Records Count: {X_test.shape[0]}")

# Logistic Regression
lr_params = {'C': [0.01, 0.1, 1], 'solver': ['lbfgs']}
lr_cv = GridSearchCV(LogisticRegression(), lr_params, cv=10)
lr_cv.fit(X_train, Y_train)

# Support Vector Machine (SVM)
svm_params = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': np.logspace(-3, 3, 5)}
svm_cv = GridSearchCV(SVC(), svm_params, cv=10)
svm_cv.fit(X_train, Y_train)

# Decision Tree Classifier
tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8]}
tree_cv = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=10)
tree_cv.fit(X_train, Y_train)

# K-Nearest Neighbors (KNN)
knn_params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn_cv = GridSearchCV(KNeighborsClassifier(), knn_params, cv=10)
knn_cv.fit(X_train, Y_train)

results = {
    "Logistic Regression Test Accuracy": f"{lr_cv.score(X_test, Y_test) * 100:.2f}%",
    "SVM Best Kernel": svm_cv.best_params_['kernel'],
    "SVM Test Accuracy": f"{svm_cv.score(X_test, Y_test) * 100:.2f}%",
    "Decision Tree Test Accuracy": f"{tree_cv.score(X_test, Y_test) * 100:.2f}%",
    "KNN Test Accuracy": f"{knn_cv.score(X_test, Y_test) * 100:.2f}%"
}

print("\n================ FINAL RESULTS ================")
for k, v in results.items():
    print(f"{k}: {v}")
print("===============================================")
