import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the dataset
data = [
    {"Year": 1987, "GDP (current US$)": 64904761904.7619},
    {"Year": 1988, "GDP (current US$)": 74842105263.1579},
    {"Year": 1989, "GDP (current US$)": 80731578947.3684},
    {"Year": 1990, "GDP (current US$)": 79523809523.8095},
    {"Year": 1991, "GDP (current US$)": 76666666666.6667},
    {"Year": 1992, "GDP (current US$)": 73906020558.0029},
    {"Year": 1993, "GDP (current US$)": 65648189143.7174},
    {"Year": 1994, "GDP (current US$)": 52549580264.8064},
    {"Year": 1995, "GDP (current US$)": 48213856468.5718},
    {"Year": 1996, "GDP (current US$)": 44558831005.0621},
    {"Year": 1997, "GDP (current US$)": 50151531591.7317},
    {"Year": 1998, "GDP (current US$)": 41882523345.1804},
    {"Year": 1999, "GDP (current US$)": 31580639553.8298},
    {"Year": 2000, "GDP (current US$)": 32375083934.8241},
    {"Year": 2001, "GDP (current US$)": 39309580983.2282},
    {"Year": 2002, "GDP (current US$)": 43956163612.0433},
    {"Year": 2003, "GDP (current US$)": 52010355753.0461},
    {"Year": 2004, "GDP (current US$)": 67220154164.3166},
    {"Year": 2005, "GDP (current US$)": 89238865118.5263},
    {"Year": 2006, "GDP (current US$)": 111884752475.248},
    {"Year": 2007, "GDP (current US$)": 148733861386.139},
    {"Year": 2008, "GDP (current US$)": 188110390659.515},
    {"Year": 2009, "GDP (current US$)": 121552153444.124},
    {"Year": 2010, "GDP (current US$)": 141209170427.233},
    {"Year": 2011, "GDP (current US$)": 169333835201.554},
    {"Year": 2012, "GDP (current US$)": 182591753827.949},
    {"Year": 2013, "GDP (current US$)": 190498811460.028},
    {"Year": 2014, "GDP (current US$)": 133503871861.723},
    {"Year": 2015, "GDP (current US$)": 91030967789.0717},
    {"Year": 2016, "GDP (current US$)": 93355869403.9223},
    {"Year": 2017, "GDP (current US$)": 112090505081.739},
    {"Year": 2018, "GDP (current US$)": 130891088293.55},
    {"Year": 2019, "GDP (current US$)": 153883047509.577},
    {"Year": 2020, "GDP (current US$)": 156617722013.342},
    {"Year": 2021, "GDP (current US$)": 199765859570.935},
    {"Year": 2022, "GDP (current US$)": 161989520721.19},
    {"Year": 2023, "GDP (current US$)": 178757021386.809},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["Year", "GDP (current US$)"]])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Add PCA results to DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Visualize the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['Year'], cmap='viridis')
plt.colorbar(label='Year')
plt.title('PCA of Ukraine GDP Data')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(df)
