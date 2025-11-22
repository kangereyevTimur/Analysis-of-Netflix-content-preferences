# Analysis of Netflix Content Preferences

## 🚀 Project Overview  
This project analyzes Netflix movies and TV shows to understand viewer preferences across genres and countries.  
We explore content trends, genre distribution, and clustering by country and genre, and produce interactive visualizations.

## 📊 Dataset  
- **Source:** [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
- **Key Features:** title, type (Movie/TV Show), release year, country, listed genres, duration, etc.

## 🏛 Architecture
Netflix-Content-Analysis/
├── data/ # Dataset CSV
├── notebooks/ # Jupyter notebooks with analysis code
│ └── netflix_analysis.ipynb
├── visuals/ # Saved PNGs & HTML interactive plots
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── LICENSE # License file
- **Data Loading & Cleaning:** Pandas used to read CSV and fill missing values.  
- **EDA & Visualization:** Seaborn, Matplotlib, Plotly.  
- **Clustering:** MultiLabelBinarizer + KMeans for genre-based clustering.  
- **Results:** Visualized trends by year, genre, and country; interactive plots saved in `visuals/`.

## 🔍 Analysis Steps  

### 1. Exploratory Data Analysis (EDA)
```python
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/netflix_titles.csv')
df.fillna(0, inplace=True)

# Count of Movies vs TV Shows
sns.countplot(data=df, x='type')
plt.title('Movies vs TV Shows on Netflix')
plt.show()
Description: This plot shows the proportion of movies vs TV shows on Netflix.
# Number of releases per year
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
year_counts = df['release_year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title('Number of Netflix Releases Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Releases')
plt.show()
Description: Visualizes the trend of Netflix content releases over the years.
2. Top Countries and Genres
# Top 10 countries
top_countries = df['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by Netflix Content')
plt.show()
Description: Shows which countries produce the most Netflix content.
# Top 10 genres
from collections import Counter
genre_list = df['listed_in'].dropna().apply(lambda x: x.split(', '))
all_genres = [genre for sublist in genre_list for genre in sublist]
genre_counts = Counter(all_genres)
top_genres = genre_counts.most_common(10)

genres, counts = zip(*top_genres)
sns.barplot(x=list(counts), y=list(genres))
plt.title('Top 10 Netflix Genres')
plt.show()
Description: Displays the most popular genres on Netflix.
3. Clustering
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans

df['genres_list'] = df['listed_in'].apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
genre_matrix = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                            columns=mlb.classes_)

kmeans = KMeans(n_clusters=5, random_state=42)
df['genre_cluster'] = kmeans.fit_predict(genre_matrix)
sns.countplot(data=df, x='genre_cluster')
plt.title('Distribution of Netflix Titles by Genre Cluster')
plt.show()
Description: Groups titles into 5 clusters based on genres, helping identify content patterns.
4. Comparative Analysis by Country
import plotly.express as px

top5_countries = df['country'].value_counts().head(5).index
df_top5 = df[df['country'].isin(top5_countries)]
year_country = df_top5.groupby(['release_year', 'country']).size().reset_index(name='count')

fig = px.line(year_country, x='release_year', y='count', color='country',
              title='Netflix Release Trends in Top 5 Countries')
fig.show()
Description: Interactive plot showing content release trends over time for the top 5 countries.
📈 Metrics & Key Results
Total titles analyzed: 8,800
Number of unique genres: 40
Top genre: Dramas
Top country by content: United States
Clustering: 5 genre clusters
Trend: steady growth of Netflix releases since 2008

📊 Visualizations
Top 10 Genres

Shows the most frequent genres on Netflix.
Releases Per Year

Shows growth in content production over the years.
Top Countries by Content Volume

Highlights countries with most Netflix content.
Genre Clustering Distribution

Visual representation of the 5 genre clusters.
Interactive Plot — Releases by Year & Country
View Interactive Plot

# Future Work
Sentiment analysis of title descriptions
Build a recommendation system based on country + genre clusters
Explore advanced clustering methods (hierarchical clustering, DBSCAN)
Deploy interactive dashboards (Dash / Streamlit)

# Technologies & Tools
Python 3.x
pandas, numpy, seaborn, matplotlib, plotly, scikit-learn
Jupyter Notebook / Google Colab

# How to Run
git clone git@github.com:kangereyevTimur/Analysis-of-Netflix-content-preferences.git
cd Analysis-of-Netflix-content-preferences
pip install -r requirements.txt
jupyter notebook notebooks/netflix_analysis.ipynb
# or open in Google Colab

# Authors
Timur Kangereyev
Zhanel Karimzhanova

# Contact
Email: kangereev.timur@gmail.com; karimzhanovazhanel4@gmail.com
GitHub: kangereyevTimur; janelkr

# Acknowledgments
Kaggle for the Netflix dataset
Seaborn, Plotly, Matplotlib communities
Mentors and colleagues for feedback

#License
This project is licensed under the MIT License.





