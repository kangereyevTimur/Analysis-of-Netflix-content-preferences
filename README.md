# Analysis of Netflix Content Preferences

## 🚀 Project Overview  
This project analyzes Netflix movies and TV shows to understand viewer preferences across genres and countries.  
We explore content trends, genre distribution, clustering by country and genre, and produce interactive visualizations.

## 📊 Dataset  
- **Source:** [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
- **Key Features:** title, type (Movie/TV Show), release year, country, listed genres, duration, etc.

## 🏛 Architecture
Netflix-Content-Analysis/
├── data/ # Original dataset CSV
├── notebooks/ # Jupyter notebook(s) with all code
│ └── netflix_analysis.ipynb
├── visuals/ # Saved PNG and HTML interactive plots
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── LICENSE # License file
- **data/** – хранит CSV с Netflix контентом  
- **notebooks/** – основной анализ: EDA, визуализация, кластеризация  
- **visuals/** – итоговые графики, PNG + интерактивные HTML  
- **README.md** – документация и описание проекта  
- **requirements.txt** – зависимости для Python  

## 🔍 Analysis Steps  

### 1. Exploratory Data Analysis (EDA)
```python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/netflix_titles.csv')
df.fillna(0, inplace=True)
```
# Count of Movies vs TV Shows
sns.countplot(data=df, x='type')
plt.title('Movies vs TV Shows on Netflix')
plt.show()
Description: Shows the proportion of movies vs TV shows. Helps to understand Netflix content type distribution.
# Number of releases per year
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
year_counts = df['release_year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title('Number of Netflix Releases Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Releases')
plt.show()
Description: Visualizes the yearly trend of Netflix content production.
```2. Top Countries and Genres
# Top 10 countries
top_countries = df['country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by Netflix Content')
plt.show()
Description: Highlights which countries produce the most Netflix content.
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
Description: Shows the most popular genres and their distribution among Netflix titles.
```
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
Description: Groups titles into 5 genre clusters. Helps to identify content patterns and similarity between genres.
```
```4. Comparative Analysis by Country
import plotly.express as px

top5_countries = df['country'].value_counts().head(5).index
df_top5 = df[df['country'].isin(top5_countries)]
year_country = df_top5.groupby(['release_year', 'country']).size().reset_index(name='count')

fig = px.line(year_country, x='release_year', y='count', color='country',
              title='Netflix Release Trends in Top 5 Countries')
fig.show()
Description: Interactive line plot showing the number of releases per year for the top 5 countries, highlighting trends and comparing regional activity.
```
📈 Metrics & Key Results
Total titles analyzed: 8,800
Number of unique genres: 40
Top genre: Dramas
Top country by content: United States
Clustering: 5 genre clusters
Trend: steady growth of Netflix releases since 2008
📊 Visualizations
Top 10 Genres
![Top Genres](visuals/top_genres.png)

Shows the frequency of the most popular genres on Netflix.

Number of Releases Per Year
![Releases per Year](visuals/releases_per_year.png)
Shows the yearly trend of Netflix content releases.

Top Countries by Content Volume
![Top Countries](visuals/top_countries.png)
Highlights countries with the highest number of Netflix titles.

Distribution of Genre Clusters
![Genre Clusters](visuals/genre_clusters.png)
Represents 5 genre clusters, showing how titles group by content type.
Interactive Plot — Releases by Year & Country
View Interactive Plot
Interactive line chart for top 5 countries, showing content release trends over time.
🔮 Future Work
Sentiment analysis of title descriptions
Build a recommendation system based on country + genre clusters
Explore advanced clustering methods (hierarchical clustering, DBSCAN)
Deploy interactive dashboards (Dash / Streamlit)
🛠️ Technologies & Tools
Python 3.x
pandas, numpy, seaborn, matplotlib, plotly, scikit-learn
Jupyter Notebook / Google Colab
✅ How to Run
git clone git@github.com:kangereyevTimur/Analysis-of-Netflix-content-preferences.git
cd Analysis-of-Netflix-content-preferences
pip install -r requirements.txt
jupyter notebook notebooks/netflix_analysis.ipynb
# or open in Google Colab
👥 Authors
Timur Kangereyev
Zhanel Karimzhanova
📬 Contact
Email: kangereev.timur@gmail.com
GitHub: kangereyevTimur
🙏 Acknowledgments
Kaggle for the Netflix dataset
Seaborn, Plotly, Matplotlib communities
Mentors and colleagues for feedback
📄 License
This project is licensed under the MIT License.







