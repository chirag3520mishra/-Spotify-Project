# import the libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import streamlit as st


data = pd.read_csv(r'C:\Users\Home\Documents\project\datasets\train.csv', encoding='latin1')

# Popularity threshold (let's consider top 25% of the data)
popularity_threshold = data['popularity'].quantile(0.75)

# Filter for the most popular tracks
most_popular_tracks = data[data['popularity'] >= popularity_threshold]

# Features to analyze
features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Plotting the distributions of these features
fig1 = go.Figure()
for feature in features:
    fig1.add_trace(go.Box(y=most_popular_tracks[feature], name=feature))

fig1.update_layout(
    title="Distribution of Musical Features in Most Popular Tracks",
    yaxis_title="Feature Value",
    xaxis_title="Musical Features"
)


# value counts of each genre
genre_counts = data['track_genre'].value_counts()

# mean popularity, energy, and danceability for each genre
genre_avg = data.groupby('track_genre')[['popularity', 'energy', 'danceability']].mean().reset_index()

# bar chart for genre counts
fig_counts = go.Figure(data=[
    go.Bar(x=genre_counts.index, y=genre_counts.values)
])
fig_counts.update_layout(
    title="Counts of Tracks by Genre",
    xaxis_title="Genre",
    yaxis_title="Count",
    xaxis={'categoryorder':'total descending'}
)


# Normalize the values to get them on a similar scale(this is done bcs the popularity values are much higher compared to other two features)
scaler = MinMaxScaler()
genre_avg[['popularity', 'energy', 'danceability']] = scaler.fit_transform(genre_avg[['popularity', 'energy', 'danceability']])

# bar chart for average normalized popularity, energy, and danceability by genre
fig_means = go.Figure(data=[
    go.Bar(name='Popularity', x=genre_avg['track_genre'], y=genre_avg['popularity']),
    go.Bar(name='Energy', x=genre_avg['track_genre'], y=genre_avg['energy']),
    go.Bar(name='Danceability', x=genre_avg['track_genre'], y=genre_avg['danceability'])
])
fig_means.update_layout(
    barmode='group',
    title="Normalized Average Popularity, Energy, and Danceability by Genre",
    xaxis_title="Genre",
    yaxis_title="Normalized Average Value",
    xaxis={'categoryorder':'total descending'}
)

# Count the occurrences of each artist
artist_counts = data['artists'].value_counts()

# Find the top artists for each genre
genre_artist_counts = data.groupby(['track_genre', 'artists']).size().reset_index(name='counts')

# Sort the artists within each genre by their count in descending order
genre_artist_counts = genre_artist_counts.sort_values(by=['track_genre', 'counts'], ascending=[True, False])

# Print the overall top artists
print("Top Artists Across All Genres:")
print(artist_counts.head(10))  # Adjust the number to display more or fewer top artists

# Print the top artists within each genre
print("\nTop Artists Within Each Genre:")
top_artists_per_genre = genre_artist_counts.groupby('track_genre').head(5)  # Adjust the number to change top N artists per genre
print(top_artists_per_genre.head(10))

# let's visualise the values
artist_counts_df = artist_counts.reset_index()
artist_counts_df.columns = ['artists', 'count']

fig2 = px.bar(artist_counts_df.head(10), x='artists', y='count', title='Top 10 Artists Across All Genres')


top_genre = genre_artist_counts['track_genre'].value_counts().idxmax()
top_artists_in_top_genre = genre_artist_counts[genre_artist_counts['track_genre'] == top_genre].head(10)

fig_genre = px.bar(top_artists_in_top_genre, x='artists', y='counts', title=f'Top Artists in {top_genre} Genre')

# Group by album and count the number of popular tracks
popular_albums = most_popular_tracks.groupby('album_name').size().reset_index(name='popular_tracks_count')

# Sort the albums by the number of popular tracks in descending order
popular_albums_sorted = popular_albums.sort_values('popular_tracks_count', ascending=False)

# Visualize the top albums with the most popular tracks
fig3 = px.bar(popular_albums_sorted.head(10),  # Adjust the number to display more or fewer top albums
             x='album_name',
             y='popular_tracks_count',
             title='Top Albums with the Most Popular Tracks')
fig3.update_layout(
    xaxis_title="Album Name",
    yaxis_title="Number of Popular Tracks",
    xaxis_tickangle=-45  # Rotate the album names for better readability
)

attributes = [
    'tempo', 'valence', 'acousticness', 'danceability',
    'energy', 'loudness', 'instrumentalness'
]

# correlation matrix for the selected attributes
correlation_matrix = data[attributes].corr()

# heatmap to visualize the correlation matrix
fig4 = px.imshow(correlation_matrix,
                x=attributes,
                y=attributes,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title='Correlation of Musical Features')

# layout to put the x-axis labels at the bottom
fig4.update_xaxes(side="bottom")


explicit_comparison = data.groupby('explicit').mean(numeric_only=True)

explicit_comparison.reset_index(inplace=True)

print(explicit_comparison)
attributes_to_compare = ['popularity', 'danceability', 'energy', 'valence']

fig5 = make_subplots(rows=1, cols=len(attributes_to_compare), subplot_titles=attributes_to_compare)

# bar plot for each attribute
for i, attribute in enumerate(attributes_to_compare, 1):
    fig5.add_trace(
        go.Bar(x=explicit_comparison['explicit'], y=explicit_comparison[attribute], name=attribute), 
        row=1, col=i
    )

# layout for a better view
fig5.update_layout(title_text="Comparison of Track Attributes by Explicit Content", showlegend=False)


fig6 = px.scatter(data, x='energy', y='danceability', title='Energy vs Danceability of Tracks',
                 trendline='ols',  # Add a trendline to see the overall trend
                 labels={'energy': 'Energy', 'danceability': 'Danceability'},
                 trendline_color_override='red')

fig7 = px.scatter(data, x='danceability', y='popularity', title='Danceability vs Popularity',
                 trendline='ols',  # Ordinary Least Squares regression line
                 labels={'danceability': 'Danceability', 'popularity': 'Popularity'},
                 trendline_color_override='red')


fig8 = px.scatter(data, x='valence', y='popularity', title='Valence vs Popularity',
                 trendline='ols',  # Ordinary Least Squares regression line
                 labels={'valence': 'Valence', 'popularity': 'Popularity'},
                 trendline_color_override='red')  # Custom trendline color for visibility

# let's assume that 50% as threshold for differentiating
instrumentalness_threshold = 0.5
data['track_type'] = data['instrumentalness'].apply(lambda x: 'Instrumental' if x > instrumentalness_threshold else 'Vocal')

# average popularity for instrumental and vocal tracks
average_popularity = data.groupby('track_type')['popularity'].mean().reset_index()

# average popularity
fig9 = px.bar(average_popularity, x='track_type', y='popularity', title='Average Popularity of Instrumental vs Vocal Tracks')
fig9.update_layout(xaxis_title="Track Type", yaxis_title="Average Popularity")


# Identify artists who have produced tracks in multiple genres
artist_genres = data.groupby('artists')['track_genre'].nunique()
multi_genre_artists = artist_genres[artist_genres > 1].index.tolist()

# Mark tracks as 'Multi Genre Artist' or 'Single Genre Artist'
data['artist_type'] = data['artists'].apply(lambda x: 'Multi Genre Artist' if not pd.isna(x) and x in multi_genre_artists
                                            else ('Single Genre Artist' if not pd.isna(x) else 'Unknown'))

# average popularity for tracks by artist type
average_popularity_by_artist_type = data.groupby('artist_type')['popularity'].mean().reset_index()

# Plot
fig10 = px.bar(average_popularity_by_artist_type, x='artist_type', y='popularity',
             title='Average Popularity of Tracks by Artist Genre Crossover')
fig10.update_layout(xaxis_title="Artist Type", yaxis_title="Average Popularity")


# thresholds for the top 10% and bottom 10% of tracks based on popularity
top_10_threshold = data['popularity'].quantile(0.90)
bottom_10_threshold = data['popularity'].quantile(0.10)

# dataset filtering for the top 10% and bottom 10% popular tracks
top_10_tracks = data[data['popularity'] >= top_10_threshold]
bottom_10_tracks = data[data['popularity'] <= bottom_10_threshold]

# attributes to compare
attributes = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness']

# mean for these attributes in both groups
top_10_means = top_10_tracks[attributes].mean()
bottom_10_means = bottom_10_tracks[attributes].mean()

# comparison plot
fig11 = go.Figure(data=[
    go.Bar(name='Top 10% Popular Tracks', x=attributes, y=top_10_means),
    go.Bar(name='Bottom 10% Popular Tracks', x=attributes, y=bottom_10_means)
])
fig11.update_layout(
    barmode='group',
    title="Comparison of Musical Attributes between Top 10% and Bottom 10% Popular Tracks",
    xaxis_title="Attributes",
    yaxis_title="Average Value"
)


high_danceability_threshold = data['danceability'].quantile(0.75)
low_energy_threshold = data['energy'].quantile(0.25)

unusual_tracks = data[(data['danceability'] >= high_danceability_threshold) & (data['energy'] <= low_energy_threshold)]

fig12 = px.histogram(unusual_tracks, x='popularity', nbins=30, title='Popularity Distribution of Tracks with High Danceability and Low Energy')
fig12.update_layout(xaxis_title="Popularity", yaxis_title="Count")

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
outliers_attributes = ['duration_ms', 'loudness', 'danceability', 'energy', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']


fig13 = make_subplots(rows=4, cols=3, subplot_titles=outliers_attributes,  horizontal_spacing=0.05, vertical_spacing=0.1) 

# control the size and spacing of the plot
fig_width = 1200
fig_height = 800

for i, attribute in enumerate(outliers_attributes): 
    row = (i // 3) + 1
    col = (i % 3) + 1
    outliers_df = detect_outliers(data, attribute) 
    fig13.add_trace(
        go.Scatter(x=outliers_df.index, y=outliers_df[attribute], mode='markers', name=attribute),
        row=row, col=col
    )


fig13.update_layout(
    title_text="Outliers in Different Track Attributes",
    showlegend=False,
    height=fig_height,
    width=fig_width
)

fig13.update_xaxes(tickangle=-45, tickfont=dict(size=10))
fig13.update_yaxes(tickfont=dict(size=10))









