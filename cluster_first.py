import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

df = pd.read_excel('RH_text.xlsx')

df['datetime'] = pd.to_datetime(df['time'])
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

mlb = MultiLabelBinarizer()
df['emotion'] = df['emotion'].apply(lambda x: x.split('、'))
emotion_encoded = mlb.fit_transform(df['emotion'])
emotion_columns = mlb.classes_
emotion_df = pd.DataFrame(emotion_encoded, columns=emotion_columns)
df = pd.concat([df, emotion_df], axis=1)

def calculate_hourly_emotion_ratios(group):
    total_count = len(group)
    emotion_ratios = group[emotion_columns].sum() / total_count
    return emotion_ratios

location_date_hour_groups = df.groupby(['location', 'date', 'hour'])
hourly_emotion_ratios = location_date_hour_groups.apply(calculate_hourly_emotion_ratios).reset_index()

location_groups = hourly_emotion_ratios.groupby('location')

def calculate_dtw_distance(ts1, ts2):
    distance, _ = fastdtw(ts1, ts2, dist=euclidean)
    return distance

location_clusters = {}
for location, group in location_groups:
    date_groups = group.groupby('date')
    date_series = []
    date_to_series = {}

    for date, date_group in date_groups:
        date_series.append(date_group[emotion_columns].values)
        date_to_series[date] = date_group[emotion_columns].values

    date_series_flattened = np.vstack(date_series)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(date_series_flattened)
    cluster_labels = kmeans.labels_

    cluster_centers = []
    for cluster in range(3):
        cluster_members = [date_series[i] for i in range(len(date_series)) if cluster_labels[i] == cluster]
        cluster_center = np.mean([np.mean(member, axis=0) for member in cluster_members], axis=0)
        cluster_centers.append(cluster_center)

    dtw_distances = np.zeros((len(date_series), len(cluster_centers)))
    for i, series in enumerate(date_series):
        for j, center in enumerate(cluster_centers):
            dtw_distances[i, j] = calculate_dtw_distance(series, center)

    Z = linkage(dtw_distances, method='ward')
    max_d = 0.5 * np.max(Z[:, 2])
    hierarchical_labels = fcluster(Z, max_d, criterion='distance')

    cluster_emotion_vectors = []
    cluster_dates = {i: [] for i in np.unique(hierarchical_labels)}

    for i, date in enumerate(date_series):
        cluster = hierarchical_labels[i]
        cluster_dates[cluster].append(list(date_to_series.keys())[i])

    location_clusters[location] = cluster_dates