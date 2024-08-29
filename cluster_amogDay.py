import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

df = pd.read_excel('RH_text.xlsx')

df = df[['time', 'emotion', 'location']]

df['time'] = pd.to_datetime(df['time'])

locations = df['location'].unique()

result = []

for location in locations:
    loc_data = df[df['location'] == location].copy()
    loc_data.loc[:, 'date'] = loc_data['time'].dt.date
    loc_data.loc[:, 'hour'] = loc_data['time'].dt.hour

    dates = loc_data['date'].unique()

    for date in dates:
        day_data = loc_data[loc_data['date'] == date]
        hourly_data = day_data.groupby('hour')['emotion'].value_counts(normalize=True).unstack(fill_value=0)

        hour_counts = day_data['hour'].value_counts().sort_index()

        hourly_vectors = hourly_data.values

        if len(hourly_vectors) < 2:
            continue

        distance_matrix = squareform(pdist(hourly_vectors, metric='euclidean'))

        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, affinity='precomputed',
                                             linkage='average').fit(distance_matrix)

        clusters = clustering.labels_
        cluster_data = pd.DataFrame({
            'hour': hourly_data.index,
            'cluster': clusters,
            'count': hour_counts.values,
            'vector': list(hourly_vectors)
        })

        new_time_segments = []

        for cluster in cluster_data['cluster'].unique():
            cluster_hours = cluster_data[cluster_data['cluster'] == cluster]
            total_count = cluster_hours['count'].sum()

            weighted_vector = np.average(
                np.array(cluster_hours['vector'].tolist()),
                weights=cluster_hours['count'].values,
                axis=0
            )

            new_time_segment = {
                'location': location,
                'date': date,
                'start_hour': cluster_hours['hour'].min(),
                'end_hour': cluster_hours['hour'].max(),
                'emotion_vector': weighted_vector
            }

            new_time_segments.append(new_time_segment)

        result.extend(new_time_segments)

result_df = pd.DataFrame(result)

result_df.to_excel('result.xlsx', index=False)
