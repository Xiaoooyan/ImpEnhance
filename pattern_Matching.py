import pandas as pd
import numpy as np

a_df = pd.read_excel('data/RH_text.xlsx')
b_df = pd.read_excel('data/day_cluster_parsed.xlsx')

a_df['time'] = pd.to_datetime(a_df['time'])
b_df['time'] = pd.to_datetime(b_df['time'])

a_df['date'] = a_df['time'].dt.date
b_df['date'] = b_df['time'].dt.date

def split_emotions(emotions):
    return set(emotions.split('、'))

a_df['emotion_list'] = a_df['emotion'].apply(split_emotions)
b_df_emotion_cols = [col for col in b_df.columns if col not in ['location', 'time', 'date']]

def compute_emotion_similarity(emotion_list, emotion_probs):
    emotion_set = set(emotion_probs.index)
    common_emotions = emotion_list.intersection(emotion_set)
    if not common_emotions:
        return 0

    similarity = sum(emotion_probs[emotion] for emotion in common_emotions)
    return similarity

locations = b_df['location'].unique()

results_df = pd.DataFrame()

for _, a_row in a_df.iterrows():
    date = a_row['date']
    emotion_list = a_row['emotion_list']

    row_result = {'a_index': a_row.name}

    matching_rows = b_df[b_df['date'] == date]

    for location in locations:
        location_rows = matching_rows[matching_rows['location'] == location]

        if not location_rows.empty:
            location_row = location_rows.iloc[0]
            emotion_probs = pd.Series(location_row[b_df_emotion_cols].values, index=b_df_emotion_cols)
            similarity = compute_emotion_similarity(emotion_list, emotion_probs)
        else:
            similarity = 0

        row_result[location] = similarity

    results_df = results_df.append(row_result, ignore_index=True)