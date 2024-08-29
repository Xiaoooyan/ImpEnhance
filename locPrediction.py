import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate, GRU, Bidirectional, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from geopy.distance import geodesic
import tensorflow_hub as hub
import tensorflow_text as text

df = pd.read_excel('data/dddaaaid.xlsx')

texts = df['content'].tolist()
labels = df['location'].tolist()
events = df['event'].tolist()
emotions = df['emotion'].apply(lambda x: x.split('、')).tolist()

preprocess_saved_model_path = 'bert_zh_preprocess'
encoder_saved_model_path = 'bert_zh_L-12_H-768_A-12'

bert_preprocess = tf.saved_model.load(preprocess_saved_model_path)
bert_encoder = tf.saved_model.load(encoder_saved_model_path)

def bert_encode(texts):
    preprocessed_texts = bert_preprocess(tf.constant(texts))
    return bert_encoder(preprocessed_texts)['pooled_output']

bert_texts = bert_encode(texts).numpy()

le_labels = LabelEncoder()
labels_encoded = le_labels.fit_transform(labels)
num_classes = len(set(labels_encoded))

labels_categorical = tf.keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

le_events = LabelEncoder()
events_encoded = le_events.fit_transform(events)
num_event_classes = len(le_events.classes_)
events_categorical = tf.keras.utils.to_categorical(events_encoded, num_classes=num_event_classes)

mlb_emotions = MultiLabelBinarizer()
emotions_encoded = mlb_emotions.fit_transform(emotions)
num_emotion_classes = len(mlb_emotions.classes_)

df['datetime'] = pd.to_datetime(df['time'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df['is_weekend'] = df['datetime'].dt.weekday >= 5
time_features = df[['year', 'month', 'day', 'hour', 'weekday', 'is_weekend']]

scaler = StandardScaler()
time_features = scaler.fit_transform(time_features)

X_train_text, X_test_text, y_train, y_test, train_indices, test_indices = train_test_split(
    bert_texts, labels_categorical, range(len(labels_categorical)), test_size=0.2, random_state=42
)
X_train_time, X_test_time = train_test_split(time_features, test_size=0.2, random_state=42)
X_train_event, X_test_event = train_test_split(events_encoded, test_size=0.2, random_state=42)
X_train_emotion, X_test_emotion = train_test_split(emotions_encoded, test_size=0.2, random_state=42)

text_input = Input(shape=(768,), dtype='float32', name='text_input')
dense_text = Dense(256, activation='relu')(text_input)
dropout_text = Dropout(0.5)(dense_text)

time_input = Input(shape=(time_features.shape[1],), name='time_input')
time_dense1 = Dense(128, activation='relu')(time_input)
time_dense2 = Dense(64, activation='relu')(time_dense1)
time_dense3 = Dropout(0.5)(time_dense2)

event_input = Input(shape=(1,), dtype='int32', name='event_input')
event_embedding = Embedding(input_dim=num_event_classes, output_dim=64, input_length=1)(event_input)
event_flatten = Flatten()(event_embedding)
event_dense = Dense(64, activation='relu')(event_flatten)

emotion_input = Input(shape=(num_emotion_classes,), name='emotion_input')
emotion_dense = Dense(64, activation='relu')(emotion_input)

concatenated = Concatenate()([dropout_text, time_dense3, event_dense, emotion_dense])
reshaped = Reshape((1, concatenated.shape[1]))(concatenated)

bi_gru = Bidirectional(GRU(64, return_sequences=True))(reshaped)
bi_gru = Bidirectional(GRU(64))(bi_gru)

main_output = Dense(num_classes, activation='softmax', name='main_output')(bi_gru)
aux_output = Dense(num_event_classes, activation='softmax', name='aux_output')(event_dense)

model = Model(inputs=[text_input, time_input, event_input, emotion_input], outputs=[main_output, aux_output])

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_encoded), y=labels_encoded)
class_weights_array = np.array([class_weights[i] for i in range(len(class_weights))])

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.class_weights = class_weights
        self.cce = tf.keras.losses.CategoricalCrossentropy()
    def call(self, y_true, y_pred):
        weights = tf.gather(self.class_weights, tf.argmax(y_true, axis=-1))
        loss = self.cce(y_true, y_pred)
        weights = tf.cast(weights, tf.float32)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss={'main_output': WeightedCategoricalCrossentropy(class_weights_array),
                    'aux_output': tf.keras.losses.SparseCategoricalCrossentropy()},
              metrics={'main_output': 'accuracy', 'aux_output': 'accuracy'},
              loss_weights={'main_output': 1.0, 'aux_output': 0.5})
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_text, X_train_time, X_train_event, X_train_emotion), (y_train, X_train_event))).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_text, X_test_time, X_test_event, X_test_emotion), (y_test, X_test_event))).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=callbacks)

y_pred = model.predict(val_dataset)
y_pred_probabilities = y_pred[0]

df = pd.read_excel('data/RH_text.xlsx')
geo_probs_df = pd.read_excel('data/day_cluster_gailv.xlsx')

df['id'] = df.index
geo_probs_df['id'] = geo_probs_df.index
test_ids = df.iloc[test_indices]['id'].values
y_pred_probabilities = y_pred[0]
geo_probs = geo_probs_df.loc[test_ids].values
assert y_pred_probabilities.shape[0] == geo_probs.shape[0], "The number of rows of prediction probability and geographical probability is inconsistent"

combined_probs = np.concatenate([y_pred_probabilities, geo_probs], axis=1)

log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg.fit(combined_probs, labels_encoded[test_indices])

y_combined_pred_probabilities = log_reg.predict_proba(combined_probs)

predicted_labels = le_labels.inverse_transform(np.argmax(y_combined_pred_probabilities, axis=1))
true_labels = le_labels.inverse_transform(labels_encoded[test_indices])

def calculate_distance(loc1, loc2):
    return geodesic(loc1, loc2).km

def find_location_coordinates(location_name, coords_df):
    loc = coords_df[coords_df['location'] == location_name]
    if not loc.empty:
        return (loc['latitude'].values[0], loc['longitude'].values[0])
    else:
        return None

def calculate_distance_between_labels(pred_label, true_label, coords_df):
    pred_coords = find_location_coordinates(pred_label, coords_df)
    true_coords = find_location_coordinates(true_label, coords_df)
    if pred_coords and true_coords:
        return calculate_distance(pred_coords, true_coords)
    return float('inf')

distances = [calculate_distance_between_labels(pred, true, location_coords_df) for pred, true in zip(predicted_labels, true_labels)]

median_distance = np.median(distances)
mean_distance = np.mean(distances)

def within_distance_threshold(distances, threshold):
    return np.sum(np.array(distances) <= threshold) / len(distances)
dist_3km_ratio = within_distance_threshold(distances, 3)
dist_5km_ratio = within_distance_threshold(distances, 5)

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"accuracy: {accuracy:.2f}")
print(f"median_distance: {median_distance:.2f} km")
print(f"mean_distance: {mean_distance:.2f} km")
print(f"acc@3km: {dist_3km_ratio:.2f}")
print(f"acc@5km: {dist_5km_ratio:.2f}")