'''import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
data = np.array([np.asarray(d).flatten() for d in data])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()'''
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dictionary from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Target feature size (2 hands * 21 landmarks * 2 coordinates = 84)
# Adjust this number if your intention was a single-hand model (42)
TARGET_FEATURES = 84

# --- FIX: Filter and Prepare Data ---
prepared_data = []
prepared_labels = []

# data_dict['data'] is a list of lists/arrays (the raw features)
# data_dict['labels'] is the corresponding list of labels
for raw_sample, label in zip(data_dict['data'], data_dict['labels']):
    # 1. Flatten the data: Convert list of lists of lists (if nested) to a single list
    flat_sample = np.asarray(raw_sample).flatten()

    # 2. Check for the correct, expected length
    if flat_sample.size == TARGET_FEATURES:
        prepared_data.append(flat_sample)
        prepared_labels.append(label)
    else:
        # This will print which samples are being dropped
        print(f"Dropping sample with incorrect feature count: {flat_sample.size}. Expected: {TARGET_FEATURES}")

# Convert the cleaned lists to NumPy arrays
data = np.asarray(prepared_data)
labels = np.asarray(prepared_labels)

# Check if any data remains
if data.size == 0:
    print("Error: No valid samples found after filtering. Check your data collection script and TARGET_FEATURES.")
else:
    print(f"Successfully loaded {data.shape[0]} valid samples.")

    # --- Original Training Logic ---
    # The redundant flattening line from the original code is removed,
    # as data is already flat and correct.

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels  # stratify is good practice for balanced classes
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    # Save the model
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    f.close()