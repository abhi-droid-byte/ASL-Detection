import pickle

from sklearn.ensemble import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check the shape of each element in data_dict['data']
shapes = [np.shape(item) for item in data_dict['data']]
print("Shapes of elements in data_dict['data']: ", shapes)

# Convert each element to a numpy array and pad or reshape if necessary
data = [np.asarray(item) for item in data_dict['data']]

# If all elements are not of the same shape, you may need to pad or truncate them
# Here's an example of how to pad the arrays to the maximum shape
max_shape = tuple(max(s) for s in zip(*[item.shape for item in data]))
padded_data = [np.pad(item, [(0, max_dim - item.shape[i]) for i, max_dim in enumerate(max_shape)], mode='constant') for item in data]

# Convert the list of padded arrays into a single numpy array
data = np.array(padded_data)

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)