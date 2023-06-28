import pandas as pd
import numpy as np
import tensorflow as tf

# Load dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Separate features and target
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df.iloc[:, -1].values.reshape(-1, 1)

# Split dataset into train and test sets
train_ratio = 0.8
train_size = int(train_ratio * diabetes_df.shape[0])
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Standardize features
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')
model.save('diabetes_model.h5')
np.save('mean.npy', mean)
np.save('std.npy', std)
