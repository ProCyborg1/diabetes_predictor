import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('diabetes_model.h5')

# Load the mean and standard deviation values used for normalization
mean = np.load('mean.npy')
std = np.load('std.npy')

# Function to preprocess the input data
def preprocess_data(data, mean, std):
    # Normalize the data using the mean and standard deviation
    data = (data - mean) / std
    return data

# Input the data
pregnancies = float(input('Enter number of times pregnant: '))
glucose = float(input('Enter plasma glucose concentration: '))
blood_pressure = float(input('Enter diastolic blood pressure: '))
skin_thickness = float(input('Enter triceps skinfold thickness: '))
insulin = float(input('Enter 2-hour serum insulin: '))
bmi = float(input('Enter body mass index: '))
dpf = float(input('Enter diabetes pedigree function: '))
age = float(input('Enter age: '))

# Preprocess the input data
data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
data = preprocess_data(data, mean, std)
data = data.reshape(1, -1) # Reshape the data to match the model input shape

# Make the prediction
prediction = model.predict(data)

# Print the result
if prediction[0][0] > 0.5:
    print('You might have diabetes.')
else:
    print('You probably do not have diabetes.')
