import keras as k
import numpy as np

input_data = np.array( [0.3, 0.7, 0.9]) # вероятность дождя
output_data = np.array([0.5, 0.9, 1.0]) # вероятность того, что понадобится зонт

model = k.Sequential()
model.add(k.layers.Dense(units=1, activation="linear"))
model.compile(loss="mse", optimizer="sgd")
fit_results = model.fit(x=input_data,y=output_data, epochs=200)

predicted = model.predict([0.5])
print(predicted)

