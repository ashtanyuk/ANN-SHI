import numpy as np
import keras as k
import matplotlib.pyplot as plt

input_data = np.loadtxt("input.csv")
output_data = np.loadtxt("output.csv")
print(input_data.shape)
print(output_data.shape)

output_data_ready = np.true_divide(output_data,100)
#print(output_data_ready)

model = k.Sequential()
model.add(k.layers.Dense(40, activation="relu", input_shape=(31,), kernel_initializer = 'he_uniform'))
model.add(k.layers.Dense(units=20, activation="relu"))
model.add(k.layers.Dense(units=20, activation="relu"))
model.add(k.layers.Dense(units=2, activation="sigmoid"))

print(model.inputs)
print(model.outputs)
print(model.summary())

#quit(0)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
fit_results = model.fit(x=input_data, y=output_data_ready, batch_size = 10, epochs=1000) #, validation_split=0.2, verbose = 1)

plt.plot(fit_results.history['loss'])
plt.grid(True)
plt.show()
plt.plot(fit_results.history['accuracy'])
plt.grid(True)
plt.show()

pred_data = np.array([[0,0,0,1,0,2,1,1,0,0,2,2,0,0,1,1,0,2,0,0,0,2,0,0,0,1,0,0,1,1,0]])

predicted_test = model.predict(input_data)

print(predicted_test)