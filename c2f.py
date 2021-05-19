# перевод из Цельсия в Фаренгейты
# F = C*1,8+32

import keras as k
import numpy as np
import matplotlib.pyplot as plt

c = np.array([-40, -10, 0,   8, 15, 22, 38])
f = np.array([-40,  14, 32, 46, 59, 72, 100])

model = k.Sequential()
model.add(k.layers.Dense(units=1,input_shape=(1,),activation='linear'))
model.compile(loss='mean_squared_error', optimizer=k.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=500, verbose=0)

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

print(model.predict([100]))
print(model.get_weights())
