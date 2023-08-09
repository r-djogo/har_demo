import os
from utils.layers import FCCaps, Length, PrimaryCaps, Mask
from utils.tools import *

# repetitions
iters = 100

# load the model from file
capsnet = tf.keras.models.load_model("./model/capshar.hdf5",
                                    custom_objects={"PrimaryCaps": PrimaryCaps, "marginLoss": marginLoss,
                                                    "FCCaps": FCCaps, "Length": Length, "Mask": Mask})

# start timer
t_start = time.time()

data = np.random.random(size=[400,52])

for i in range(iters):
    # expand input dims
    data_temp = np.expand_dims(data, axis=0)
    data_temp = np.expand_dims(data_temp, axis=3)
    # normalize data
    data_temp = (data_temp - np.mean(data_temp)) / np.std(data_temp)
    # run inference
    activity_caps, pred = capsnet.predict(data_temp, verbose=0)

t_end = time.time() - t_start
print('Total time = ' ,t_end)
print('Time per iference = ', t_end/iters)
print('Inferences per second = ', 1.0/(t_end/iters))
