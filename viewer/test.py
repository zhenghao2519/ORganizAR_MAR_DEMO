import numpy as np

target_pos = [1,2,3]

target_pos_reshaped = np.reshape(target_pos, (1, 3))

print(target_pos_reshaped.shape)
