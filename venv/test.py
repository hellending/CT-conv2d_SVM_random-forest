import numpy as np
x_train = [1,2,3,4,5,6,7,8]
y_train = [11,12,13,14,15,16,17,18]
for i in range(0,10):
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    print(x_train)
    print(y_train)