import random
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense

class DenseModel:
    def __init__(self, n_hours=10, dense_config=None):
        if dense_config is None:
            dense_config = [10, 50, 100, 100, 50, 20, 10, 1]

        input = Input(shape=(n_hours,))

        x = input

        for num_neurons in dense_config:
            x = Dense(num_neurons)(x)

        output = x

        self.model = Model(inputs=[input], outputs=[output])

    def train(self, data, iteration_amount=100, batch_size=32):
        if not self.model.compiled:
            self.model.compile(optimizer="sgd", loss="mse")

        for i in range(iteration_amount):
            batch = random.sample(data, batch_size)
            batch_x, batch_y = zip(*batch)

            arr_x = np.array(batch_x)
            arr_y = np.array(batch_y)

            self.model.fit(arr_x, arr_y)


if __name__ == "__main__":
    model = DenseModel()

    test_data = [(np.random.sample(10), np.random.random()) for _ in range(200)]  # [(10 timmar, värdet nästa timme)]
    model.train(test_data)
