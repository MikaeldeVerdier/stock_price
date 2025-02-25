import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense

from get_data import get_data

class DenseModel:
    def __init__(self, scaling_factor=1e-5, n_hours=10, dense_config=None):
        self.scaling_factor = scaling_factor

        if dense_config is None:
            dense_config = [10, 50, 100, 100, 50, 20, 10, 1]

        input = Input(shape=(n_hours,))

        x = input  # * scaling_factor

        for num_neurons in dense_config:
            x = Dense(num_neurons)(x)

        output = x  # / scaling_factor

        self.model = Model(inputs=[input], outputs=[output])

    def prepare_data(self, data):
        data_x, data_y = zip(*data)

        arr_x = np.array(data_x) * self.scaling_factor
        arr_y = np.array(data_y) * self.scaling_factor

        return arr_x, arr_y

    def train(self, train_data, validation_data=None, iteration_amount=200, batch_size=32, learning_rate=1e-4):
        if not self.model.compiled:
            self.model.compile(optimizer=SGD(learning_rate=learning_rate), loss="mse")

        train_data_x, train_data_y = self.prepare_data(train_data)
        if validation_data is not None:
            validation_data = self.prepare_data(validation_data)
        else:
            validation_data = None

        history = self.model.fit(train_data_x, train_data_y, batch_size=batch_size, epochs=iteration_amount, steps_per_epoch=1, validation_data=validation_data)

        losses = history.history["loss"]
        val_losses = history.history.get("val_loss", None)

        """
        losses = []
        for i in range(iteration_amount):
            batch = random.sample(data, batch_size)
            batch_x, batch_y = zip(*batch)

            scaling_factor = 1e5
            arr_x = np.array(batch_x) / scaling_factor
            arr_y = np.array(batch_y) / scaling_factor

            history = self.model.fit(arr_x, arr_y)
            losses += history.history["loss"]
        """

        self.plot_model(losses, val_losses)

    def plot_model(self, losses, val_losses=None):
        plt.plot(losses, label="Loss (MSE)")
        if val_losses is not None:
            plt.plot(val_losses, label="Validation loss (MSE)")

        plt.suptitle("Loss during training")
        plt.legend()

        plt.savefig("loss.png")

    def test(self, data):
        data_x, data_y = self.prepare_data(data)
        results = self.model.predict(data_x)

        pos_mask = data_x[:, -1] < results[:, 0]
        bought_prices = data_x[pos_mask, -1] / self.scaling_factor
        sold_prices = data_y[pos_mask] / self.scaling_factor

        winnings = (bought_prices - sold_prices) / sold_prices
        avg_winning = np.mean(winnings)

        print(f"Average winning was {avg_winning * 100:.2f}%!")


if __name__ == "__main__":
    model = DenseModel()

    train_data = get_data("btc_20240101_20250101_1h.csv")
    validation_data = get_data("btc_20230101_20240101_1h.csv")
    model.train(train_data, validation_data)

    test_data = get_data("btc_20220101_20230101_1h.csv")
    model.test(test_data)
