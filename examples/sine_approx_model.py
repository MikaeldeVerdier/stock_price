import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense

def get_data(n=1000):
    x = np.linspace(0, 2 * np.pi, n)[:, None]
    y = np.sin(x)

    return x, y


def build_model():
    input = Input(shape=(1,))

    x = Dense(10, activation="relu")(input)  # x value
    x = Dense(100, activation="relu")(x)
    x = Dense(100, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    output = Dense(1)(x)  # Predicted sine value

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer=SGD(learning_rate=1e-2), loss="mse")

    model.summary()

    return model


def train_model(model, data_x, data_y):
    metrics_history = model.fit(data_x, data_y, epochs=200)
    loss_vals = metrics_history.history["loss"]

    return loss_vals


def plot_model(model, test_inputs, test_outputs, loss_vals):
    model_outputs = model.predict(test_inputs)

    fig, axs = plt.subplots(2)

    axs[0].plot(test_inputs[:, 0], test_outputs[:, 0], label="Real values")
    axs[0].plot(test_inputs[:, 0], model_outputs[:, 0], label="Precited values")

    axs[1].plot(loss_vals, label="Loss")

    plt.suptitle("Sine Approximation Model")
    plt.legend()

    plt.savefig("sine_approx_model.png")


if __name__ == "__main__":
    model = build_model()
    data_x, data_y = get_data()
    loss_vals = train_model(model, data_x, data_y)

    plot_model(model, data_x, data_y, loss_vals)
