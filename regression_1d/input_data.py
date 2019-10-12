import numpy as np


def generate_data(invert=False):
    amplitude = 0.3
    freq = 1
    t = np.linspace(0.0, 1.0, num=1000)

    y = amplitude * np.sin(2 * np.pi * freq * t)
    drift = 1.0 * t
    sequence_len = t.shape[0]

    noise = np.random.uniform(low=-0.1, high=0.1, size=sequence_len)

    y_hat = y + drift + noise

    t = t.astype(np.float32)
    y_hat = y_hat.astype(np.float32)

    if invert:
        t, y_hat = y_hat, t

    return np.expand_dims(t, -1), np.expand_dims(y_hat, -1)


def generate_heteroscedastic_line():
    t = np.linspace(0.0, 5.0, num=1000)
    drift = -1.0 * t

    noise = np.r_[
        0.01 * np.random.standard_normal((200,)),
        0.40 * np.random.standard_normal((400,)),
        0.10 * np.random.standard_normal((400,)),
    ]

    y_hat = 1 + drift + noise

    return np.expand_dims(t, -1).astype(np.float32), np.expand_dims(y_hat, -1).astype(np.float32)
