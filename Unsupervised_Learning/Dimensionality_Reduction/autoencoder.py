#Requires: pip install tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_train = X_train.reshape((-1, 784))

# Encoder
input_layer = Input(shape=(784,))
encoded = Dense(32, activation="relu")(input_layer)

# Decoder
decoded = Dense(784, activation="sigmoid")(encoded)

# Autoencoder
model = Model(input_layer, decoded)

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train
model.fit(
    X_train,
    X_train,
    epochs=5,
    batch_size=256,
    verbose=1
)
