from keras import layers, Input, Model, utils

from constants import TILE_SIZE

def unet_model(input_shape=(TILE_SIZE, TILE_SIZE, 3), num_classes=7):
    """
    U-Net completa para segmentação semântica.
    """
    inputs = Input(shape=input_shape)

    # Encoder (4 blocos com pooling)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Botleneck
    bn = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
    bn = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(bn)

    # Decoder (4 blocos com transpose convolutions)
    u1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(bn)
    concat1 = layers.concatenate([u1, c4])
    c5 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(concat1)
    c5 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c5)

    u2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    concat2 = layers.concatenate([u2, c3])
    c6 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(concat2)
    c6 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c6)

    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    concat3 = layers.concatenate([u3, c2])
    c7 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(concat3)
    c7 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c7)

    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    concat4 = layers.concatenate([u4, c1])
    c8 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(concat4)
    c8 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c8)

    # Camada final
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(c8)

    model = Model(inputs, outputs, name="U-Net")
    return model

def get_compiled_model():
    num_classes = 7  
    model = unet_model(input_shape=(TILE_SIZE, TILE_SIZE, 3), num_classes=num_classes)
    model.summary()

    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = unet_model()
    utils.plot_model(model, show_shapes=True)