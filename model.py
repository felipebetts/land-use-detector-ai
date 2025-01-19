from keras import layers, Input, Model

from constants import TILE_SIZE

def unet_model(input_shape=(TILE_SIZE, TILE_SIZE, 3), num_classes=7):
    """
    U-Net para segmentação semantica.
    """
    inputs = Input(shape=input_shape)

    # Encoder (downsampling)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    c1 = layers.Dropout(0.1)(c1)  # Dropout adicionado aqui
    p1 = layers.MaxPooling2D((2,2))(c1)
    
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    c2 = layers.Dropout(0.1)(c2)  # Dropout adicionado aqui
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Botleneck
    bn = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    bn = layers.Conv2D(256, (3,3), activation='relu', padding='same')(bn)
    bn = layers.Dropout(0.1)(bn)  # Dropout adicionado aqui
    
    # Decoder (upsampling)
    u1 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(bn)
    concat1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(concat1)
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    c3 = layers.Dropout(0.1)(c3)  # Dropout adicionado aqui

    u2 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c3)
    concat2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(concat2)
    c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c4)
    c4 = layers.Dropout(0.1)(c4)  # Dropout adicionado aqui

    # Camada final de previsão de classes (pixel-wise)
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c4)

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

