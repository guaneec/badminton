from pathlib import Path

def train_model(path):
    import tensorflow as tf
    from tensorflow.keras import layers

    # fix "No algorithm worked" error
    # https://github.com/tensorflow/tensorflow/issues/43174
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    train_path = str(Path(path) / "train")
    test_path = str(Path(path) / "test")

    image_size=(32,32)
    batch_size=64

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path, image_size=image_size, batch_size=batch_size
    )


    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path, image_size=image_size, batch_size=batch_size
    )

    model = tf.keras.Sequential([
        layers.Input((32, 32, 3)),
        layers.experimental.preprocessing.Rescaling(1./255),

        layers.Conv2D(16, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),


        layers.Conv2D(16, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dropout(0.5),    
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[
        tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), 'accuracy'
    ])
    model.fit(train_ds, validation_data=test_ds, epochs=20)
    return model