import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_data(data_dir="data", img_size=(224, 224), batch_size=32):
    """
    Loads microplastic images after H2O2 digestion and imaging.
    Applies light augmentation to simulate natural variation in:
    - particle orientation
    - lighting
    - background noise
    """

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Augmentation pipeline
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Apply augmentation only to training data
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
