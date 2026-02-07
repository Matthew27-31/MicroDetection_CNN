import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(img_size=(224, 224), num_classes=3):
    """
    CNN for microplastic classification using transfer learning.
    Matches the workflow described in the Methods section:
    - Images collected after H2O2 digestion
    - Consistent optical imaging setup
    - Classification by shape/color/size categories
    """

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False  # freeze pretrained layers

    inputs = layers.Input(shape=img_size + (3,))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model