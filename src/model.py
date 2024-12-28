import tensorflow as tf

def build_model(input_size,ouptut_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8,input_shape=(input_size,),activation='relu'),
        tf.keras.layers.Dense(8,activation='relu'),
        tf.keras.layers.Dense(ouptut_size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

