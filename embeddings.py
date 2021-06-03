# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

# Create a sample of reviews
reviews = ['nice food',
           'amazing restuarant',
           'too good',
           'just loved it',
           'will go again',
           'horrible food',
           'never go there',
           'poor service',
           'poor quality',
           'needs improvement']

# Create sentiment labels for the reviews
sentiment = np.array([1,1,1,1,1,0,0,0,0,0])

VOCAB_SIZE = 30
MAX_LEN = 3
EMBEDDING_VECTOR_SIZE = 5

encoded_reviews = [one_hot(d, VOCAB_SIZE) for d in reviews]
print(encoded_reviews)

# Pad the reviews with max length length 3
padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(encoded_reviews,
                                                            maxlen=MAX_LEN,
                                                            padding='post')

print(padded_reviews)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,
                                EMBEDDING_VECTOR_SIZE,
                                input_length=MAX_LEN,
                                name='embedding'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

X = padded_reviews
y = sentiment

history = model.fit(X, y, epochs=25)

# save the weights of the model's embeddinglayer in the weights variable
weights = model.get_layer('embedding').get_weights()[0]

# Show the weights for different encoded inputs
print(weights[0])
print(weights[1])
