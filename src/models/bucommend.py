import tensorflow as tf
import keras
import numpy as np

MAX_SEQUENCE_LENGTH = 100

class Bucommend(keras.Model):
    def __init__(self, vocab_size, embedding_dim, top_n, *args, **kwargs):
        super(Bucommend, self).__init__(*args, **kwargs)
        
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding_layer")
        self.flatten = keras.layers.Flatten(name="flatten")
        self.dense1 = keras.layers.Dense(128, activation="relu", name="dense1")
        self.dense2 = keras.layers.Dense(64, activation="relu", name="dense2")
        self.dense3 = keras.layers.Dense(32, activation="relu", name="dense3")

        self.output_layer = keras.layers.Dense(embedding_dim, activation='linear')

        self.top_n = top_n


    def call(self, inputs):
        embedding = self.embedding(inputs[0])
        x = self.flatten(embedding)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return self.output_layer(x)
    
    def get_top_n_similar(self, query_embedding, restaurant_embeddings, top_n):
        similarity = tf.keras.losses.cosine_similarity(query_embedding, restaurant_embeddings)
        
        top_n_indices = np.argsort(similarity.numpy())[:top_n]
        return top_n_indices
        
