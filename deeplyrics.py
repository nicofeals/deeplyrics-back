import pickle
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow import expand_dims, squeeze
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


class deeplyrics():

    def __init__(self):
        self.model = tf.keras.models.load_model('deeplyrics.h5')

    def generate_text(self, idx2char, char2idx, input):
        # Number of characters to generate
        num_generate = 500

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in input]
        input_eval = expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model.predict(input_eval)
            # remove the batch dimension
            predictions = squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # Pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (input + ''.join(text_generated))
  
    def predict(self, input):
        file = "merged_lyrics_metalcore.txt"
        file_augmented = "merged_lyrics_metalcore_augmented.txt"
        
        lyrics = open(file, "r").read() + open(file_augmented, "r").read()
        chars = sorted(list(set(lyrics)))

        char2idx = dict((c,i) for i, c in enumerate(chars))
        idx2char = np.array(chars)
        
        return self.generate_text(idx2char, char2idx, input)