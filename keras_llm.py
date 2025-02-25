import os
import tensorflow as tf
import keras
import numpy as np
import json
import collections
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tokenizers
import transformers
from tokenizers.normalizers import NFKC
import pandas as pd

from keras_tokenizer import tokenize_input

os.environ["KERAS_BACKEND"] = "tensorflow"

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.dense_proj_1 = keras.layers.Dense(dense_dim, activation = "relu")
        self.dense_proj_2 = keras.layers.Dense(embed_dim)
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask = None):
        if mask is not None:
            padding_mask = keras.ops.cast(mask[:,None,:], dtype="int32")
        else:
            padding_mask = None
        attention_output = self.attention(query = inputs, value = inputs, key = inputs, attention_mask = padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj_1(proj_input)
        proj_output = self.dense_proj_2(proj_output)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "embed_dim": self.embed_dim,
                    "dense_dim": self.dense_dim,
                    "num_heads": self.num_heads,
                }
            )
        return config

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(input_dim = vocab_size, output_dim = embed_dim)
        self.position_embeddings = keras.layers.Embedding(input_dim = sequence_length, output_dim = embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = keras.ops.shape(inputs)[-1]
        positions = keras.ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask = None):
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "sequence_length": self.sequence_length,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                }
            )
        return config

class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.attention_2 = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim) # possibile punto di "ripetizione" per rendere questo un vero layer e non praticamente un modello
        self.dense_proj_1 = keras.layers.Dense(dense_dim, activation = "relu")
        self.dense_proj_2 = keras.layers.Dense(embed_dim)

        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        
        self.supports_masking = True

    def call(self, inputs, mask = None):
        inputs, encoder_outputs = inputs
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is None:
            inputs_padding_mask, encoder_outputs_padding_mask = None, None
        else:
            inputs_padding_mask, encoder_outputs_padding_mask = mask

        attention_output_1 = self.attention_1(query = inputs,value = inputs,key = inputs,attention_mask = causal_mask, query_mask = inputs_padding_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(query = out_1, value = encoder_outputs, key = encoder_outputs, query_mask = inputs_padding_mask, key_mask = encoder_outputs_padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj_1(out_2)
        proj_output = self.dense_proj_2(proj_output)

        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = keras.ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = keras.ops.arange(sequence_length)[:, None]
        j = keras.ops.arange(sequence_length)
        mask = keras.ops.cast(i >= j, dtype="int32")
        mask = keras.ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = keras.ops.concatenate([keras.ops.expand_dims(batch_size, -1), keras.ops.convert_to_tensor([1,1])], axis=0)

        return keras.ops.tile(mask, mult)


    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "embed_dim": self.embed_dim,
                    "dense_dim": self.dense_dim,
                    "num_heads": self.num_heads
                }
            )
        return config

def make_dataset(texts):
    texts = list(texts)
    dataset = tf.data.Dataset.from_tensor_slices((texts, texts))
    dataset = dataset.batch(batch_size)
    return dataset.cache().shuffle(2048).prefetch(16)



vocab_size = 8000
sequence_length = 100

embed_dim = 256
dense_dim = 64
num_heads = 12

batch_size = 1024 

# inp = keras.layers.Input(shape=(None,), dtype="int64", name="encoder_input")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inp)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(inp)

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)([x, encoder_outputs])
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

model = keras.Model(
    {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
    decoder_outputs,
    name="transformer",
)


#model = keras.models.Model(inp,x)
model.summary()

trainset, valset, testset = tokenize_input("reddit_content.txt",vocab_size, sequence_length, batch_size)

epochs = 3  # This should be at least 30 for convergence

model.compile(
    "rmsprop",
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"],
)
for inputs, targets in trainset.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

model.fit(trainset, epochs=epochs, validation_data = valset)

model.save("reddit_model.keras")


































