import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
import random



def tokenize_input(input_file: str, vocab_size: int, sequence_length: int, batch_size: int):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    content_list = [line.strip() for line in lines]
    # for _ in range(5):
    #     print(random.choice(content_list))

    random.shuffle(content_list)
    val_size = test_size = int(0.15 * len(content_list))
    train_size = int(0.70 * len(content_list))
    
    trainlist = content_list[:train_size]
    vallist = content_list[train_size:val_size+train_size]
    testlist = content_list[train_size+val_size:]

    vectorization = keras.layers.TextVectorization(max_tokens = vocab_size, output_mode = "int", output_sequence_length = sequence_length)

    vectorization.adapt(trainlist)

    def format_dataset(text, text_2):
        text = vectorization(text)
        text_2 = vectorization(text_2)
        return (
                {
                    "encoder_inputs": text,
                    "decoder_inputs": text_2[:, :-1]
                },
                text_2[:, 1:],
                )
    trainset = tf.data.Dataset.from_tensor_slices((trainlist,trainlist))
    trainset = trainset.batch(batch_size)
    trainset = trainset.map(format_dataset)
    # trainset = trainset.cache().shuffle(2048).prefetch(16)

    valset = tf.data.Dataset.from_tensor_slices((vallist,vallist))
    valset = valset.batch(batch_size)
    valset = valset.map(format_dataset)
    # valset = valset.cache().shuffle(2048).prefetch(16)

    testset = tf.data.Dataset.from_tensor_slices((testlist,testlist))
    testset = testset.batch(batch_size)
    testset = testset.map(format_dataset)
    # testset = testset.cache().shuffle(2048).prefetch(16)

    return trainset, valset, testset



#for inputs, targets in trainset.take(1):
#    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
#    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
#    print(f"targets.shape: {targets.shape}")

