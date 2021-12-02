from __future__ import print_function

import csv
import glob
import random
import sys
import time

import tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy

# batch_size = 64  # Batch size for training.
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

epochs = 300  # Number of epochs to train for.
# latent_dim = 256  # Latent dimensionality of the encoding space.
latent_dims = [256, 256, 256]
# Path to the data txt file on disk.
# data_path = 'fra.txt'

# This should be consisted
# numpy.random.seed(5)
# Params
# hyper-parameter
sequence_size = 28
target_size = 7
input_size = sequence_size
total_size = sequence_size + target_size
# epoch = 300
batch_size = 100

# Load data
# File path
# sequence_file_path_2019 = '../00.data_processing/processed_csv_file_msp_padding/2019/'+str(sequence_size)+'sequence/*.csv'
# sequence_file_path_2020 = '../00.data_processing/processed_csv_file_msp_padding/2020/'+str(sequence_size)+'sequence/*.csv'
sequence_file_path_2019 = './00.data_processing/processed_csv_file_ed/2019/' + str(target_size) + "sequence/" + str(
    sequence_size) + 'sequence/*.csv'
# sequence_file_path_2020 = '../../00.data_processing/processed_csv_file_msp/2020/'+str(sequence_size)+'sequence/*.csv'
# It represent training and testing dataset percentage n : (100-n)
dataset_division_percentage = 70
# Dataset Count
total_dataset_count = 0
training_dataset_count = 0
test_dataset_count = 0
# Load data
total_dataset = []
for csv_file in glob.glob(sequence_file_path_2019):
    file = open(csv_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(file)
    for line in csv_reader:
        # line = list(map(int, line))
        total_dataset.append(line[:])
#
# for csv_file in glob.glob(sequence_file_path_2020):
#     file = open(csv_file, 'r', encoding='utf-8')
#     csv_reader = csv.reader(file)
#     for line in csv_reader:
#         # line = list(map(int, line))
#         total_dataset.append(line[:])

random.shuffle(total_dataset)
total_dataset_count = len(total_dataset)
print("total dataset : ", len(total_dataset))
test_dataset = total_dataset[0:int(len(total_dataset) * (100 - dataset_division_percentage) / 100)]
test_dataset_count = len(test_dataset)
training_dataset = total_dataset[test_dataset_count:len(total_dataset)]
training_dataset_count = len(training_dataset)
print("train dataset : ", training_dataset_count)
print("test dataset : ", test_dataset_count)

# # Vectorize the data.
# input_characters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '\n', '\t'}
# target_characters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '\n', '\t'}

input_texts = []
target_texts = []
for seq in total_dataset:
    # print(seq)
    input_texts.append(seq[0:input_size])
    # formatting target_text
    temp = seq[input_size:sequence_size + target_size]
    temp.insert(0, '\t')
    temp.append('\n')
    target_texts.append(temp)
    # print(input_texts[-1])
    # print(target_texts[-1])

input_characters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'}
target_characters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'x', '\n', '\t'}

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])  # 8
max_decoder_seq_length = max([len(txt) for txt in target_texts])  # 3

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

start_train = time.time()
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = numpy.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = numpy.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = numpy.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # print(t, char)
        encoder_input_data[i, t, input_token_index[char]] = 1.
    # encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        # print(target_text)
        # print(i, t, target_token_index[char], char)
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            # print(i, t - 1, target_token_index[char], char)

    # decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    # decoder_target_data[i, t:, target_token_index[' ']] = 1.

# decoder_input \t + characters + \n
# decoder_target characters + \n

# check the reference and modifying
# make single layer encoder-decoder model to stacked
# https://stackoverflow.com/questions/50915634/multilayer-seq2seq-model-with-lstm-in-keras

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))

outputs = encoder_inputs
encoder_states = []
for j in range(len(latent_dims))[::-1]:
    outputs, h, c = LSTM(latent_dims[j], return_state=True, return_sequences=bool(j))(outputs)
    encoder_states += [h, c]

# encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# # We discard `encoder_outputs` and only keep the states.
# encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))

outputs = decoder_inputs
output_layers = []

for j in range(len(latent_dims)):
    output_layers.append(LSTM(latent_dims[len(latent_dims) - j - 1], return_sequences=True, return_state=True))
    outputs, dh, dc = output_layers[-1](outputs, initial_state=encoder_states[2 * j:2 * (j + 1)])

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model = multi_gpu_model(model, gpus=2)
# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', tensorflow.keras.metrics.FalsePositives()])
model.summary()
model.fit([encoder_input_data[test_dataset_count:total_dataset_count], decoder_input_data[test_dataset_count:total_dataset_count]],
          decoder_target_data[test_dataset_count:total_dataset_count], batch_size=batch_size, epochs=epochs, verbose=2)
# Save model
# model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
# encoder_model = Model(encoder_inputs, encoder_states)
#
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)
encoder_model = Model(encoder_inputs, encoder_states)

d_outputs = decoder_inputs
decoder_states_inputs = []
decoder_states = []
for j in range(len(latent_dims))[::-1]:
    current_state_inputs = [Input(shape=(latent_dims[j],)) for _ in range(2)]
    temp = output_layers[len(latent_dims) - j - 1](d_outputs, initial_state=current_state_inputs)
    d_outputs, cur_states = temp[0], temp[1:]
    decoder_states += cur_states
    decoder_states_inputs += current_state_inputs
decoder_outputs = decoder_dense(d_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

spend_time_train = time.time() - start_train

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
#
#     # Generate empty target sequence of length 1.
#     target_seq = numpy.zeros((1, 1, num_decoder_tokens))
#     # print("test")
#     # print(target_seq)
#     # print("test")
#
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, target_token_index['\t']] = 1.
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [target_seq] + states_value)
#
#         # Sample a token
#         sampled_token_index = numpy.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]
#         decoded_sentence += sampled_char
#
#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_char == '\n' or
#                 len(decoded_sentence) > max_decoder_seq_length):
#             stop_condition = True
#
#         # Update the target sequence (of length 1).
#         target_seq = numpy.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.
#
#         # Update states
#         states_value = [h, c]
#
#     return decoded_sentence
spend_time_test = 0


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    global spend_time_test
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = numpy.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []  # Creating a list then using "".join() is usually much faster for string creation
    s_time = 0
    while not stop_condition:
        start_test = time.time()
        to_split = decoder_model.predict([target_seq] + states_value)
        spend_time_test += (time.time() - start_test)
        output_tokens, states_value = to_split[0], to_split[1:]

        # Sample a token
        sampled_token_index = numpy.argmax(output_tokens[0, 0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return "".join(decoded_sentence)


# for seq_index in range(8441):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     print(total_dataset[seq_index])

# fir_cnt = 0
# sec_cnt = 0
# thr_cnt = 0
# start_test = time.time()
target_list = ["" for i in range(test_dataset_count)]
for i in range(test_dataset_count):
    for t in range(target_size, 0, -1):
        target_list[i] = target_list[i] + total_dataset[i][total_size - t]
    target_list[i] = target_list[i] + '\n'

decoded_seq_list = []
for seq_index in range(0, test_dataset_count):
    sys.stdout.write('\r' + "processing..." + str(seq_index) + "/" + str(test_dataset_count))
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    # print(input_seq)
    decoded_sentence = decode_sequence(input_seq)
    # print(decoded_sentence)
    # print('-')
    # print(total_dataset[seq_index])
    # print('Input sequence', input_texts[seq_index])
    # print('Decoded sequence:', decoded_sentence)
    decoded_seq_list.append(decoded_sentence)
sys.stdout.write('\r' + "processing...done..." + str(test_dataset_count))
print("\n")
# spend_time_test = time.time() - start_test

print("time_cost | train : " + str(spend_time_train))
print("time_cost | test : " + str(spend_time_test))
print("time_cost | test_individual " + str(spend_time_test / test_dataset_count))

TFPN_matrix = [[0 for j in range(12)] for i in range(12)]
print(TFPN_matrix)

seq_cnt = 0
cnt_list = [0 for i in range(target_size)]
for i in range(0, test_dataset_count):
    # print(target_list[i] + decoded_seq_list[i])
    if target_list[i] == decoded_seq_list[i]:
        seq_cnt += 1
    else:
        for t in range(0, target_size):
            if target_list[i][t] == decoded_seq_list[i][t]:
                cnt_list[t] += 1
            else:
                break

for i in range(0, test_dataset_count):
    # print(target_list[i] + decoded_seq_list[i])
    for t in range(0, target_size):
        print(t)
        print(ord(target_list[i][t]) - 97)
        print(ord(decoded_seq_list[i][t]) - 97)
        TFPN_matrix[ord(target_list[i][t]) - 97][ord(decoded_seq_list[i][t]) - 97] += 1

#     # if target[0] == decoded_sentence[0]:
#     #     fir_cnt += 1
#     #     if target[1] == decoded_sentence[1]:
#     #         sec_cnt += 1
#     #         if target[2] == decoded_sentence[2]:
#     #             thr_cnt += 1
# # fir_cnt += seq_cnt
# # sec_cnt += seq_cnt
# # thr_cnt += seq_cnt
#
# print(seq_cnt, "/", str(len(total_dataset) - int(len(total_dataset) * 0.7)))
# print(str(seq_cnt / (len(total_dataset) - int(len(total_dataset) * 0.7)) * 100))
#
for i in range(0, target_size):
    print(str((cnt_list[i] + seq_cnt) / test_dataset_count * 100))


def TFPN_calculation(matrix):
    classes = len(matrix)
    sum_ = 0
    for list in matrix:
        sum_ += sum(list)
    result = [[0 for x in range(4)] for result in range(classes)]  # TP, TN, FP, FN in order
    for i in range(classes):
        result[i][0] = matrix[i][i]
        result[i][2] = sum(numpy.array(matrix).T[i]) - matrix[i][i]
        result[i][3] = sum(matrix[i]) - matrix[i][i]
        result[i][1] = sum_ - result[i][0] - result[i][2] - result[i][3]
        print(result[i])
        print(sum(result[i]))
        # print(matrix[0:classes][i])


for i in range(12):
    print(TFPN_matrix[i])

TFPN_calculation(TFPN_matrix)

# print(str(fir_cnt / (len(total_dataset) - int(len(total_dataset) * 0.7)) * 100))
# print(str(sec_cnt / (len(total_dataset) - int(len(total_dataset) * 0.7)) * 100))
# print(str(thr_cnt / (len(total_dataset) - int(len(total_dataset) * 0.7)) * 100))

# print(str((cnt_list[0] + seq_cnt) / test_dataset_count * 100))
# print(str((cnt_list[1] + seq_cnt) / test_dataset_count * 100))
# print(str((cnt_list[2] + seq_cnt) / test_dataset_count * 100))
