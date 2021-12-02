import random
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import CuDNNLSTM
from keras.utils import np_utils
import numpy

import csv
import glob

import matplotlib.pyplot as plot

# reference
# https://tykimos.github.io/2017/04/09/RNN_Layer_Talk/


# csv file path
sequence_file_path = '../00.data_processing/processed_csv_file_ssp/15sequence/*.csv'
# This should be consisted
numpy.random.seed(5)

# Params
# hyper-parameter
sequence_size = 15
epoch = 300
batch_size = 100

# Load data
# It represent training and testing dataset percentage n : (100-n)
dataset_division_percentage = 70
# Dataset Count
total_dataset_count = 0
training_dataset_count = 0
test_dataset_count = 0
# Load data
total_dataset = []
for csv_file in glob.glob(sequence_file_path):
    file = open(csv_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(file)
    for line in csv_reader:
        line = list(map(int, line))
        total_dataset.append(line[:])

random.shuffle(total_dataset)
total_dataset = numpy.array(total_dataset)
# Divide train and test
total_dataset_count = len(total_dataset)
training_dataset_count = int(total_dataset_count * dataset_division_percentage / 100)
test_dataset_count = total_dataset_count - training_dataset_count
training_dataset_count = int(training_dataset_count / batch_size) * batch_size
test_dataset_count = int(test_dataset_count / batch_size) * batch_size
test_dataset = total_dataset[0:test_dataset_count]
train_dataset = total_dataset[test_dataset_count:training_dataset_count + test_dataset_count]
# train_dataset = total_dataset[0:training_dataset_count]
# test_dataset = total_dataset[training_dataset_count:training_dataset_count + test_dataset_count]

print('Total_dataset_count :: ' + str(total_dataset_count))
print('Train&Test Ratio :: ' + str(dataset_division_percentage) + ' : ' + str(100 - dataset_division_percentage))
print('Train_dataset_count :: ' + str(len(train_dataset)))
print('Test_dataset_count :: ' + str(len(test_dataset)))

# pre-processing
max_idx_value = 11  # add 1 to all data and make 12 then test // if origin code doesn't work
one_hot_vec_size = 12

train_x = train_dataset[0:training_dataset_count, 0:sequence_size]
train_y = train_dataset[0:training_dataset_count, sequence_size]
train_x = train_x / float(max_idx_value)  # normalization
train_x = numpy.reshape(train_x, (training_dataset_count, train_dataset.shape[1] - 1, 1))
train_y = np_utils.to_categorical(train_y, num_classes=one_hot_vec_size)

test_x = test_dataset[0:test_dataset_count, 0:sequence_size]
test_y = test_dataset[0:test_dataset_count, sequence_size]
test_x = test_x / float(max_idx_value)  # normalization
test_x = numpy.reshape(test_x, (test_dataset_count, test_dataset.shape[1] - 1, 1))
test_y = np_utils.to_categorical(test_y, num_classes=one_hot_vec_size)

# # ReLoad Balanced Training Dataset
# balanced_train_dataset=[]
# file = open('../00.data_processing/balanced_list.csv', 'r', encoding='utf-8')
# csv_reader = csv.reader(file)
# for line in csv_reader:
#     line = list(map(int, line))
#     balanced_train_dataset.append(line[:])
#
# balanced_train_dataset = numpy.array(balanced_train_dataset)
# training_dataset_count = int(len(balanced_train_dataset) / batch_size) * batch_size
#
# train_x = balanced_train_dataset[0:training_dataset_count, 0:sequence_size]
# train_y = balanced_train_dataset[0:training_dataset_count, sequence_size]
# train_x = train_x / float(max_idx_value)  # normalization
# train_x = numpy.reshape(train_x, (training_dataset_count, train_dataset.shape[1] - 1, 1))
# train_y = np_utils.to_categorical(train_y)
#
# print('Balanced Origin Train_dataset_count :: ' + str(len(balanced_train_dataset)))
# print('Balanced Train_dataset_count :: ' + str(training_dataset_count))

# model
model = Sequential()
# model.add(LSTM(128, batch_input_shape=(batch_size, sequence_size, 1), stateful=True))
# model.add(CuDNNLSTM(256, batch_input_shape=(batch_size, sequence_size, 1), stateful=True))
model.add(LSTM(256, return_sequences=True, batch_input_shape=(batch_size, sequence_size, 1), stateful=True))
model.add(LSTM(256, return_sequences=True, stateful=True))
model.add(LSTM(256, stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])

print(model.summary())

# Train
start = time.time()
hist = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epoch, batch_size=batch_size, verbose=2)
spend_time = time.time() - start
# hist = model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=2)

# for epoch_idx in range(epoch):
#     print('epochs : ' + str(epoch_idx))
#     model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False,
#               callbacks=[history])  # 50 is X.shape[0]
# model.reset_states()

# # Validation
scores = model.evaluate(test_x, test_y, batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Test
# print(numpy.argmax(model.predict(train_x, 189)))
i = 0
cnt = 0
# test_x = numpy.reshape(test_x, (test_dataset_count, sequence_size, 1))
for output in model.predict(test_x, batch_size):
    # print(str(numpy.argmax(output)) + " / " + str(numpy.argmax(test_y[i])))
    if numpy.argmax(test_y[i]) == numpy.argmax(output):
        # print(str(test_x[i])+str(numpy.argmax(output)))
        cnt += 1
    i += 1

# Results
print(str(cnt) + '/' + str(i) + ' :: ' + str(round(cnt / i, 6) * 100) + "%\n")

# Graph
# fig, loss_ax = plot.subplots()
# acc_ax = loss_ax.twinx()
#
# # Train
# train_loss_line = loss_ax.plot(range(epoch), history.losses, 'r', label='train loss')
# train_acc_line = acc_ax.plot(range(epoch), history.acces, 'b', label='train acc')
#
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
#
# lines = train_loss_line + train_acc_line
# labels = [l.get_label() for l in lines]
# plot.legend(lines, labels, loc='upper left')
# plot.show()

fig, loss_ax = plot.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='lower left')

plot.show()

# # logging
# log_file = open('../00.data_processing/experiment_logs/LSTM_mobility_prediction_keras.txt', 'a')
#
# log_file.write('\nExperiment <' + str(datetime.datetime.now()) + '>' + '\n')
# log_file.write('----------------------' + '\n')
# log_file.write('sequence_length : ' + str(sequence_size) + '\n')
# log_file.write('epoch : ' + str(epoch) + '\n')
# log_file.write('batch_size : ' + str(batch_size) + '\n')
# model.summary(print_fn=lambda x: log_file.write(x + '\n'))
# log_file.write('======================' + '\n')
# log_file.write('acc : ' + str(scores[1] * 100) + '\n')
# log_file.write('time : ' + str(spend_time) + '\n')
# log_file.write('======================\n')
# log_file.close()
