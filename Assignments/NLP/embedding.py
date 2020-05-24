import tensorflow
import tensorflow_datasets as tfds
import numpy as np
import os
import csv
from tensorflow.keras.preprocessing import sequence
import h5py

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
traindata = imdb['train']
testdata = imdb['test']

training_sentences = []
training_labels = []

test_sentences = []
test_labels = []

for x,y in traindata:
    training_sentences.append(str(x.numpy()))
    training_labels.append(y.numpy())

for s,l in testdata:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
test_labels_final = np.array(test_labels)
vocab_size=10000
vector_dim=300

tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(training_sentences)
words = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = tensorflow.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=120, truncating = "post")


word_target_final = []
word_context_final = []
couples_final = []
labels_final = []

for i in range(1, int(len(padded)/100)):
    sampling_table = sequence.make_sampling_table(vocab_size)
    couples, labels = sequence.skipgrams(padded[i], vocab_size, window_size=2)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    labels_final.append(labels)
    word_target_final.append(word_target)
    word_context_final.append(word_context)

input_target = tensorflow.keras.layers.Input((1,))
input_context = tensorflow.keras.layers.Input((1,))

embedding = tensorflow.keras.layers.Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = tensorflow.keras.layers.Reshape((vector_dim,1))(target)
context = embedding(input_context)
context = tensorflow.keras.layers.Reshape((vector_dim, 1))(context)

# similarity = tensorflow.keras.layers.Dot([target, context], normalize=True)

# now perform the dot product operation to get a similarity measure
dot_product = tensorflow.keras.layers.dot([target, context], axes=1)
dot_product = tensorflow.keras.layers.Reshape((1,))(dot_product)

output = tensorflow.keras.layers.Dense(1, activation='sigmoid')(dot_product)

model = tensorflow.keras.models.Model([input_target, input_context], output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

epochs = 15000

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
lentot=0
for x in range(0, len(labels_final)-1):
    lentot+=(len(labels_final[x])-1)

for cnt in range(epochs):
    i = np.random.randint(0, lentot)
    j = 0
    while(i>len(labels_final[j])-1):
        i = i - len(labels_final[j]) + 1
        j=j+1
    arr_1[0,] = word_target_final[j][i]
    arr_2[0,] = word_context_final[j][i]
    arr_3[0,] = labels_final[j][i]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if loss<0.005:
        break

vectors = embedding.get_weights()

size = 10000
max_length = 120

tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(num_words = size, oov_token = "<OOV>")
tokenizer.fit_on_texts(training_sentences)
words = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = tensorflow.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, truncating = "post")

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = tensorflow.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, truncating = "post")

Embedding = tensorflow.keras.layers.Embedding(size, 300, weights=vectors, trainable=False)

model = tensorflow.keras.Sequential([
    Embedding,
    tensorflow.keras.layers.GlobalAveragePooling1D(),
    tensorflow.keras.layers.Dense(6, activation='relu'),
    tensorflow.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(test_padded, test_labels_final))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

