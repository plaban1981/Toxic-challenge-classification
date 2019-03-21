#Toxic comment challenge script
##concatenate of different scripts used for the toxic comment challenge
#these model configuration are motivated from the comments and kernel shared.
#total six different keras deeplearning models are used.these are trained on nvidia m40 gpu processor.
#'glove.840B.300d' embedding file is needed.
#and all the model parameter setting are done in the 'config.ini' file.
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply,GlobalAveragePooling1D, GlobalMaxPooling1D ,CuDNNGRU
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import configparser

SEED = 42  # always use a seed for randomized procedures
config=configparser.ConfigParser()
config.read('config.ini')

print('loading embeddings vectors')
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.split(' ')) for o in open('glove.6B//glove.6B.50d.txt',encoding="utf8"))
embeddings_index = dict(get_coefs(*o.split(' ')) for o in open('glove.840B.300d.txt'))


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

min_count=int(config.get('KerasModel', 'min_count'))	        #the minimum required word frequency
max_features=int(config.get('KerasModel', 'max_features'))	#max feature (vocubulary size)
maxlen=int(config.get('KerasModel', 'maxlen'))		        #maximum length for input
num_folds=int(config.get('KerasModel', 'num_folds'))	        #number of folds
batch_size=int(config.get('KerasModel', 'batch_size'))		#batch size
epochs=int(config.get('KerasModel', 'epochs'))	                #number of epochs
embed_size=int(config.get('KerasModel', 'embed_size')) 	        #embeddings dimension
train_file = config.get('FileRelated', 'train_file')            #training filename
test_file = config.get('FileRelated', 'test_file')              #test filename
out_folder = config.get('General', 'foldername')        #output filename
filename = ''

train = pd.read_csv(train_file).fillna("no comment")
test = pd.read_csv(test_file).fillna("no comment")

list_sentences_train = train["comment_text"].str.lower()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train[list_classes] = train[list_classes].astype(np.int8)
target = train[list_classes]

list_sentences_test = test["comment_text"]
trb_nan_idx = list_sentences_test[pd.isnull(list_sentences_test)].index.tolist()
list_sentences_test.loc[trb_nan_idx] = ' '
list_sentences_test = list_sentences_test.str.lower()

print('mean text len:',train["comment_text"].str.count('\S+').mean())
print('max text len:',train["comment_text"].str.count('\S+').max())

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
print('padding sequences')
X_train = train["comment_text"]
X_test = test["comment_text"]
X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
X_train = np.array(X_train)
X_test = np.array(X_test)

print('numerical variables')
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print('create embedding matrix')
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#1
def get_single_grucnn_model():
    filename = 'single_grucnn'
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x2 = Bidirectional(CuDNNGRU(80, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x2 = Conv1D(120, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x2)
    #x = Dropout(0.2)(x)
    max_pool = GlobalMaxPooling1D()(x2)
    avg_pool = GlobalAveragePooling1D()(x2)
    x = Concatenate()([avg_pool,max_pool])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename
#2
def get_two_gru_connect_model():
    filename = 'two_gru_connect'
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(100, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    #x = Dropout(0.2)(x)
    x = Bidirectional(CuDNNGRU(80, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    #x = Dropout(0.2)(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate()([avg_pool,max_pool])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename
#3
def get_two_grucnn_trainable_model():
    filename = 'two_grucnn_trainable'
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(CuDNNGRU(40, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x1 = Conv1D(60, kernel_size = 3, padding = "valid",activation = "relu",strides = 1)(x1)
    #x = Dropout(0.2)(x)
    x2 = Bidirectional(CuDNNGRU(80, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x2 = Conv1D(120, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x2)
    #x = Dropout(0.2)(x)
    max_pool1 = GlobalMaxPooling1D()(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    x = Concatenate()([avg_pool1,max_pool1, avg_pool2,max_pool2])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename
#4
def get_two_lstmcnn_model():
    filename = 'two_lstmcnn'
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(CuDNNLSTM(40, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x1 = Conv1D(60, kernel_size = 3, padding = "valid",activation = "relu",strides = 1)(x1)
    #x = Dropout(0.2)(x)
    x2 = Bidirectional(CuDNNLSTM(80, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x2 = Conv1D(120, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x2)
    #x = Dropout(0.2)(x)
    max_pool1 = GlobalMaxPooling1D()(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    x = Concatenate()([avg_pool1,max_pool1, avg_pool2,max_pool2])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename
#5
def get_three_grucnn_model():
    filename = 'three_grucnn'

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(CuDNNGRU(40, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x1 = Conv1D(60, kernel_size = 3, padding = "valid",activation = "relu",strides = 1)(x1)
    #x = Dropout(0.2)(x)
    x2 = Bidirectional(CuDNNGRU(60, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x2 = Conv1D(80, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x2)
    x3 = Bidirectional(CuDNNGRU(80, kernel_initializer='glorot_uniform', return_sequences=True))(x2)
    x3 = Conv1D(108, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x3)
    #x = Dropout(0.2)(x)
    max_pool1 = GlobalMaxPooling1D()(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    max_pool3 = GlobalMaxPooling1D()(x3)
    avg_pool3 = GlobalAveragePooling1D()(x3)
    x = Concatenate()([avg_pool1,max_pool1, avg_pool2,max_pool2,avg_pool3, max_pool3])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename
#6
def get_two_grucnn_model():
    filename = 'two_grucnn'

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.33)(x)
    x1 = Bidirectional(CuDNNGRU(40, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x1 = Conv1D(60, kernel_size = 3, padding = "valid",activation = "relu",strides = 1)(x1)
    #x = Dropout(0.2)(x)
    x2 = Bidirectional(CuDNNGRU(80, kernel_initializer='glorot_uniform', return_sequences=True))(x)
    x2 = Conv1D(120, kernel_size = 2, padding = "valid",activation = "relu",strides = 1)(x2)
    #x = Dropout(0.2)(x)
    max_pool1 = GlobalMaxPooling1D()(x1)
    avg_pool1 = GlobalAveragePooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    x = Concatenate()([avg_pool1,max_pool1, avg_pool2,max_pool2])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model,filename

print('start modeling')
early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 5)
scores = []
predict = np.zeros((test.shape[0],6))
oof_predict = np.zeros((train.shape[0],6))

print("model selection")
model_select=config.get('Model', 'model_select')	#model selection

kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
for train_index, test_index in kf.split(X_train):
    X_train1 , X_valid = X_train[train_index] , X_train[test_index]
    y_train, y_val = target.loc[train_index], target.loc[test_index]
    model,filename = locals()[model_select]()
    model.fit(X_train1, y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks = [early_stop])
    print('Predicting....')
    oof_predict[test_index] = model.predict(X_valid, batch_size=1024)
    #cv_score = roc_auc_score(y_test, oof_predict[test_index])
    #scores.append(cv_score)
    #print('score: ',cv_score)
    print('pridicting test')
    predict += model.predict(X_test, batch_size=1024)

predict = predict / num_folds
#print('Total CV score is {}'.format(np.mean(scores)))
sample_submission = pd.DataFrame.from_dict({'id': test['id']})
oof = pd.DataFrame.from_dict({'id': train['id']})
for c in list_classes:
    oof[c] = np.zeros(len(train))
    sample_submission[c] = np.zeros(len(test))
    
sample_submission[list_classes] = predict
sample_submission.to_csv(out_folder+'\\'+filename + str(num_folds) + '_test.csv', index=False)
oof[list_classes] = oof_predict
oof.to_csv(out_folder+'\\'+filename + str(num_folds) + '_train.csv', index=False)
