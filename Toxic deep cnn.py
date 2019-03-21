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
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
import re
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from keras.models import load_model
from keras import initializers, regularizers, constraints, callbacks

print('loading embeddings vectors')
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.split(' ')) for o in open('glove.6B//glove.6B.50d.txt',encoding="utf8"))
embeddings_index = dict(get_coefs(*o.split(' ')) for o in open('glove.840B.300d.txt'))

def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001 , 0.0001]
    return a[ind] 

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

min_count = 6 #the minimum required word frequency in the text
max_features = 150000
maxlen = 200
num_folds = 6 #number of folds
batch_size = 256
epochs = 5
embed_size = 300 #embeddings dimension

train = pd.read_csv("spell_train.csv").fillna("no comment")
test = pd.read_csv("spell_test.csv").fillna("no comment")

list_sentences_train = train["comment_text"].str.lower()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train[list_classes] = train[list_classes].astype(np.int8)
target = train[list_classes]

#y = train[list_classes].values
#list_sentences_test = test["comment_text"].fillna("").values
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

filter_nr = 64
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 256
spatial_dropout = 0.2
dense_dropout = 0.5
train_embed = False
conv_kern_reg = regularizers.l2(0.00001)
conv_bias_reg = regularizers.l2(0.00001)

def get_model_cnn():

    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    
    #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add x directly toblock1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
    resize_emb = PReLU()(resize_emb)
    
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)
    
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
    
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    block4_output = add([block4, block3_output])
    block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)
    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)
    
    block5_output = add([block5, block4_output])
    block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)
    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)

    block6_output = add([block6, block5_output])
    block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)
    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)

    block7_output = add([block7, block6_output])
    output = GlobalMaxPooling1D()(block7_output)

    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(6, activation='sigmoid')(output)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

print('start modeling')
scores = []
predict = np.zeros((test.shape[0],6))
oof_predict = np.zeros((train.shape[0],6))

early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 5)
lr = callbacks.LearningRateScheduler(schedule)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
for train_index, test_index in kf.split(X_train):
    
    X_train1 , X_valid = X_train[train_index] , X_train[test_index]
    y_train, y_val = target.loc[train_index], target.loc[test_index]
    
    #y_train,y_val = target[train_index], target[test_index]

    model = get_model_cnn()
    model.fit(X_train1, y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks = [lr,early_stop])
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
sample_submission.to_csv('spell_dpcnn_' + str(num_folds) + '_test.csv', index=False)

oof[list_classes] = oof_predict
oof.to_csv('spell_dpcnn_'+str(num_folds)+'_train.csv', index=False)
