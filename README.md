# Toxic-Challenge-Solution
Ranked 139 in the public and 528 in the private leaderboard. entered 15 days before competition end.link to the competition:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Scripts used for the kaggle toxic comment challenge.
1. some comment cleaning done including replacement of incorrect or misspeled words.removing puncuations,ipaddress,hyperlinks,username from the comments.
2. spell correction including replacement of the words which are not found in the embedding file.
3. using boosting and linear model on the preprocessed data.models used are: lightgbm,xgboost,ridge,logisticRegression.
4. using deep neural network model including- Lstm,Gru,Gru-Cnn and multiple layer of these models.
5. used corrrelation script to get least correlated models for the ensembling and Blending.
6. final two submits are one blending model and one ensemble model.
7. other models that are used but not in the final ensemble are Textcnn,Deepcnn,capsuleNet.All the Deep models are using pretrained glove 840b 300d embedding file and non deep models used tfidf and countvectorizer for vectorization.

## Mistakes:
1. using Gpu during last 3 days only.taking 7-24 hours to train a deep learning model for 2-3 fold on cpu against 1-2 hour for 5-8 fold on Gpu.
2. using nltk lemmatizer and stemming is a big mistake because it removed the context words which are used by the rnn models.
3. using only single embedding files for all the NN model which lead to highly correlated 1 layer models. have to use the fasttext,common wiki crawl,glove twitter.
4. start ensembling on the last day
5. last overfitting on the public leaderboard means instead of depend upon our local cv accuracy gave high preference to the models which scored well on public leaderboard(which only show accuracy result for 25% data).

## Leason Learned and top winner techniques:
1. Different Embedding - Have to use different type of embeddings for the model diversity.using same type of embedding leads to correlated models. using fasttext,glove for common-crawl,wikipedia and twitter,BPEmb embedding solve the words not in vocabulary problem. this is the subword embedding which try to solve the unknown word problem.LexVec is one more word embedding created using the wikipedia corpus.
2. pseudo_labelling - if accuracy is high near to 98 to 99 percent then use the pseudo-labelling technique and use the test high probability prediction from the top model and use them as the training set.
3. data augmentation - where we have to create the artificial data from the existing data.first split the training data into train and validation set and then do the data augmentation on the training set and used the validation data to get the idea that your model will work well on augmented data or not.
4. translation train time augmentation - translate the english comment to french,german or latin then again translate back to english by this way model will learn the noise in the data.
4.(i) translate the english training data to some other language like (french,german,spanish) then used word embedding for those language for training.
5. concating Embedding - try multiple embedding (glove,fasttext) in one single model like (image RGB channels).concating two embedding together improve the score.
see which word are not in fasttext then find those words in glove and add it to the fasttext embedding file.
6. Ensemble different type of models in the first layer of stacking try to use subset of models in all the ensembles. then in the 2nd layer of stacking average those first layer output or use different model on that.
7. try to combine different embeddings to make the 300d embedding to 900d like combined(Fasttext,glove,word2vec) vectors.
8. try to use the bayesian optimization to find hyperparameters of the model.

## Team Member
- [utsav aggarwal](https://github.com/utsav1)
