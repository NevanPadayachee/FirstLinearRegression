import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as fc
fc = fc.compat.v2.feature_column

#loads dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#print(dftrain.head()) prints the first 5 items in the cls
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#print(dftrain.describe())
#print(dftrain.shape)

categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []

for feature_name in categorical_columns:
    vocab=dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn (data_df, label_df, num_epochs=10, shuffle = True, batch_size=32 ):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))#create tf.data.Dataset object and its label
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)#splits dataset into batches of 32 and repeats procces
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain,y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)#applies featured column created above

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[4])#details of passenger
print(result[4]['probabilities'][1])#index 0 is not surviving and 1 is surviving
print(y_eval.loc[4])#did they survive 0 is did not survive and 1 is survived