import time
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from getfeature import read_feature


seed = 7
numpy.random.seed(seed)

X,Y = read_feature("ALL3.txt")

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


'''print(X)
print(encoded_Y)
start_time = time.time()
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold) # 如果get_parameters方法报错,查看 https://github.com/fchollet/keras/pull/5121/commits/01c6b7180116d80845a1a6dc1f3e0fe7ef0684d8
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
end_time = time.time()
print("用时: ", end_time - start_time)
# larger model'''



def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, init='normal', activation='relu'))
    from keras.layers import Dropout
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


start_time = time.time()
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
end_time = time.time()
print("用时: ", end_time - start_time)
