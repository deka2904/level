import pandas as pd
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/mobile_NA2.csv")

df = pd.DataFrame(data=df)
df

df.info()

df.isna().sum()

df['REPORTED_SATISFACTION'].value_counts()

df.info()

df['HOUSE'] = df['HOUSE'].fillna(df['HOUSE'].mean())
df['REPORTED_SATISFACTION'].fillna(df['REPORTED_SATISFACTION'].mode()[0], inplace=True)

df_numeric = df.select_dtypes(include=['float64', 'int64'])  # 숫자형 열만 선택
sns.heatmap(df.corr(), annot=True)

df.isna().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['REPORTED_USAGE_LEVEL'] = le.fit_transform(df['REPORTED_USAGE_LEVEL'])

df['REPORTED_USAGE_LEVEL'] = df['REPORTED_USAGE_LEVEL'].astype('float')

cols = df.select_dtypes('object').columns.to_list()
cols

df = pd.get_dummies(data=df, columns=cols, drop_first=True)
df

target = 'CHURN'
x=df.drop(target,axis=1)
y=df[target]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2023)

x_train

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import to_categorical

x_train.shape

clear_session()
model = Sequential([
    Input(shape=(18,)),
    Dense(64,activation='relu'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(16,activation='relu'),
    Dropout(0.2),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, restore_best_weights=True)

mc = ModelCheckpoint('my_checkpoint.h5',monitor='val_accuracy',save_best_only=True,verbose=1)

hist = model.fit(x_train, y_train,
                callbacks=[es,mc],
                validation_data=(x_test, y_test),
                epochs=1000,
                batch_size=32
                )

hist.history

import matplotlib.pyplot as plt

plt.title('ACC')
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()