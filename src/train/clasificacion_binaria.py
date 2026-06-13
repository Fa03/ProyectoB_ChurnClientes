

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


class ClasificacionBinaria:



    def __init__(self):
        self.df = pd.read_csv('C:/Users/fab_t/Downloads/telco_churn_reducido.csv')



    def dividir_df (self,df):
        X = df.drop('Churn_Yes',axis=1).values
        Y = df['Churn_Yes'].values

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

        return X,Y, X_train, X_test, y_train, y_test

    def crearModeloBinario(self,X,Y, X_train, X_test, y_train, y_test):
        model = Sequential()
        num_neuronas = X.shape[1] #~** número de neuronas según tamaño del DF
        model.add(Dense(units=num_neuronas,activation='relu'))
        model.add(Dropout(0.5))  # la mitad de las neuronas en cada epoch para esta capa

        model.add(Dense(units=int(np.round(num_neuronas//2)),activation='relu'))  #~** // agregado para que de un número entero
        model.add(Dropout(0.5))  # la mitad de las neuronas en cada epoch para esta capa

        model.add(Dense(units=1,activation='sigmoid')) # ~** neuronas de salida igual a variables a predecir, en este caso solo 1 variable "Churn

        # Para clasificación binaria "binary_crossentropy"
        model.compile(loss='binary_crossentropy', optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=23)

        model.fit(x=X_train,
                  y=y_train,
                  epochs=600,
                  validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stop]
                  )

        model_loss = pd.DataFrame(model.history.history)
        model_loss.plot()
        plt.show()

        return model


    def entrena_modelo(self,X_test,y_test, model):
        predictions = model.predict_classes(X_test)
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


prueba_binaria = ClasificacionBinaria()
X,Y, X_train, X_test, y_train, y_test = prueba_binaria.dividir_df(prueba_binaria.df)
prueba_binaria.crearModeloBinario(X,Y, X_train, X_test, y_train, y_test)
































