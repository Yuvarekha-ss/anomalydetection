import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow import Tensor
import joblib
import os
import math as Math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import requests
import json
from requests.compat import urljoin
from pandas import read_csv


class AutoEncoder(Model):
  """
  Parameters
  ----------
  output_units: int
    Number of output units

  code_size: int
    Number of units in bottle neck
  """

  def __init__(self, output_units, code_size=8):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])

  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded


def CreateModel(features,target):
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size = 0.1, stratify=target)
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        x_train_scaled = min_max_scaler.fit_transform(features.copy())
        scaler = min_max_scaler
        x_test_scaled = min_max_scaler.transform(x_test.copy())
        return x_train_scaled, x_test_scaled,scaler
    except Exception as e:
        return e


def TrainModel(x_train_scaled,x_test_scaled):
    try:
        model = AutoEncoder(output_units=x_train_scaled.shape[1])
        # configurations of model
        model.compile(loss='msle', metrics=['mse'], optimizer='SGD')
        history = model.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=30,
            batch_size=512,
            validation_data=(x_test_scaled , x_test_scaled))
        reconstructions = model.predict(x_train_scaled)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
        # threshold for anomaly scores
        threshold = np.mean(reconstruction_errors.numpy())         + np.std(reconstruction_errors.numpy())
        print("The model is generated.")
        return model, threshold
    except Exception as e:
        return e


def SaveModel(model,sessionid,modelname):
    try:
        Filename = modelname +"_" + sessionid
        tf.keras.models.save_model(model,'./Tensormodel/%s' %(Filename))
        print("Model is Saved in %s location" %(Filename))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        path = "./TFlite/%s"%(Filename)
        if(os.path.isdir(path) == False):
            os.mkdir(path)

        with open("./TFlite/%s/Model.tflite"%(Filename), 'wb') as f:
            f.write(tflite_model)

        print("TFliteModel is Saved " )

        return Filename
    except Exception as e:

        return e


def CreateTrainSave(dataset,sessionid,modelname):
    try :
        if len(dataset) == 0:
            invalid_CTS_dataset = "There is no data. Please enter valid dataset"
            return invalid_CTS_dataset,0, sessionid, ""

        else:
            data1 = np.array(dataset)
            data = data1.reshape(len(data1),1)
            target = np.ones((len(data),),dtype=int)
            x_train_scaled, x_test_scaled,scaler = CreateModel(data,target)
            model, threshold = TrainModel(x_train_scaled,x_test_scaled)
            Filename = SaveModel(model,sessionid,modelname)

            path = './/Tensormodel//%s//minmaxscalar.pkl' %(Filename)
            joblib.dump(scaler, path)
            path1 = './TFlite/%s/minmaxscalar.pkl' %(Filename)
            joblib.dump(scaler, path1)

            a_file = open("./Tensormodel/%s/data.txt" %(Filename), "w")
            for row in data:
                np.savetxt(a_file, row)
            a_file.close()

            return "", threshold, sessionid, Filename

    except Exception as e:

        print("Inside Exception")
        return e, 0, sessionid, ""


def get_predictions(model, row_scaled,threshold):
    try:
        print("Inside get_predictions")
        print(threshold)
        predictions = model.predict(row_scaled)
        # provides losses of individual instances
        errors = tf.keras.losses.msle(predictions, row_scaled)
        # 0 = anomaly, 1 = normal
        anomaly_mask = pd.Series(errors) > float(threshold)
        preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
        return "",preds
    except Exception as e:
        print(e)
        return e,0


def GetModel(modelname):
    try:
        print(modelname)
        model = tf.keras.models.load_model('.//Tensormodel//%s' %(modelname))
        return model
    except (FileNotFoundError, IOError):
        print("Model not found.")


def inferencing(modelname,dataset,threshold, sessionid):
    print("Inside inferencing ML "+modelname)
    try:
        if len(dataset) == 0:
            invalid_dataset = "There is no data. Please enter valid dataset"
            return invalid_dataset,0, sessionid
        else:
            data = np.array(dataset)
            data = data.reshape(len(data),1)
            try:
                scalar = joblib.load('.//Tensormodel//%s//minmaxscalar.pkl'%(modelname))
            except (FileNotFoundError, IOError):
                infer_model ="Scalar file not found"
                return infer_model,0, sessionid
            row_scaled = scalar.transform(data.copy())
            model = GetModel(modelname)
            err,pred = get_predictions(model, row_scaled,threshold)
            if(err == ""):
                print("pred")
                print(pred)
                res = {dataset[i]: pred[i] for i in range(len(dataset))}
                res_st= str(res)
                print(res_st)
                return "", res_st,sessionid
            else:
                return err, 0, sessionid

    except Exception as e:
        print("Inside Exception")
        return e, 0, sessionid



def RetrainModel(x_train_scaled,x_test_scaled,model):
    try:
        model.fit(
        x_train_scaled,x_train_scaled,epochs=30,batch_size=512,
        validation_data=(x_test_scaled , x_test_scaled))

        reconstructions = model.predict(x_train_scaled)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
        # threshold for anomaly scores
        threshold = np.mean(reconstruction_errors.numpy())         + np.std(reconstruction_errors.numpy())
        print("The model is Retrained.")
        return model, threshold
        print(model)
        print(threshold)
    except Exception as e:
        return e

def OpenReTrainSave(dataset,model,sessionid,modelname):
    try:
        original_array = np.loadtxt(
        ".//Tensormodel//%s//data.txt" %(modelname))
        if len(dataset) == 0:
            raise Exception("There is no data. Please enter valid dataset")

        else:
            data = np.array(dataset)
            data = np.append(original_array,data)
            data = data.reshape(len(data),1)
            target = np.ones((len(data),),dtype=int)
            x_train_scaled, x_test_scaled,scaler= CreateModel(data,target)
            model = GetModel(model)

            retrainmodel, threshold = RetrainModel(x_train_scaled,x_test_scaled, model)
            modelname = SaveModel(retrainmodel,sessionid,modelname)
            path = './/Tensormodel//%s//minmaxscalar.pkl' %(modelname)
            joblib.dump(scaler, path)
            a_file = open(".//Tensormodel//%s//data.txt" %(modelname), "w")
            for row in data:
                np.savetxt(a_file, row)
            a_file.close()

            return "",threshold, sessionid,modelname

    except Exception as e:

        print("Inside Exception")
        return e, 0, sessionid, ""


def getbearing(φ1,φ2,λ1,λ2):
    y = Math.sin(λ2-λ1) * Math.cos(φ2)
    x = Math.cos(φ1)*Math.sin(φ2) -Math.sin(φ1)*Math.cos(φ2)*Math.cos(λ2-λ1)
    θ = Math.atan2(y, x)
    brng = (θ*180/Math.pi + 360) % 360
    return brng


def celllocation(globalcellid,XB,yB):
    try:
        datx = XB.to_numpy(dtype='int64').tolist()
        daty = yB.to_numpy(dtype='float64').tolist()
        index = datx.index(globalcellid)
        return daty[index][0],daty[index][1]
    except Exception as e:
        print(e)
        print("Not found")
        return 0,0


def distance(lat1,lat2, lon1, lon2):
  try:
    lon1 = lon1 * Math.pi / 180;
    lon2 = lon2 * Math.pi / 180;
    lat1 = lat1 * Math.pi / 180;
    lat2 = lat2 * Math.pi / 180;
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = Math.pow(Math.sin(dlat / 2), 2)+ Math.cos(lat1) * Math.cos(lat2)* Math.pow(Math.sin(dlon / 2),2)
    c = 2 * Math.asin(Math.sqrt(a))
    r = 6371

    return(c * r)
  except Exception as e:
    return 0


def getglobalmaxdist(glcellid,brng,X2):
   dist=avg =0
   dist1=dist2=dist3=dist4=0
   av1=av2=av3=av4=0
   d = X2.shape[0]
   xdat = X2.to_numpy(dtype='float64')


   count=0
   prev =""
   current =""
   for i in range(d):
      current = xdat[i][0]
      if (xdat[i][0] == glcellid):
           dist1 = xdat[i][1] if xdat[i][1] > dist1 else dist1
           dist2= xdat[i][2] if xdat[i][2] > dist2 else dist2
           dist3 = xdat[i][3] if xdat[i][3] > dist3 else dist3
           dist4 = xdat[i][4] if xdat[i][4] > dist4 else dist4
           av1 += xdat[i][5]
           av2 += xdat[i][6]
           av3 += xdat[i][7]
           av4 += xdat[i][8]
           count = count+1
           prev = xdat[i][0]
      elif ((prev != "") and (prev != current)):
            break

      else:
           continue

   if brng >0  and brng <=90:
      dist = dist1
      avg = 0 if count == 0 else av1/count
   if brng > 90 and brng <=180:
      dist = dist2
      avg = 0 if count == 0 else av2/count
   if brng > 180 and brng <=270:
      dist = dist3
      avg = 0 if count == 0 else av3/count
   if brng > 270 and brng <=360:
      dist = dist4
      avg = 0 if count == 0 else av4/count
   return dist,avg

def csv_merger(inputpath_to_csv,input_folder):

    li=[]
    for i in input_folder:
        path1 = os.path.join(inputpath_to_csv,i)
        df = pd.read_csv(path1, index_col=None, header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

def GetGeoLocations(path_to_csv, outputpath,filename_ltesector,filename,report_type):
    anomalyavg_all = []
    start_time = time.time()
    try:
        if report_type == "ran":

            df = csv_merger(path_to_csv,filename)
            df = df.dropna()
            #df = df.fillna(0)
            #print(df)
            #df = df.drop(df.columns[[2,5,6,7,8,10,11,12,14]], axis=1)
            if len(df) >8:
                df = df.drop(df.columns[[2,5,6,7,8,10,11,12,14]], axis=1)
            data = df.values
            col = len(df.columns)
            X, y = df[df.columns[0:col - 2]], df[df.columns[col - 2:col]]
            model = KNeighborsRegressor()
            model.fit(X, y)
            df1 = read_csv(path_to_csv+outputpath, header=None)
            #df1 = df1.drop(df1.columns[[2,5,6,7,8,10,11,12,14]], axis=1)
            #df1 = df1.dropna()
            df1 = df1.fillna(0)
            data1 = df1.values
            col = len(df1.columns)
            X1, y1 = df1[df1.columns[0:col - 2]], df1[df1.columns[col - 2:col]]
            #X1 = df1[df1.columns[0:col]]
            

        else: 
            df = csv_merger(path_to_csv,filename)
            df = df.dropna()
            if len(df) >8:
                df = df.drop(df.columns[[2,5,6,7,8,10,11,12,14]], axis=1)
            data = df.values
            col = len(df.columns)
            X, y = df[df.columns[0:col - 2]], df[df.columns[col - 2:col]]
            model = KNeighborsRegressor()
            model.fit(X, y)
            df1 = read_csv(path_to_csv+outputpath, header=None)
            df1 = df1.drop(df1.columns[[2,5,6,7,8,10,11,12,14]], axis=1)

            data1 = df1.values
            col = len(df1.columns)
            X1 = df1[df1.columns[0:col]]

        print(X1)
        globalcelllist = df1[df1.columns[4]]
        dat = X1.to_numpy(dtype='float64')
        d = X1.shape[0]
        st=""
        df2 = read_csv(path_to_csv+filename_ltesector, header=None)
        data2 = df2.values
        col2=len(df2.columns)
        X2,y2 = df2[df2.columns[0]],df2[df2.columns[col2-2:col2]]
        df_from_each_file = (read_csv(path_to_csv +"bearing_"+ f, header=None) for f in filename)
        df3 = pd.concat(df_from_each_file)
        data3 = df3.values
        col3=len(df3.columns)
        X3 = df3[df3.columns[0:col3]]
        X3 = X3.sort_values(X3.columns[0])

        prevglid=""
        data = []
        for i in range(d):
            yhat1 = model.predict([dat[i]])
            list = yhat1[0].tolist()
            pred_lat = list[0]
            pred_lon = list[1]
            if prevglid != globalcelllist[i]:
                globallat,globallon = celllocation(globalcelllist[i],X2,y2)
                prevglid = globalcelllist[i]

            dist1 = distance(float(globallat),list[0],float(globallon),list[1])
            brng = getbearing(list[0],float(globallat),list[1],float(globallon))
            if globallat > 0:
                gldist,avg = getglobalmaxdist(globalcelllist[i],brng,X3)

            else:
                gldist=avg=0
            anomaly = 0 if (dist1 < (gldist)) else 1
            anomalyavg = 0 if dist1 < avg*1.1 else 1

            anomalyavg_all.append(anomalyavg)
            list = yhat1[0].tolist()
            data.append(list)

        return 0,data,anomalyavg_all, ""
        #return data, ""
    except Exception as e:
        print(e)
        return "", [],[], e


