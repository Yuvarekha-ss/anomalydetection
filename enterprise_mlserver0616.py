from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from socketserver import ThreadingMixIn
import json
import time
import enterprise_locate0616 as ML
import requests
import os
from requests.compat import urljoin
from time import sleep
import sys
import joblib
from pandas import read_csv
import math as Math
import csv
import pandas as pd
from threading import Timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_URL = os.environ.get("BASE_URL", "http://staging.viaviedge.net:7512/")

def call_at_interval(perriod,callback,args):
    while True:
        sleep(perriod)
        callback(*args)


def setInterval(period,callback,*args):
     Thread(target=call_at_interval,args=(perriod,callback,args)).start()


def setTimeout(seconds,callback,*args):
    t = Timer(seconds,callback,args=(args))
    t.start()


def update_error_status(sessionid,status,reason=None):

        update_request_data ={
        "status":status,
        "reason":reason
        }

        data_json = json.dumps(update_request_data)
        print(data_json)
        path = f"/gwdevice/trainingrequest/{sessionid}/_update"

        session_url = urljoin(BASE_URL, path)
        r = requests.put(session_url, data_json)
        print(r.status_code)


def Create_And_Save(tuned_data,modelname,sessionid):
    print('Status is processing Inside Create_And_Save')


    if tuned_data == []:

        print("update_trainingrequest for empty dataset")

        update_error_status(sessionid,"Failed", "Empty dataset")


    else:
        err, Threshold, Sessionid, Filename  = ML.CreateTrainSave(tuned_data,sessionid,modelname)
        if err == "" :

            print("update_trainingrequest for valid data")
            results_data = {
            "sessionid":sessionid,
            "modelname":Filename,
            "metadata":"Metadata",
            "threshold":Threshold
           }

            path = f"/gwdevice/trainingresult/_create?_id={sessionid}"

            session_url = urljoin(BASE_URL, path)
            r = requests.post(session_url, results_data)
            print(results_data)
            print(r.status_code)
            update_error_status(sessionid,"Success")

        else:
            print("PRINTING ERROR")
            print(err)
            print("update_trainingrequest for error occured")
            update_error_status(sessionid,"Failed", "Error occured")


def Open_And_Save(tuned_data, sessionid,modelname):

    print('Status is processing Inside Open_And_Save')

    if tuned_data == []:

        print("update_trainingrequest for empty dataset")

        update_error_status(sessionid,"Failed", "Empty dataset")

    else:
        err, Threshold, Sessionid, Modelname = ML.OpenReTrainSave(tuned_data,modelname, sessionid, modelname)
        if err == "" :
            print("model")
            print(Threshold)

            print(sessionid)
            print("update_trainingrequest for valid data")
            results_data = {
            "sessionid":Sessionid,
            "modelname":Modelname,
            "metadata":"Metadata",
            "threshold":Threshold
           }

            path = f"/gwdevice/trainingresult/_create?_id={sessionid}"

            session_url = urljoin(BASE_URL, path)
            r = requests.post(session_url,  results_data)
            print(results_data)
            print(r.status_code)
            update_error_status(sessionid,"Success")


def fetchdata(deviceid,starttime,endtime):
    newfilter = {

      "query": {
                                "bool": {
                                "must": [
                                        {"query_string": {"query": deviceid, "type": "phrase", "fields": ["deviceid"]}},
                                        {"range":{"_kuzzle_info.createdAt":{"gte":starttime,"lte":endtime}}},
                    {"range":{"Geolocation.lat":{"gt":0}}},


                                        ]

                                        }
                                },
                                "sort":{"_kuzzle_info.createdAt":"asc"}

        }

    print(newfilter)
    data_json = json.dumps(newfilter)
    path = f"/gwdevice/network/_search?size=500"
    session_url = urljoin(BASE_URL, path)

    r = requests.post(session_url, data_json)
    print(r.status_code)
    output = json.loads(r.text)
    total = output["result"]["total"]
    results = output["result"]["hits"]
    print("Fetching data")
    print(total)
    return results,total


def fetchrandata(imsi_imei,fieldname,starttime,endtime):
    st=starttime
    et=endtime
    newfilter = {
    
      "query": {
                                "bool": {
                                "must": [
                                        {"query_string": {"query": imsi_imei, "type": "phrase", "fields": [fieldname]}},
                                        {"range":{"ts":{"gte":st,"lte":et}}}
                   


                                        ]

                                        }
                                },
                                "sort":{"ts":"asc"}

        }

    print(newfilter)
    data_json = json.dumps(newfilter)
    path = f"/edge/network_ran/_search?size=500"
    session_url = urljoin(BASE_URL, path)

    r = requests.post(session_url, data_json)
    print(r.status_code)
    output = json.loads(r.text)
    total = output["result"]["total"]
    results = output["result"]["hits"]
    print("Fetching data")
    print(total)
    return results,total

def writetofile(results,filename, X,y):
   with open(filename, 'a') as f:

    prev_glid=""
    for result in results:
        try:
          val = result["_source"]
          parameters = val["PARAMETER"];

          for parm in parameters:
             for i in parm:
                if i == "HSDPA Channel Quality":
                   cqivalue = parm["value"][0]["CQI"]
                   break

          lat=lon=0
          if (prev_glid != int(val["Cell location info"]["Global Cell ID:"])) :
            lat,lon = ML.celllocation(int(val["Cell location info"]["Global Cell ID:"]),X,y)
          dist = ML.distance ( float(lat),val["Geolocation"]["lat"],float(lon),val["Geolocation"]["lon"])
          bearing = ML.getbearing(val["Geolocation"]["lat"],float(lat),val["Geolocation"]["lon"],float(lon))
          value = cqivalue if type(cqivalue) == int else 0
          mccsplit = val["MCC/MNC"].split("/")
          mcc = mccsplit[0]
          mnc = mccsplit[1]

          bandfdd = val["Band FDD"]
          numeric_filter = filter(str.isdigit, bandfdd)
          numeric_bandfdd = "".join(numeric_filter)

          if ((numeric_bandfdd !="") and (lat != 0)):
            output = str(mcc)+","+str(mnc)+","+str(val["rssi"])+","+str(val["rsrp"])+","+str(val["rsrq"])+","+str(val["rscp"])+","+str(val["snr"])+","+str(numeric_bandfdd)+","+str(val["Cell location info"]["PLMN"])+","+str(val["Cell location info"]["Global Cell ID:"])+","+ str(dist)+","+str(bearing)+","+str(val["Cell location info"]["TAC"])+","+str(val["Cell location info"]["Serving Cell ID"])+","+str(value)+","+str(val["Geolocation"]["lat"])+","+str(val["Geolocation"]["lon"])+"\n"
            f.write(output)

          else:
            print(result["_id"] + " "+"No value in bandfdd")

        except Exception as e:
            print(e)

   f.close()


def extractdata(results,filename,X,y):
   outdata=[]
   prev_glid = ""

   lat=lon=0
   with open(filename, 'a') as f:
    for result in results:
      try:
          val = result["_source"]
          parameters = val["PARAMETER"];

          for parm in parameters:
             for i in parm:
                if i == "HSDPA Channel Quality":
                   cqivalue = parm["value"][0]["CQI"]
                   break

          value = cqivalue if type(cqivalue) == int else 0
          mccsplit = val["MCC/MNC"].split("/")
          mcc = mccsplit[0]
          mnc = mccsplit[1]

          bandfdd = val["Band FDD"]
          numeric_filter = filter(str.isdigit, bandfdd)
          numeric_bandfdd = "".join(numeric_filter)

          if (numeric_bandfdd !=""):
                ts = val["time_stamp"][0]["SystemEpoch"]
                output = str(mcc)+","+str(mnc)+","+str(val["rssi"])+","+str(val["rsrp"])+","+str(val["rsrq"])+","+str(val["rscp"])+","+str(val["snr"])+","+str(numeric_bandfdd)+","+str(val["Cell location info"]["PLMN"])+","+str(val["Cell location info"]["Global Cell ID:"])+","+str(val["Cell location info"]["TAC"])+","+str(val["Cell location info"]["Serving Cell ID"])+","+str(value)+","+str(val["Geolocation"]["lat"])+","+str(val["Geolocation"]["lon"])+"\n"
                outdata.append(ts)
                f.write(output)

          else:
            print(result["_id"] +" "+ "No value in bandfdd")
      except Exception as e:
        print(e)

    f.close()
    return outdata
    

def extractrandata(results,filename,X,y):
   outdata=[]
   prev_glid = ""

   lat=lon=0
   with open(filename, 'a') as f:
    for result in results:
      try:
          val = result["_source"]
          mcc = val["mcc"]
          mnc = val["mnc"]
          rsrp = val["rsrp"]
          rsrq = val["rsrq"]
          globalcellid = val["globalcellid"]
          phycellid = val["phycellid"]
          lat = val["lat"]
          lon = val["lon"]
          ts=val["ts"]
          output = str(mcc)+","+str(mnc)+","+str(rsrp)+","+str(rsrq)+","+str(globalcellid)+","+str(phycellid)+","+str(lat)+","+str(lon)+"\n"
          val["mcc"]
          outdata.append(ts)
          f.write(output)
          

      except Exception as e:
        print(e)

    f.close()
    return outdata

path_to_csv = "./Knnmodel/"

def sorting(path_to_csv,modelname,sessionid):
  try:
    file = str(path_to_csv) + str(modelname) +"_"+ str(sessionid)+".csv"
    df2 = read_csv(file, header=1)
    data2 = df2.values
    col2=len(df2.columns)
    X3 = df2[df2.columns[9:12]]
    X2 = X3.sort_values(X3.columns[0])
    print(X2)

    d = X2.shape[0]

    previd =""
    xdat = X2.to_numpy(dtype='float64')
    max1=max2=max3=max4=0
    avg1=avg2=avg3=avg4=0
    count1=count2=count3=count4=0
    filename = str(path_to_csv) + "bearing_"+ str(modelname) +"_"+ str(sessionid)+".csv"


    with open(filename, 'w') as f:
     for i in range(d):
        if (previd != xdat[i][0]) and  (previd != ""):
            av1 = 0 if count1 == 0 else avg1/count1
            av2 = 0 if count2 == 0 else avg2/count2
            av3 = 0 if count3 == 0 else avg3/count3
            av4 = 0 if count4 == 0 else avg4/count4
            output = "{},  {}, {}, {}, {},{},{},{},{}".format(xdat[i-1][0],max1,max2,max3,max4,av1,av2,av3,av4)
            print(output)
            f.write(output)
            f.write('\n')
            max1=max2=max3=max4=0
            avg1=avg2=avg3=avg4=0
            count1=count2=count3=count4=0

        previd = xdat[i][0]

        if (xdat[i][2] > 0 and xdat[i][2] <= 90):
            max1 = xdat[i][1] if xdat[i][1] > max1 else max1
            avg1 += xdat[i][1]
            count1 += 1

        if (xdat[i][2] > 90 and xdat[i][2] <= 180):
            max2 = xdat[i][1] if xdat[i][1] > max2 else max2
            avg2 += xdat[i][1]

            count2 +=1

        if (xdat[i][2] > 180 and xdat[i][2] <= 270):
            max3 = xdat[i][1] if xdat[i][1] > max3 else max3
            avg3 += xdat[i][1]
            count3 +=1

        if (xdat[i][2] > 270 and xdat[i][2] <= 360):
            max4 = xdat[i][1] if xdat[i][1] > max4 else max4
            avg4 += xdat[i][1]
            count4 +=1

     av1 = 0 if count1 == 0 else avg1/count1
     av2 = 0 if count2 == 0 else avg2/count2
     av3 = 0 if count3 == 0 else avg3/count3
     av4 = 0 if count4 == 0 else avg4/count4

     output = "{},  {}, {}, {}, {},{},{},{},{}".format(xdat[i][0],max1,max2,max3,max4,av1,av2,av3,av4)
     f.write(output)
     f.write('\n')
    f.close()

  except Exception as e:
            print(e)


def Knncreate(sessionid,deviceid,starttime,endtime,modelname):
  
  try:
    file ="ltesectors.csv"

    url = str(path_to_csv) +str(file)
    if os.path.isfile(url):
        print("File exists")
    else:
        raise Exception("File doesn't exists")

    df1 = read_csv(url,header=None)
    col=len(df1.columns)
    X,y = df1[df1.columns[0]],df1[df1.columns[col-2:col]]
    results=[]
    total = 1
    file = modelname +"_"+ sessionid
    filename = modelname+"_"+sessionid+".csv"

    while (total > len(results)):
        try:
            results,total = fetchdata(deviceid,starttime,endtime)
            if (len(results) > 0):
                result = results[len(results)-1]
                print("Total {} Results {}".format(total,len(results)))

                starttime = result["_source"]["_kuzzle_info"]["createdAt"]
                print(starttime)

                writetofile(results,path_to_csv+filename,X,y)
            else:
              results = total
              print("In Knncreate else")
              break
        except Exception as e:
            print(e)

    print("Outside while loop")
    sorting(path_to_csv,modelname,sessionid)

    results_data = {
            "sessionid":sessionid,
            "modelname":file,
            "metadata":"Metadata"
           }

    path = f"/gwdevice/trainingresult/_create?_id={sessionid}"

    session_url = urljoin(BASE_URL, path)
    r = requests.post(session_url, results_data)

    update_error_status(sessionid,"Success")
  except Exception as e:
     update_error_status(sessionid,"Failed",e)


def Knnpredict(sessionid,deviceid,starttime,endtime,filename):
  try:
    file ="ltesectors.csv"

    url = str(path_to_csv) +str(file)
    if os.path.isfile(url):
        print("File exists")
    else:
        raise Exception("File doesn't exists")

    df1 = read_csv(url,header=0)
    data = df1.values
    col=len(df1.columns)
    X,y = df1[df1.columns[0]],df1[df1.columns[col-2:col]]
    results=[]
    total = 1

    extracteddata=[]
    while (total > len(results)):
     try:
        results,total = fetchdata(deviceid,starttime,endtime)
        result = results[len(results)-1]
        print( "Total {} Results {}".format(total,len(results)))
        starttime = result["_source"]["_kuzzle_info"]["createdAt"]
        print(starttime)
        extracteddata= extracteddata +extractdata(results,path_to_csv+"tmp"+str(sessionid)+".csv", X,y)

     except Exception as e:
            print(e)

    print("Outside while loop")
    print("Length of timestamp array {}".format(len(extracteddata)))
    return path_to_csv+"tmp"+str(sessionid)+".csv",extracteddata

  except Exception as e:
            print(e)
            
            
            
def Knnpredict_randata(sessionid,imsi_imei,fieldname,starttime,endtime,filename):
  try:
    file ="ltesectors.csv"

    url = str(path_to_csv) +str(file)
    if os.path.isfile(url):
        print("File exists")
    else:
        raise Exception("File doesn't exists")

    df1 = read_csv(url,header=0)
    data = df1.values
    col=len(df1.columns)
    X,y = df1[df1.columns[0]],df1[df1.columns[col-2:col]]
    results=[]
    total = 1

    extractedrandata=[]
    while (total > len(results)):
     try:
        print(starttime)
        print(endtime)
        results,total = fetchrandata(imsi_imei,fieldname,starttime,endtime)
        result = results[len(results)-1]
        print( "Total {} Results {}".format(total,len(results)))
        starttime = result["_source"]["ts"]
        print(starttime)
        extractedrandata= extractedrandata +extractrandata(results,path_to_csv+"tmp"+str(sessionid)+".csv", X,y)

     except Exception as e:
            print(e)

    print("Outside while loop")
    print("Length of timestamp array {}".format(len(extractedrandata)))
    return path_to_csv+"tmp"+str(sessionid)+".csv",extractedrandata

  except Exception as e:
            print(e)


def Infer(modelname, data, threshold, sessionid):
    print('Status is processing inside Infer model')

    if data == []:
        print("update_trainingrequest for empty dataset")
        update_error_status(sessionid,"Failed", "Empty dataset")

    else:
        err, Predictions, Sessionid = ML.inferencing(modelname, data, threshold,sessionid)
        if err == "" :
            print("Predictions")
            print(Predictions)
            results_data = {
            "sessionid":Sessionid,
            "modelname":modelname,
            "result":Predictions
            }
            print(results_data)
            data_json = json.dumps(results_data)
            print(data_json)
            path = f"/gwdevice/trainingresult/_create?_id={sessionid}"
            session_url = urljoin(BASE_URL, path)
            r = requests.post(session_url, data_json)
            print(r.status_code)
            update_error_status(sessionid,"Success")

        else:
            print("PRINTING ERROR")
            print(err)
            print("update_trainingrequest for error occured")
            update_error_status(sessionid,"Failed", err)


def getdeviceid(value, variable_name):
   try:
    getdeviceid_filter = {"query": {"bool": {"must": [{"query_string": {"query": value,"type": "phrase", "fields": [variable_name]}}]}}}
    #print(getdeviceid_filter)
    data_json = json.dumps(getdeviceid_filter)
    path = f"/gwdevice/edgedev/_search?size=5"
    session_url = urljoin(BASE_URL, path)
    deviceid = requests.post(session_url, data_json)
    #print(deviceid.status_code)
    output = json.loads(deviceid.text)
    deviceid_data = output["result"]["hits"][0]["_source"]["deviceid"]
    return deviceid_data
    
   except Exception as e:
    return ""


serve_from = "//Tensormodel/"
class handler(BaseHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        BaseHTTPRequestHandler.end_headers(self)


    #def do_GET(self):
       # path = serve_from + self.path.strip("/")
        #print(path)
        #if not os.path.abspath(path).startswith(serve_from):
           # self.send_response(403)
           # self.end_headers()
           # self.wfile.write(b'Private!')
        #elif os.path.isdir(path[2:]):
           # try:
               # self.send_response(200)
               # self.end_headers()
               # self.wfile.write(str(os.listdir(path[2:])).encode())
            #except Exception as e:
               # print(e)
               # self.send_response(500)
               # self.end_headers()
               # self.wfile.write(b'error')
       # else:
           # try:
               # with open(path[2:], 'rb') as f:
                   # data = f.read()
               # self.send_response(200)
               # self.end_headers()
               # self.wfile.write(data)

           # except Exception as e:
               # print(e)
               # self.send_response(500)
               # self.end_headers()
               # self.wfile.write(b'error')


    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")

    def do_POST(self):
      if self.path == "/cts":
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        content_len = int(self.headers.get('Content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body.decode('utf-8'))
        dat = json.loads(post_body)

        print(dat["sessionid"])

        tuned_data = list(dict.fromkeys(dat["data"]))
        print(tuned_data)

        setTimeout(2,Create_And_Save,tuned_data,dat["modelname"], dat["sessionid"])
        message = "Create Train and Save Model"
        self.wfile.write(bytes(message, "utf8"))

      elif self.path == "/ots":
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        content_len = int(self.headers.get('Content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body.decode('utf-8'))
        dat = json.loads(post_body)

        print(dat["sessionid"])


        tuned_data = list(dict.fromkeys(dat["data"]))
        print(tuned_data)

        setTimeout(2,Open_And_Save,tuned_data, dat["sessionid"],dat["modelname"])

        message = "Open train and save model"
        self.wfile.write(bytes(message, "utf8"))

      elif self.path == "/infer":
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        content_len = int(self.headers.get('Content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body.decode('utf-8'))
        dat = json.loads(post_body)
        print(dat["sessionid"])
        tuned_data = list(dict.fromkeys(dat["data"]))
        print(tuned_data)
        threshold = dat["threshold"]
        setTimeout(2,Infer,dat["modelname"],dat["data"],threshold, dat["sessionid"])
        message = "Infer model"
        self.wfile.write(bytes(message, "utf8"))

      elif self.path == "/knncreate":
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        content_len = int(self.headers.get('Content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body.decode('utf-8'))
        dat = json.loads(post_body)
        sessionid=dat["sessionid"]
        deviceid = dat["deviceid"]
        starttime = dat["starttime"]
        endtime = dat["endtime"]
        filename = dat["modelname"]
        setTimeout(2,Knncreate,sessionid,deviceid,starttime,endtime,filename)
        message = "Knn Create"
        self.wfile.write(bytes(message, "utf8"))

      elif self.path == "/knnpredict":
        content_len = int(self.headers.get('Content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body.decode('utf-8'))
        dat = json.loads(post_body)
        sessionid = dat["sessionid"]
        if "imsi" in dat:
         imsi_imei = dat["imsi"]
         fieldname="imsi"
         deviceid = getdeviceid(imsi_imei, "IMSI")
        elif "imei" in dat:
         imsi_imei = dat["imei"]       
         fieldname="imei"
         deviceid = getdeviceid(imsi_imei, "IMEI")
        starttime = dat["starttime"]
        endtime = dat["endtime"]
        filename = dat["modelname"]
        report_type=dat["type"]
        
        if(report_type=="gps"):
        
         if ((deviceid != None and deviceid !="") and (starttime != None and starttime !="") and (endtime != None and endtime !="") and len(filename) > 0):
            print(deviceid)
            outfile,ts = Knnpredict(sessionid,deviceid,starttime,endtime,filename)
            inputfile = filename
            filename_ltesector="ltesectors.csv"
            outfile = "tmp"+str(sessionid)+".csv"
            mae,data,anomalyavg_all, err = ML.GetGeoLocations(path_to_csv, outfile,filename_ltesector,inputfile,report_type)
            print(err)
            print("*********************")
            print(len(data))
            print(len(ts))
            data = {"ts":ts,"locations":data,"anomaly":anomalyavg_all}

            if os.path.exists(path_to_csv+outfile):
                  os.remove(path_to_csv+outfile)

            if err == "":
                self.send_response(200)
                self.send_header('Content-type','application/json')
                self.end_headers()
                self.wfile.write(str(data).replace("'",'"').encode("utf-8"))

            else:
                self.send_response(400)
                self.send_header('Content-type','application/json')
                self.end_headers()
        
        
        elif(report_type=="ran"):
         print("randata")
         imsi_imei_string = str(imsi_imei)    
        
         if(len(imsi_imei_string)>14):
            imsi_imei = imsi_imei_string[:-1]
            print(imsi_imei)
         if ((imsi_imei != None and imsi_imei !="") and (starttime != None and starttime !="") and (endtime != None and endtime !="") and len(filename) > 0):
            
            outfile,ts = Knnpredict_randata(sessionid,imsi_imei,fieldname,starttime/1000,endtime/1000,filename)
            inputfile = filename
            filename_ltesector="ltesectors.csv"
            outfile = "tmp"+str(sessionid)+".csv"
            mae,data,anomalyavg_all,err = ML.GetGeoLocations(path_to_csv, outfile,filename_ltesector,inputfile,report_type)
            print(err)
            print("*********************")
            print(len(data))
            print(len(ts))
            data = {"ts":ts,"locations":data}
  
            if os.path.exists(path_to_csv+outfile):
                  os.remove(path_to_csv+outfile)

            if err == "":
                self.send_response(200)
                self.send_header('Content-type','application/json')
                self.end_headers()
                self.wfile.write(str(data).replace("'",'"').encode("utf-8"))

            else:
                self.send_response(400)
                self.send_header('Content-type','application/json')
                self.end_headers()
            
            
        else:
            self.send_response(400)
            self.send_header('Content-type','application/json')
            self.end_headers()

# with HTTPServer(('', 8001), handler) as server:
    # server.serve_forever()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

with ThreadingHTTPServer(('', 8003), handler) as server:
    server.serve_forever()