from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime as dt

# 테스트 url
# http://localhost/prediction/7612/20220420/15

#플라스크 설정
app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

def reverse_min_max_scaling(org_x, x): #종가 예측값
    org_x_np = np.asarray(org_x) 
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
    # return (x_np * (55 - 0 + 1e-7)) + org_x_np.min()


def make_sequene_dataset(feature, label, window_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list
    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7 )



@app.route("/prediction/<busnum>/<date>/<time>", methods=["GET"])
def get_busseat_prediction(busnum,date,time):

    std_date = dt.date(2022,4,15)
    input_date = dt.date(int(date[0:4]),int(date[5:7]),int(date[8:10]))
    diff_date = input_date - std_date

    time=time[0:2]
    rest_date = int(diff_date.days%14)
    std_date = std_date + dt.timedelta(days=rest_date)

    date=std_date

    date=date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")

    # 버스 모델 불러오기
    model = keras.models.load_model('./models/20220305_20220415_{}.h5'.format(busnum))

    # 버스데이터 받아오기
    raw_df = pd.read_csv('./data/20220305_20220415_{}.csv'.format(busnum))

    columns = list(raw_df.columns)

    del columns[0]
    scale_cols = list(columns)

    scaled_df = min_max_scaling(raw_df[scale_cols])
    scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

    feature_cols=columns[0:len(columns)]
    label_cols=columns[0:len(columns)]

    feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
    label_df = pd.DataFrame(scaled_df, columns=label_cols)

    feature_np = feature_df.to_numpy()
    label_np = label_df.to_numpy()

    split = -336
    window_size = 168

    X, Y = make_sequene_dataset(feature_np, label_np, window_size)

    x_test = X[split:]

    #예측하기
    pred = model.predict(x_test)

    result = reverse_min_max_scaling(raw_df[scale_cols], pred)
    int_pred = np.asarray(result, dtype = int)


    # 예측 데이터에서 결과값 찾기
    presenttime=dt.datetime(2022,4,16,3,0,0)

    predict_year = date[0:4]
    predict_month = date[4:6]
    predict_day = date[6:8]
    predict_time = time
    predict_time=dt.datetime(int(predict_year),int(predict_month),int(predict_day),int(predict_time),0,0)

    result_hour = int(((predict_time-presenttime).total_seconds())/60/60)

    #결과값 리스트 변환
    int_pred=list(int_pred[result_hour])

    for i in range(len(int_pred)):
        if(int_pred[i]<=0):
            int_pred[i]=0
        else:
            int_pred[i]=int(int_pred[i])

    # 혼잡도 추가해야함
    complexity=[]
    for i in range(len(int_pred)):
        if(int_pred[i]>=35):
            complexity.append(3)
        elif(20<=int_pred[i] and int_pred[i]<35):
            complexity.append(2)
        else:
            complexity.append(1)

    # 지금은 1주인데 적어도 1달까지는 예측시켜줘야함!

    station_info_data=[]

    for i in range(len(scale_cols)):
        tmp_dict={}
        tmp_dict['station']=scale_cols[i]
        tmp_dict['int_pred']=int_pred[i]
        tmp_dict['complexity']=complexity[i]
        station_info_data.append(tmp_dict)
    # 데이터 리턴하기 
    output = {
        'busnum':str(busnum),
        'station_info_data':station_info_data,
        'date':str(date),
        'time':str(time)
        }
    return jsonify({'result' : output})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
