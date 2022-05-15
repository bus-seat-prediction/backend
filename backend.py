from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
from tensorflow import keras
import numpy as np
import pandas as pd


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

    split = -168
    window_size = 168

    X, Y = make_sequene_dataset(feature_np, label_np, window_size)

    x_test = X[split:]

    # 예측 데이터에서 결과값 찾기
    pred = model.predict(x_test)

    result = reverse_min_max_scaling(raw_df[scale_cols], pred)
    int_pred = np.asarray(result, dtype = int)
    int_pred=list(int_pred[5])
    print(int_pred)
    # 정류장 json 데이터랑 연동하기
    # with open('station_dict.json') as f:
    #     station_dict = json.load(f)

    # stations = station_dict[busnum]

    # 데이터 리턴하기 
    output = {
        'busnum':busnum,
        'stations':scale_cols,
        'prediction':int_pred,
        'date':date,
        'time':time
        }
    return jsonify({'result' : output})



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
