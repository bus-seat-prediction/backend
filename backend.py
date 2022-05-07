from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
# https://flask-pymongo.readthedocs.io/en/latest/

#플라스크 설정
app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

@app.route("/prediction/<busnum>/<date>/<time>", methods=["GET"])
def get_busseat_prediction(busnum,date,time):
    array=[1,2,3]

    # 버스 모델 불러오기
    
    # 정류장 json 데이터랑 연동하기

    # 예측 데이터에서 결과값 찾기

    # 데이터 리턴하기 
    output = {
        'corpname':busnum,
        'date':date,
        'time':time}
    return jsonify({'result' : output})


# @app.route("/data/<corpname>", methods=["GET"])
# def get_data(corpname):
#     s = db.corpdata.find_one({'corpname':corpname})
#     d = []
#     for i in range(len(s['yearMonth'])):
#         d.append({
#             'month': s['yearMonth'][i],
#             'EG':s['EG'][i],
#             'EB':s['EB'][i],
#             'SG':s['SG'][i],
#             'SB':s['SB'][i],
#             'GG':s['GG'][i],
#             'GB':s['GB'][i],
#             'stock': s['stock'][i]
#         })

#     output = {
#         "data" : d
#     }
#     return jsonify({'result' : output})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
