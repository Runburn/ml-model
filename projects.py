from flask import Flask, request, json, jsonify
import pickle
import numpy as np
app = Flask(__name__)
rfc_model = pickle.load(open('latest_ABR_reg.pkl', 'rb'))


@app.route('/resellpredict', methods=['POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        try:
            prediction = rfc_model.predict([[
                json["make"],
                json["model"],
                json["fuel"],
                json["type"],
                json["condition"],
                json["year"],
                json["odometer"]]])
            # print(json["name"])
            print(prediction)
            prediction = np.power(2.71828, prediction)
            data = {
                "pred_price":int(prediction[0])
            }
            response = jsonify(data)
            response.status_code = 200
            return response
        except Exception as e:
            data = {
                "error":str(e)
            }
            response = jsonify(data)
            response.status_code = 500
            return response
            print("Error ",e)
    else:
        data = {
            "error":'error'
        }
        response = jsonify(data)
        response.status_code = 500
        return response
        # return 'Content-Type not supported!'
    # # Make a prediction using the model

    # # Render the result template with the prediction result
    # return render_template('result.html', prediction=prediction)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)
