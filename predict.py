# # Loading the Model
import pickle

from flask import Flask
from flask import request
from flask import jsonify

filename = 'stud_performance.pkl'

loaded_model = pickle.load(open(filename,'rb'))

app = Flask('student_performance')
@app.route('/predict', methods=['POST'])

def predict():
    student = request.get_json()

    r2_score(y_test, y_pred_sel)
   
    result = {
        'prediction': float(y_pred),
    }

    return jsonify(result)

from flask import Flask

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


