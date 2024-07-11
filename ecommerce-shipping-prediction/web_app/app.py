import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("models/rf_acc_68.pkl", "rb"))
scaler = pickle.load(open("models/normalizer.pkl", "rb"))

@app.route('/')
def input():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            warehouse_block = request.form["Warehouse_block"]
            mode_of_shipment = request.form["Mode_of_Shipment"]
            customer_care_calls = float(request.form["Customer_care_calls"])
            customer_rating = float(request.form["Customer_rating"])
            cost_of_the_product = float(request.form["Cost_of_the_Product"])
            prior_purchases = float(request.form["Prior_purchases"])
            product_importance = request.form["Product_importance"]
            gender = request.form["Gender"]
            discount_offered = float(request.form["Discount_offered"])
            weight_in_gms = float(request.form["Weight_in_gms"])

            # Combine all features into a single array
            features = [
                warehouse_block,
                mode_of_shipment,
                customer_care_calls,
                customer_rating,
                cost_of_the_product,
                prior_purchases,
                product_importance,
                gender,
                discount_offered,
                weight_in_gms
            ]

            # Convert categorical features using LabelEncoder
            le = LabelEncoder()
            features[0] = le.fit_transform([features[0]])[0]
            features[1] = le.fit_transform([features[1]])[0]
            features[6] = le.fit_transform([features[6]])[0]
            features[7] = le.fit_transform([features[7]])[0]

            # Normalize the features
            features_normalized = scaler.transform([features])

            # Predict
            prediction = model.predict(features_normalized)
            probability = model.predict_proba(features_normalized)[0]

            reach = probability[1]
            return render_template("index.html", p=f'There is a {reach * 100:.2f}% chance that your product will reach in time')

        except Exception as e:
            return str(e)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=4000)
