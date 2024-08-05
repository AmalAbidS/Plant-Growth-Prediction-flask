# app.py

from flask import Flask, request, render_template
import torch
import joblib
from model import PlantGrowthModel  # Import the model class

app = Flask(__name__)

# Load the model and encoders
model = PlantGrowthModel(input_dim=6, output_dim=2)  # Update with actual dimensions
model.load_state_dict(torch.load('plant_growth_model.pth'))
model.eval()

label_encoders = joblib.load('label_encoders.pkl')
le_growth = joblib.load('le_growth.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    soil_type = request.form.get('soil_type')
    sunlight_hours = request.form.get('sunlight_hours')
    water_frequency = request.form.get('water_frequency')
    fertilizer_type = request.form.get('fertilizer_type')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    
    # Prepare the input data
    feature_dict = {
        'Soil_Type': soil_type,
        'Sunlight_Hours': sunlight_hours,
        'Water_Frequency': water_frequency,
        'Fertilizer_Type': fertilizer_type,
        'Temperature': temperature,
        'Humidity': humidity
    }
    
    processed_features = []
    for col in ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Fertilizer_Type', 'Temperature', 'Humidity']:
        if col in label_encoders:
            encoder = label_encoders[col]
            if feature_dict[col] in encoder.classes_:
                processed_features.append(encoder.transform([feature_dict[col]])[0])
            else:
                # Handle unseen label by setting to a default value or some appropriate handling
                processed_features.append(encoder.transform([encoder.classes_[0]])[0])
        else:
            processed_features.append(float(feature_dict[col]))
    
    features_tensor = torch.tensor([processed_features], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(features_tensor)
    
    prediction = torch.argmax(output, dim=1).item()
    growth_stage = le_growth.inverse_transform([prediction])[0]
    
    return render_template('index.html', prediction_text=f"Predicted Growth Milestone: {growth_stage}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
