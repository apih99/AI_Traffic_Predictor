import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Generation (Improved)
def generate_data(num_data_points=10000):
    np.random.seed(42)
    traffic_density = np.random.rand(num_data_points)
    vehicle_speed = np.random.rand(num_data_points)
    weather_condition = np.random.rand(num_data_points)
    time_of_day = np.random.rand(num_data_points)
    
    # More realistic congestion level calculation
    congestion_level = (0.4 * traffic_density + 0.3 * (1 - vehicle_speed) + 
                        0.2 * weather_condition + 0.1 * np.sin(time_of_day * np.pi) > 0.5).astype(int)

    return pd.DataFrame({
        'traffic_density': traffic_density,
        'vehicle_speed': vehicle_speed,
        'weather_condition': weather_condition,
        'time_of_day': time_of_day,
        'congestion_level': congestion_level
    })

# Step 2: Data Preprocessing (Unchanged)
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = data.copy()
    scaled_data[['traffic_density', 'vehicle_speed', 'weather_condition', 'time_of_day']] = scaler.fit_transform(
        scaled_data[['traffic_density', 'vehicle_speed', 'weather_condition', 'time_of_day']])
    return scaled_data, scaler

# Step 3: Fuzzy Logic Implementation (Unchanged)
def fuzzify(value, low_threshold, high_threshold):
    return {
        'low': max(0, min(1, (low_threshold - value) / low_threshold)) if low_threshold > 0 else int(value == 0),
        'medium': max(0, min(1, (value - low_threshold) / (high_threshold - low_threshold), 
                              (high_threshold - value) / (high_threshold - low_threshold))) if high_threshold > low_threshold else int(value == 0.5),
        'high': max(0, min(1, (value - high_threshold) / (1 - high_threshold))) if high_threshold < 1 else int(value == 1)
    }

def fuzzy_inference(density, speed, weather, time):
    traffic_density = fuzzify(density, 0.3, 0.7)
    vehicle_speed = fuzzify(speed, 0.3, 0.7)
    weather_condition = fuzzify(weather, 0.3, 0.7)
    time_of_day = fuzzify(time, 0.3, 0.7)

    rules = [
        min(traffic_density['high'], vehicle_speed['low']),
        min(traffic_density['medium'], vehicle_speed['medium'], weather_condition['medium']),
        min(traffic_density['low'], vehicle_speed['high'], weather_condition['low']),
        min(time_of_day['high'], traffic_density['high']),
        min(weather_condition['high'], vehicle_speed['low'])
    ]

    high_congestion = max(rules[0], rules[3], rules[4])
    medium_congestion = rules[1]
    low_congestion = rules[2]

    total = high_congestion + medium_congestion + low_congestion
    if total == 0:
        return 0.5
    else:
        return (high_congestion * 2 + medium_congestion * 1 + low_congestion * 0) / total

# Step 4: Neural Network Training (Unchanged)
def create_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = create_model(features.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping], verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    
    return model, history, (accuracy, precision, recall, conf_matrix)

def neuro_fuzzy_predict(model, scaler, density, speed, weather, time):
    nn_input = scaler.transform([[density, speed, weather, time]])
    nn_output = model.predict(nn_input)[0][0]
    
    fuzzy_output = fuzzy_inference(density, speed, weather, time)
    
    final_prediction = (nn_output + fuzzy_output) / 2
    return final_prediction

# Gradio Interface
def predict_congestion(density, speed, weather, time):
    prediction = neuro_fuzzy_predict(model, scaler, density, speed, weather, time)
    congestion_level = "High" if prediction > 0.5 else "Low"
    return f"Predicted Congestion Level: {congestion_level} ({prediction:.2f})"

# Main execution
if __name__ == "__main__":
    # Generate and preprocess data
    data = generate_data()
    scaled_data, scaler = preprocess_data(data)
    
    # Train the model
    features = scaled_data[['traffic_density', 'vehicle_speed', 'weather_condition', 'time_of_day']]
    target = scaled_data['congestion_level']
    model, history, metrics = train_model(features, target)
    
    # Print model performance
    accuracy, precision, recall, conf_matrix = metrics
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_congestion,
        inputs=[
            gr.Slider(0, 1, label="Traffic Density (0-1)", info="Represents the density of traffic on the road."),
            gr.Slider(0, 1, label="Vehicle Speed (0-1)", info="Indicates the average speed of vehicles."),
            gr.Slider(0, 1, label="Weather Condition (0-1)", info="0: Clear, 0.5: Cloudy, 1: Rainy"),
            gr.Slider(0, 1, label="Time of Day (0-1)", info="Represents the time of day, from morning to night.")
        ],
        outputs=gr.Textbox(label="Predicted Congestion Level"),
        title="Traffic Congestion Predictor",
        description="Predict the level of traffic congestion based on input features."
    )
    
    # Launch the Gradio app
    interface.launch(share=True)
