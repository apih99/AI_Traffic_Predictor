# Traffic Congestion Predictor 🚗🚦

This project uses a combination of a neural network and fuzzy logic to predict traffic congestion levels based on various input features.  It provides a user-friendly interface built with Gradio, allowing users to interactively explore the predictions. ✨

## Features

* **Neuro-Fuzzy System:** Combines the strengths of neural networks and fuzzy logic for a potentially more robust prediction. 🧠 + 🤔 = 💪
* **Interactive Interface:** The Gradio interface allows users to adjust input features (traffic density, vehicle speed, weather condition, time of day) and see the predicted congestion level in real-time. 🖱️➡️📊
* **Realistic Data Generation:** The synthetic data generation process uses a more nuanced formula to simulate real-world traffic conditions. 🌍➡️💻
* **Model Evaluation:** The trained neural network model's performance is evaluated using metrics like accuracy, precision, recall, and confusion matrix. 📏📈
* **Early Stopping:** Prevents overfitting during neural network training. 🛑🚫

## How it Works

1. **Data Generation:** Synthetic traffic data is generated, including features like traffic density, vehicle speed, weather condition, time of day, and the resulting congestion level. 📊🏭
2. **Data Preprocessing:** The input features are scaled using MinMaxScaler to a range of 0-1. 🔢➡️0️⃣-1️⃣
3. **Fuzzy Logic System:** A fuzzy logic system is implemented to provide an initial estimate of the congestion level based on predefined fuzzy rules. 🤔💭
4. **Neural Network Training:** A neural network is trained on the preprocessed data to learn the relationship between the input features and the congestion level. 🧠🏋️
5. **Neuro-Fuzzy Prediction:** The predictions from the neural network and the fuzzy logic system are combined (averaged) to produce the final congestion level prediction. 🧠🤝🤔
6. **Gradio Interface:** The user interacts with the Gradio interface, providing input feature values. The application then uses the trained model and fuzzy system to predict the congestion level and display it to the user.  🧑‍💻↔️🤖

## Getting Started 🚀

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/traffic-congestion-predictor.git

   ```markdown
# Traffic Congestion Predictor 🚗🚦

This project uses a combination of a neural network and fuzzy logic to predict traffic congestion levels based on various input features.  It provides a user-friendly interface built with Gradio, allowing users to interactively explore the predictions. ✨

## Features

* **Neuro-Fuzzy System:** Combines the strengths of neural networks and fuzzy logic for a potentially more robust prediction. 🧠 + 🤔 = 💪
* **Interactive Interface:** The Gradio interface allows users to adjust input features (traffic density, vehicle speed, weather condition, time of day) and see the predicted congestion level in real-time. 🖱️➡️📊
* **Realistic Data Generation:** The synthetic data generation process uses a more nuanced formula to simulate real-world traffic conditions. 🌍➡️💻
* **Model Evaluation:** The trained neural network model's performance is evaluated using metrics like accuracy, precision, recall, and confusion matrix. 📏📈
* **Early Stopping:** Prevents overfitting during neural network training. 🛑🚫

## How it Works

1. **Data Generation:** Synthetic traffic data is generated, including features like traffic density, vehicle speed, weather condition, time of day, and the resulting congestion level. 📊🏭
2. **Data Preprocessing:** The input features are scaled using MinMaxScaler to a range of 0-1. 🔢➡️0️⃣-1️⃣
3. **Fuzzy Logic System:** A fuzzy logic system is implemented to provide an initial estimate of the congestion level based on predefined fuzzy rules. 🤔💭
4. **Neural Network Training:** A neural network is trained on the preprocessed data to learn the relationship between the input features and the congestion level. 🧠🏋️
5. **Neuro-Fuzzy Prediction:** The predictions from the neural network and the fuzzy logic system are combined (averaged) to produce the final congestion level prediction. 🧠🤝🤔
6. **Gradio Interface:** The user interacts with the Gradio interface, providing input feature values. The application then uses the trained model and fuzzy system to predict the congestion level and display it to the user.  🧑‍💻↔️🤖

## Getting Started 🚀

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/traffic-congestion-predictor.git
   ```

2. **Create and activate a virtual environment:** (Recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate  # Windows
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py  # Or the name of your main Python file
   ```

This will launch the Gradio interface. You can then interact with the sliders to adjust the input features and see the predicted congestion level. 🎉

```markdown
# Traffic Congestion Predictor

This project uses a combination of a neural network and fuzzy logic to predict traffic congestion levels based on various input features.  It provides a user-friendly interface built with Gradio, allowing users to interactively explore the predictions.

## Features

* **Neuro-Fuzzy System:** Combines the strengths of neural networks and fuzzy logic for a potentially more robust prediction.
* **Interactive Interface:**  The Gradio interface allows users to adjust input features (traffic density, vehicle speed, weather condition, time of day) and see the predicted congestion level in real-time.
* **Realistic Data Generation:**  The synthetic data generation process uses a more nuanced formula to simulate real-world traffic conditions.
* **Model Evaluation:**  The trained neural network model's performance is evaluated using metrics like accuracy, precision, recall, and confusion matrix.
* **Early Stopping:** Prevents overfitting during neural network training.

## How it Works

1. **Data Generation:**  Synthetic traffic data is generated, including features like traffic density, vehicle speed, weather condition, time of day, and the resulting congestion level.
2. **Data Preprocessing:**  The input features are scaled using MinMaxScaler to a range of 0-1.
3. **Fuzzy Logic System:** A fuzzy logic system is implemented to provide an initial estimate of the congestion level based on predefined fuzzy rules.
4. **Neural Network Training:** A neural network is trained on the preprocessed data to learn the relationship between the input features and the congestion level.
5. **Neuro-Fuzzy Prediction:**  The predictions from the neural network and the fuzzy logic system are combined (averaged) to produce the final congestion level prediction.
6. **Gradio Interface:**  The user interacts with the Gradio interface, providing input feature values. The application then uses the trained model and fuzzy system to predict the congestion level and display it to the user.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/traffic-congestion-predictor.git
   ```

2. **Create and activate a virtual environment:** (Recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate  # Windows
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py  # Or the name of your main Python file
   ```

This will launch the Gradio interface. You can then interact with the sliders to adjust the input features and see the predicted congestion level.

## Project Structure

* `AI_Traffic_Predictor.py` (or your main file name): Contains the Python code for data generation, preprocessing, fuzzy logic, neural network training, and the Gradio interface.
* `requirements.txt`: Lists the required Python packages.
* `README.md`: This file.


## Future Enhancements

* **Real-World Data:** Train the model on real-world traffic data for improved accuracy.
* **More Sophisticated Fuzzy Rules:**  Refine the fuzzy logic system with more complex rules or fuzzy inference methods.
* **Hyperparameter Tuning:** Optimize the neural network architecture and training parameters.
* **Explainability:** Provide insights into how the fuzzy system and neural network arrive at their predictions.
* **Deployment:** Deploy the model as a web service or API for real-time predictions.


## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.



   
