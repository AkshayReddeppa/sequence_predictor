from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO
import os

app = Flask(__name__)

# Function to prepare data
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps], sequence[i + n_steps]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to build and train the model
def build_and_train_model(sequence, n_steps):
    X, y = prepare_data(sequence, n_steps)
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict the next sequence
def predict_sequence(model, input_sequence, n_steps_out):
    predictions = []
    current_input = list(input_sequence)
    for _ in range(n_steps_out):
        next_number = model.predict([current_input])[0]
        next_number = round(next_number)
        predictions.append(next_number)
        current_input.pop(0)
        current_input.append(next_number)
    return predictions

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_sequence = None
    input_sequence_str = ""
    sequence = []
    
    if request.method == "POST":
        # Get user input from form
        input_sequence_str = request.form.get("sequence")
        n_steps_out = int(request.form.get("n_steps_out"))
        
        # Convert input sequence to a list of integers
        sequence = list(map(int, input_sequence_str.split(',')))
        n_steps_in = 3  # You can set this to a different value or make it dynamic

        # Build and train the model
        model = build_and_train_model(sequence, n_steps_in)
        
        # Prepare input for prediction
        input_sequence = sequence[-n_steps_in:]
        predicted_sequence = predict_sequence(model, input_sequence, n_steps_out)
        
        # Generate the plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(sequence)), sequence, label="Original Sequence", marker='o', linestyle='-', color='blue')
        plt.plot(range(len(sequence), len(sequence) + n_steps_out), predicted_sequence, label="Predicted Sequence", marker='x', linestyle='--', color='red')
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Original vs Predicted Sequence")
        plt.legend()
        plt.grid(True)
        
        # Save the plot as a static image
        plot_path = os.path.join("static", "plot.png")
        plt.savefig(plot_path)
        plt.close()

    return render_template("index.html", sequence=input_sequence_str, predicted_sequence=predicted_sequence)

if __name__ == "__main__":
    app.run(debug=True)
