from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from plotly import graph_objects as go
import plotly.io as pio
import os
import base64



app = Flask(__name__)

# Function to prepare data
@app.template_filter('base64')
def base64_filter(data):
    return base64.b64encode(data.encode('utf-8')).decode('utf-8')

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
    plot_div = None
    
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
        
        # Create a Plotly graph
        fig = go.Figure()

        # Original sequence plot
        fig.add_trace(go.Scatter(
            x=list(range(len(sequence))),
            y=sequence,
            mode='lines+markers',
            name='Original Sequence',
            line=dict(color='blue')
        ))

        # Predicted sequence plot
        fig.add_trace(go.Scatter(
            x=list(range(len(sequence), len(sequence) + n_steps_out)),
            y=predicted_sequence,
            mode='lines+markers',
            name='Predicted Sequence',
            line=dict(color='red', dash='dash')
        ))

        # Add labels and title
        fig.update_layout(
            title="Original vs Predicted Sequence",
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=True
        )

        # Generate HTML div for Plotly chart
        plot_div = pio.to_html(fig, full_html=False)

    return render_template("index.html", sequence=input_sequence_str, predicted_sequence=predicted_sequence, plot_div=plot_div)

if __name__ == "__main__":
    app.run(debug=True)
