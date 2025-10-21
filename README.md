Stock_Price_Prediction_NIfty_50 
# Next-Gen Forecasting: Applying Deep Learning to Time Series Data 💹

This project demonstrates the application of deep learning, specifically Long Short-Term Memory (LSTM) networks, for time series forecasting. The goal is to predict future NIFTY 50 stock index closing prices based on historical data.

This notebook showcases an end-to-end workflow for building an LSTM-based forecasting model, including data loading, preprocessing for time series (scaling and sequence creation), model architecture definition using Keras, training, evaluation, and visualization of predictions.

**Dataset:** `NIFTY 50.csv` (Historical daily data for the NIFTY 50 index, including Open, High, Low, Close, Volume, etc.)
**Focus:** Time series forecasting using LSTMs, data preprocessing (scaling, sequence generation), deep learning model building with Keras/TensorFlow, model training and evaluation (RMSE), visualizing predictions against actual values.
**Repository:** [https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data](https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`Stock_Price_Prediction_NIfty_50.ipynb`), the following key concepts and techniques are applied:

* **Time Series Forecasting:** Predicting future values based on historical sequential data.
* **Deep Learning for Time Series:** Utilizing Recurrent Neural Networks (RNNs), specifically LSTMs, to capture temporal dependencies.
* **Long Short-Term Memory (LSTM):** Implementing LSTM layers in Keras to model long-range dependencies in the stock price data.
* **Data Loading & Preparation:**
    * Loading time series data using Pandas.
    * Selecting the target variable (`Close` price).
    * Converting the dataset into a NumPy array.
* **Data Preprocessing for LSTMs:**
    * **Normalization:** Scaling the `Close` price data to the range [0, 1] using `MinMaxScaler` to improve model stability and performance.
    * **Sequence Generation:** Creating input sequences (using a lookback window, e.g., `time_step=100`) and corresponding target values (the next day's price) suitable for training sequence models like LSTMs.
* **Train-Test Split (Time Series):** Splitting the sequential data into training and testing sets while preserving the temporal order (training on earlier data, testing on later data).
* **Model Architecture (Keras):** Building a `Sequential` Keras model consisting of:
    * Multiple `LSTM` layers (with `return_sequences=True` for intermediate layers).
    * `Dropout` layers for regularization to prevent overfitting.
    * A final `Dense` layer with one output unit for predicting the scaled closing price.
* **Model Compilation:** Configuring the model using the `adam` optimizer and `mean_squared_error` loss function.
* **Model Training:** Fitting the LSTM model to the prepared training sequences (`X_train`, `y_train`), including validation using the test sequences (`X_test`, `y_test`).
* **Prediction & Evaluation:**
    * Making predictions on both the training and test datasets.
    * **Inverse Scaling:** Transforming the scaled predictions back to the original price scale using `scaler.inverse_transform`.
    * **Performance Metrics:** Calculating the Root Mean Squared Error (RMSE) to quantify prediction accuracy on the original scale.
* **Visualization:** Plotting the original `Close` prices against the model's training and testing predictions to visually assess the forecast quality.

***

## Analysis Workflow

The notebook follows a standard deep learning workflow for time series forecasting:

1.  **Setup & Data Loading:** Importing libraries (Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras) and loading the `NIFTY 50.csv` dataset.
2.  **Data Preparation & Preprocessing:**
    * Selecting the 'Close' price column.
    * Scaling the data using `MinMaxScaler`.
    * Splitting the scaled data into training and testing sets based on time.
    * Creating input sequences (`X_train`, `X_test`) and target values (`y_train`, `y_test`) using a sliding window approach (`create_dataset` function).
    * Reshaping input data to be compatible with LSTM layers (samples, time steps, features).
3.  **LSTM Model Architecture:** Defining a `Sequential` Keras model with stacked LSTM layers and Dropout.
4.  **Model Compilation:** Compiling the model with the Adam optimizer and Mean Squared Error loss.
5.  **Model Training:** Training the model on the training sequences, using the test sequences for validation during training.
6.  **Prediction & Inverse Scaling:** Generating predictions for both training and test sets and transforming them back to the original price scale.
7.  **Evaluation:** Calculating the RMSE for both training and test predictions.
8.  **Visualization:** Plotting the predicted prices against the actual historical prices for visual comparison and model assessment.

***

## Technologies Used

* **Python**
* **Pandas & NumPy:** For data loading, manipulation, array operations, and sequence creation.
* **Scikit-learn:** For data scaling (`MinMaxScaler`) and evaluation (`mean_squared_error`).
* **TensorFlow & Keras:** For building, compiling, training, and predicting with the LSTM model (`Sequential`, `LSTM`, `Dense`, `Dropout`).
* **Matplotlib:** For plotting the results (predictions vs. actuals).
* **Jupyter Notebook / Google Colab:** For the interactive development environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data.git](https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data.git)
    cd Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
    ```
3.  **Ensure Dataset:** Make sure the `NIFTY 50.csv` file is present in the repository directory.
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "Stock_Price_Prediction_NIfty_50.ipynb"
    ```
5.  **Run Cells:** Execute the cells sequentially. Model training (`model.fit`) may take some time depending on your hardware (especially if running on CPU) and the number of epochs.

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data](https://github.com/Jayasurya227/Next-Gen-Forecasting-Applying-Deep-Learning-to-Time-Series-Data)) demonstrates the application of deep learning (LSTMs) to solve a challenging time series forecasting problem (stock prediction). It showcases skills in data preprocessing for sequential data, building RNNs using Keras, and evaluating forecast accuracy. Suitable for GitHub, resumes/CVs, LinkedIn, and interviews for Data Scientist, AI/ML Engineer, or Quantitative Analyst roles.
* **Notes:** Recruiters can assess the understanding of time series data handling, LSTM model implementation, the importance of scaling and sequence generation, model training/evaluation procedures, and the ability to visualize and interpret forecasting results.
