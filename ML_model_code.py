
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('DataSet.csv')

# Data Preprocessing
X = df[['Load (in KN)', 'Moment in x (in Knm)', 'Moment in y (in Knm)', 'Length (in m)']]
y = df[['Diameter of Pile', 'Area of Steel (mm^2)', 'Pitch (in mm)']]

# Split of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_cv_scaled = scaler_X.transform(X_cv)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_cv_scaled = scaler_y.transform(y_cv)
y_test_scaled = scaler_y.transform(y_test)

# Model architecture
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),  # Additional hidden layer
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(y_train.shape[1])
])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')


history = model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=32,
                    validation_data=(X_cv_scaled, y_cv_scaled), verbose=1)

# Predictions on the test set
predictions_scaled = model.predict(X_test_scaled)

# Inverse transform the predictions to the original scale
predictions = scaler_y.inverse_transform(predictions_scaled)

# Evaluate the model
r2_scores = []
explained_variances = []
accuracy_levels = [0.1, 0.08, 0.05, 0.04, 0.03, 0.02]

for tol in accuracy_levels:
    within_tolerance = np.all(np.abs((predictions - y_test.values) / y_test.values) < tol, axis=1)

    # Overall accuracy
    accuracy_overall = accuracy_score(np.ones_like(within_tolerance), within_tolerance) * 100

    # Individual output metrics
    r2_scores_individual = [r2_score(y_test[output], predictions[:, i]) for i, output in enumerate(y.columns)]
    explained_variances_individual = [explained_variance_score(y_test[output], predictions[:, i]) for i, output in enumerate(y.columns)]

    # Append to the lists
    r2_scores.append(r2_scores_individual)
    explained_variances.append(explained_variances_individual)

    print(f'Tolerance Level: ±{tol * 100}%')
    print(f'Overall Accuracy: {accuracy_overall:.2f}%')
    print('Individual R2 Scores:')
    for i, output in enumerate(y.columns):
        print(f'{output}: {r2_scores_individual[i]:.4f}')
    print('Individual Explained Variances:')
    for i, output in enumerate(y.columns):
        print(f'{output}: {explained_variances_individual[i]:.4f}')
    print('\n')

    # Visualize the distribution of predictions within tolerance using a pie chart
    within_tolerance_count = np.sum(within_tolerance)
    not_within_tolerance_count = len(within_tolerance) - within_tolerance_count
    labels = [f'Within Tolerance ({within_tolerance_count})', f'Not Within Tolerance ({not_within_tolerance_count})']
    sizes = [within_tolerance_count, not_within_tolerance_count]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution of Predictions within ±{tol * 100}% Tolerance')
    plt.show()

    # Plot histograms for predicted and actual values of each output parameter
    for i, output in enumerate(y.columns):
        plt.figure(figsize=(12, 6))

        # Histogram for actual values
        plt.subplot(2, 3, 1)
        plt.hist(y_test[output], bins=30, color='blue', alpha=0.7, label='Actual Values')
        plt.title(f'Histogram of Actual {output}')
        plt.xlabel(output)
        plt.ylabel('Frequency')
        plt.legend()

        # Histogram for predicted values
        plt.subplot(2, 3, 2)
        plt.hist(predictions[:, i], bins=30, color='orange', alpha=0.7, label='Predicted Values')
        plt.title(f'Histogram of Predicted {output}')
        plt.xlabel(output)
        plt.ylabel('Frequency')
        plt.legend()

        # Scatter plot of actual vs predicted values
        plt.subplot(2, 3, 3)
        plt.scatter(y_test[output], predictions[:, i], color='green', alpha=0.5)
        plt.title(f'Actual vs Predicted {output}')
        plt.xlabel(f'Actual {output}')
        plt.ylabel(f'Predicted {output}')

        plt.show()

# Bar chart for individual output metrics
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(y.columns))
opacity = 0.8

for i, output in enumerate(y.columns):
    ax.bar(index + i * bar_width, r2_scores[-1][i], bar_width, alpha=opacity, label=f'R2 - {output}')

ax.set_xlabel('Output Parameters')
ax.set_ylabel('R2 Score')
ax.set_title('Individual R2 Scores for Output Parameters')
ax.set_xticks(index + bar_width * (len(accuracy_levels) - 1) / 2)
ax.set_xticklabels(y.columns)
ax.legend()

plt.show()

# Loss curve during training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training and Validation Loss Curve')
plt.show()
