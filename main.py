import numpy as np

# Функция активации (сигмоида) и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 1. Подготовка данных (задача XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Инициализация весов
np.random.seed(1)
weights_hidden = np.random.uniform(size=(2, 4))
weights_output = np.random.uniform(size=(4, 1))

# 3. Обучение нейросети
epochs = 10000
learning_rate = 0.1

print("Начинаем обучение...")

for epoch in range(epochs):
    # Прямое распространение
    hidden_layer_output = sigmoid(np.dot(X, weights_hidden))
    predicted_output = sigmoid(np.dot(hidden_layer_output, weights_output))

    # Обратное распространение
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Обновляем веса
    weights_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_hidden += X.T.dot(d_hidden_layer) * learning_rate

    if (epoch % 1000) == 0:
        print(f"Ошибка после {epoch} эпох: {np.mean(np.abs(error))}")

print("\nОбучение завершено!")

# 4. Тестирование сети
print("\nРезультаты после обучения:")
print("Предсказанные значения (округленные):")
print(np.round(predicted_output))
