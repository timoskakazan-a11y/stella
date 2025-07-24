from flask import Flask, request, jsonify, render_template
import numpy as np

# --- Код нашей нейросети, упакованный в функции ---

# Функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Модель нейросети (веса)
# Мы "обучим" её один раз при запуске сервера
weights_hidden = None
weights_output = None

def train_network():
    """Обучает нейросеть и сохраняет веса в глобальные переменные."""
    global weights_hidden, weights_output
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    np.random.seed(1)
    weights_hidden = np.random.uniform(size=(2, 4))
    weights_output = np.random.uniform(size=(4, 1))
    
    epochs = 10000
    learning_rate = 0.1

    for _ in range(epochs):
        hidden_layer_output = sigmoid(np.dot(X, weights_hidden))
        predicted_output = sigmoid(np.dot(hidden_layer_output, weights_output))

        error = y - predicted_output
        d_predicted_output = error * (predicted_output * (1 - predicted_output))
        
        error_hidden_layer = d_predicted_output.dot(weights_output.T)
        d_hidden_layer = error_hidden_layer * (hidden_layer_output * (1 - hidden_layer_output))

        weights_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_hidden += X.T.dot(d_hidden_layer) * learning_rate
    
    print("Нейросеть обучена и готова к работе!")

def predict(input_data):
    """Делает предсказание на основе обученных весов."""
    input_array = np.array([input_data])
    hidden_layer_output = sigmoid(np.dot(input_array, weights_hidden))
    predicted_output = sigmoid(np.dot(hidden_layer_output, weights_output))
    return predicted_output[0][0]


# --- Создание веб-сервера на Flask ---

app = Flask(__name__)

# Главная страница, которая будет отображать наш чат (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Эндпоинт для чата, который будет принимать запросы от JavaScript
@app.route('/chat', methods=['POST'])
def chat():
    # Получаем данные, которые прислал JavaScript
    user_input = request.json.get('message')

    try:
        # Пытаемся обработать ввод пользователя (например, "1,0")
        parts = [int(p.strip()) for p in user_input.split(',')]
        if len(parts) != 2 or not all(p in [0, 1] for p in parts):
            raise ValueError("Неверный формат ввода.")

        # Делаем предсказание с помощью нашей нейросети
        prediction = predict(parts)
        result = round(prediction)

        # Формируем ответ
        bot_response = f"Для входа [{parts[0]}, {parts[1]}] я думаю, что результат будет: {result}. (Точное значение: {prediction:.3f})"
    
    except Exception:
        bot_response = "Пожалуйста, введите два числа (0 или 1) через запятую. Например: 1, 0"

    return jsonify({'response': bot_response})


# --- Запуск всего ---
if __name__ == '__main__':
    # Сначала обучаем нашу нейросеть
    train_network()
    # Затем запускаем веб-сервер
    app.run(host='0.0.0.0', port=5000)
