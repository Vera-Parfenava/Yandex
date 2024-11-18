import numpy as np

def softmax(vector):

    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):

    # Шаг 1: Вычислить e_i для каждого состояния энкодера
    e_values = np.dot(decoder_hidden_state.T @ W_mult, encoder_hidden_states)
    
    # Убедимся, что e_values имеет нужную форму
    if e_values.ndim == 1:
        e_values = e_values.reshape(1, -1)
    
    # Шаг 2: Применить softmax для нормализации
    attention_weights = softmax(e_values)

    # Шаг 3: Итоговый attention vector — это взвешенная сумма скрытых состояний энкодера
    # Убедимся, что attention_weights имеет правильную форму для умножения
    attention_weights = attention_weights.reshape(-1, 1)
    attention_vector = np.sum(encoder_hidden_states * attention_weights.T, axis=1, keepdims=True)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):

    n_states = encoder_hidden_states.shape[1]
    
    # Преобразуем декодерное скрытое состояние
    dec_output = W_add_dec @ decoder_hidden_state  # (n_features_int, 1)

    # Инициализируем массив для хранения e_i значений
    e = np.zeros((n_states, 1))

    # Вычисляем e_i для каждого скрытого состояния кодировщика
    for i in range(n_states):
        enc_output = W_add_enc @ encoder_hidden_states[:, i].reshape(-1, 1)  # (n_features_int, 1)
        combined = enc_output + dec_output  # Сложение
        activation = np.tanh(combined)  # Применение активации tanh
        e[i] = v_add.T @ activation  # Скаляное произведение с v_add

    # Приведение e к форме (1, n) перед подачей в softmax
    e = e.reshape(1, -1)
    attention_weights = softmax(e).reshape(-1, 1)

    # Итоговый attention vector — это взвешенная сумма скрытых состояний энкодера
    attention_vector = np.dot(encoder_hidden_states, attention_weights)
    
    
    return attention_vector
