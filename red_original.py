model_2 = Sequential([
    Dense(100, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid'),
])