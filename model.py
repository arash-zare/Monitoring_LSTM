from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    """
    ورودی: input_shape به شکل (timesteps, features)
    خروجی: مدل LSTM کامپایل‌شده
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model



# 🧠 منطق:

#     از دو لایه LSTM استفاده شده: اولی خروجی sequence می‌ده برای بعدی، دومی به‌صورت خلاصه.

#     Dropout برای جلوگیری از overfitting.

#     خروجی یک عدد هست (پیش‌بینی مقدار بعدی در سری زمانی).

#     مدل برای تشخیص anomaly بر اساس اختلاف پیش‌بینی‌شده و واقعی عمل می‌کنه.