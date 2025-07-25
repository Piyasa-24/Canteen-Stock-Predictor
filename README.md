# 🍽️ Canteen Stock Predictor

A Streamlit-based web app to predict how many cups/plates of an item should be prepared based on the day and item type.

## 📦 Features
- Simple UI with Streamlit
- Predicts demand based on past patterns
- Two inputs: Date & Item
- Trained using sample data (Linear Regression)

## 🧪 Run Locally

```bash
git clone https://github.com/yourusername/canteen-stock-predictor.git
cd canteen-stock-predictor
pip install -r requirements.txt
python train_model.py
streamlit run app.py
