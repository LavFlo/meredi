import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Δεδομένα
data = {
    'Year': [2020, 2020, 2020, 2020,
             2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021,
             2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022,
             2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
             2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
    'Month': ['Σεπ', 'Οκτ', 'Νοε', 'Δεκ',
              'Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαϊ', 'Ιουν', 'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ',
              'Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαϊ', 'Ιουν', 'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ',
              'Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαϊ', 'Ιουν', 'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ',
              'Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαϊ', 'Ιουν', 'Ιουλ', 'Αυγ','Σεπ', 'Οκτ', 'Νοε', 'Δεκ'],
    'Rain_mm': [0.2, 65, 0, 158.4,
                94.4, 70, 36.8, 7.2, 0, 7.4, 0, 0, 1, 70.2, 64.8, 166.2,
                129, 73, 40.4, 45.4, 1.8, 0, 0.4, 41.4, 0.4, 81.8, 109.8, 54.6,
                131.6, 17, 32.2, 26.6, 14.8, 11.2, 0, 20, 93.2, 2.6, 104.2, 158,
                84.8, 50.8, 23.4, 11.2, 1.2, 0.0, 0.0, 13.6, 19.4, 0.6, 152.8, 125.6]
}

# Δημιουργία του DataFrame
df = pd.DataFrame(data)

# Μετατροπή των μηνών σε αριθμούς για να μπορέσουμε να τα χρησιμοποιήσουμε στη γραμμική παλινδρόμηση
month_map = {'Ιαν': 1, 'Φεβ': 2, 'Μαρ': 3, 'Απρ': 4, 'Μαϊ': 5, 'Ιουν': 6, 'Ιουλ': 7, 'Αυγ': 8,
             'Σεπ': 9, 'Οκτ': 10, 'Νοε': 11, 'Δεκ': 12}
df['Month_num'] = df['Month'].map(month_map)

# Δημιουργία του χαρακτηριστικού X και του στόχου y για τη γραμμική παλινδρόμηση
X = df[['Year', 'Month_num']]  # Χρόνια και μήνες ως χαρακτηριστικά
y = df['Rain_mm']  # Η βροχόπτωση ως στόχος

# Εκπαίδευση του μοντέλου γραμμικής παλινδρόμησης
model = LinearRegression()
model.fit(X, y)

# Προβλέψεις για το μέλλον (π.χ. 2025)
future_years_months = pd.DataFrame({
    'Year': [2025] * 12,
    'Month_num': list(range(1, 13))
})

future_rain = model.predict(future_years_months)

# Δημιουργία γραφήματος με τα πραγματικά δεδομένα και τις προβλέψεις
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Year'] + (df['Month_num'] - 1) / 12, df['Rain_mm'], label='Πραγματική Βροχόπτωση', color='blue', marker='o')
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain, label='Προβλεπόμενη Βροχόπτωση 2025', color='red', linestyle='--', marker='x')
ax.set_xlabel('Έτος')
ax.set_ylabel('Βροχόπτωση (mm)')
ax.set_title('Προβλέψεις Βροχόπτωσης για το 2025')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# Δημιουργία του μοντέλου πολυωνυμικής γραμμικής παλινδρόμησης
degree = 2  # Πολυωνυμικός βαθμός
model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_poly.fit(X, y)

# Προβλέψεις για το μέλλον (2025)
future_rain_poly = model_poly.predict(future_years_months)

# Δημιουργία γραφήματος με τις προβλέψεις πολυωνυμικής παλινδρόμησης
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Year'] + (df['Month_num'] - 1) / 12, df['Rain_mm'], label='Πραγματική Βροχόπτωση', color='blue', marker='o')
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain_poly, label='Προβλεπόμενη Βροχόπτωση 2025 (Πολυωνυμική)', color='green', linestyle='--', marker='x')
ax.set_xlabel('Έτος')
ax.set_ylabel('Βροχόπτωση (mm)')
ax.set_title('Προβλέψεις Βροχόπτωσης για το 2025 (Πολυωνυμική)')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# Δημιουργία του μοντέλου Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# Προβλέψεις για το μέλλον (2025)
future_rain_rf = model_rf.predict(future_years_months)

# Δημιουργία γραφήματος με τις προβλέψεις του Random Forest
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Year'] + (df['Month_num'] - 1) / 12, df['Rain_mm'], label='Πραγματική Βροχόπτωση', color='blue', marker='o')
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain_rf, label='Προβλεπόμενη Βροχόπτωση 2025 (Random Forest)', color='purple', linestyle='--', marker='x')
ax.set_xlabel('Έτος')
ax.set_ylabel('Βροχόπτωση (mm)')
ax.set_title('Προβλέψεις Βροχόπτωσης για το 2025 (Random Forest)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Δημιουργία γραφήματος με πραγματικά δεδομένα και προβλέψεις για το 2025
fig, ax = plt.subplots(figsize=(10, 6))

# Πραγματικά δεδομένα
ax.plot(df['Year'] + (df['Month_num'] - 1) / 12, df['Rain_mm'], label='Πραγματική Βροχόπτωση', color='blue', marker='o')

# Προβλέψεις για το 2025
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain, label='Προβλεπόμενη Βροχόπτωση 2025 (Linear Regression)', color='red', linestyle='--', marker='x')
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain_poly, label='Προβλεπόμενη Βροχόπτωση 2025 (Πολυωνυμική)', color='green', linestyle='--', marker='x')
ax.plot(2025 + (future_years_months['Month_num'] - 1) / 12, future_rain_rf, label='Προβλεπόμενη Βροχόπτωση 2025 (Random Forest)', color='purple', linestyle='--', marker='x')

# Ρυθμίσεις γραφήματος
ax.set_xlabel('Έτος')
ax.set_ylabel('Βροχόπτωση (mm)')
ax.set_title('Πραγματική Βροχόπτωση και Προβλέψεις για το 2025 με την Πάροδο του Χρόνου')
ax.legend()
ax.grid(True)

# Εμφάνιση γραφήματος στο Streamlit
st.pyplot(fig)

# Δημιουργία γραφήματος για την επίδραση του μήνα στις προβλέψεις
fig, ax = plt.subplots(figsize=(10, 6))

# Μήνες του 2025
months_2025 = future_years_months['Month_num']

# Προβλέψεις για το 2025 από τα τρία μοντέλα
ax.plot(months_2025, future_rain, label='Linear Regression', color='red', linestyle='--', marker='o')
ax.plot(months_2025, future_rain_poly, label='Πολυωνυμική Παλίνδρομηση', color='green', linestyle='--', marker='x')
ax.plot(months_2025, future_rain_rf, label='Random Forest', color='purple', linestyle='--', marker='s')

# Ρυθμίσεις γραφήματος
ax.set_xlabel('Μήνας')
ax.set_ylabel('Βροχόπτωση (mm)')
ax.set_title('Επίδραση του Μήνα στις Προβλέψεις για το 2025')
ax.legend()
ax.grid(True)

# Εμφάνιση γραφήματος στο Streamlit
st.pyplot(fig)




# Δημιουργία σελίδας Streamlit
st.title('Προβλέψεις Βροχόπτωσης για το 2025')
st.subheader('Πραγματική Βροχόπτωση και Προβλέψη για το 2025')


# Γράφημα και πίνακας για τα πραγματικά δεδομένα και τις προβλέψεις
fig, ax = plt.subplots(figsize=(10, 6))
# Πραγματικά δεδομένα
ax.scatter(df['Month_num'], df['Rain_mm'], color='blue', label='Actual')
# Προβλέψεις
ax.plot(future_years_months['Month_num'], future_rain, color='red', label='Predictions')
ax.set_xlabel("Month")
ax.set_ylabel("Rainfall (mm)")
ax.set_title("Rainfall Predictions for 2025")
ax.legend()
st.pyplot(fig)
