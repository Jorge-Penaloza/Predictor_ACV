import tkinter as tk
from tkinter import ttk
from keras.models import load_model 
import joblib
import numpy as np

# Cargar el modelo y MinMaxScaler
model = load_model('my_model.h5')
min_max_scaler = joblib.load('min_max_scaler.pkl')

def submit():
    data = [
        int(gender_cb.get()[0]),  # Extraemos el primer carácter y lo convertimos a int
        float(age_entry.get()),
        int(hypertension_entry.get()),
        int(heart_disease_cb.get()),
        int(ever_married_cb.get()[0]),  # Extraemos el primer carácter y lo convertimos a int
        int(work_type_cb.get()[0]),  # Extraemos el primer carácter y lo convertimos a int
        int(residence_type_cb.get()[0]),  # Extraemos el primer carácter y lo convertimos a int
        float(avg_glucose_level_entry.get()),  # Aquí asumo que estos pueden ser valores de punto flotante.
        float(bmi_entry.get()),  # Aquí asumo que estos pueden ser valores de punto flotante.
        int(smoking_status_cb.get()[0])  # Extraemos el primer carácter y lo convertimos a int
    ]
    inputs = data
    # Asume que las entradas son unidimensionales
    inputs = np.array(inputs).reshape(1, -1)

    # Normalizar las entradas
    inputs = min_max_scaler.transform(inputs)

    # Hacer una predicción con el modelo
    predictions = model.predict(inputs)
    print(predictions)
    result_label['text'] = str(data) + "  Dato final =  " + str(predictions[0][0] )


root = tk.Tk()

# Creación de widgets
gender_label = tk.Label(root, text="Gender")
gender_cb = ttk.Combobox(root, values=['0-Male', '1-Female', '2-Other'])

age_label = tk.Label(root, text="Age")
age_entry = tk.Entry(root)

hypertension_label = tk.Label(root, text="Hypertension")
hypertension_entry = tk.Entry(root)

heart_disease_label = tk.Label(root, text="Heart Disease")
heart_disease_cb = ttk.Combobox(root, values=['0', '1'])

ever_married_label = tk.Label(root, text="Ever Married")
ever_married_cb = ttk.Combobox(root, values=['0-No', '1-Yes'])

work_type_label = tk.Label(root, text="Work Type")
work_type_cb = ttk.Combobox(root, values=['0-Never_worked', '1-children', '2-Govt_job', '3-Private', '4-Self-employed'])

residence_type_label = tk.Label(root, text="Residence Type")
residence_type_cb = ttk.Combobox(root, values=['0-Urban', '1-Rural'])

avg_glucose_level_label = tk.Label(root, text="Avg Glucose Level")
avg_glucose_level_entry = tk.Entry(root)

bmi_label = tk.Label(root, text="BMI")
bmi_entry = tk.Entry(root)

smoking_status_label = tk.Label(root, text="Smoking Status")
smoking_status_cb = ttk.Combobox(root, values=['0-never smoked', '1-smokes', '2-unknowns', '3-formerly smoked'])

submit_button = tk.Button(root, text="Submit", command=submit)
result_label = tk.Label(root, text="")

# Organización de widgets con grid()
rows = [
    (gender_label, gender_cb),
    (age_label, age_entry),
    (hypertension_label, hypertension_entry),
    (heart_disease_label, heart_disease_cb),
    (ever_married_label, ever_married_cb),
    (work_type_label, work_type_cb),
    (residence_type_label, residence_type_cb),
    (avg_glucose_level_label, avg_glucose_level_entry),
    (bmi_label, bmi_entry),
    (smoking_status_label, smoking_status_cb)
]

for i, (label, entry) in enumerate(rows):
    label.grid(row=i, column=0, sticky='e')
    entry.grid(row=i, column=1)

submit_button.grid(row=10, column=0, columnspan=2)
result_label.grid(row=11, column=0, columnspan=2)

root.mainloop()

