from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle

app = Flask(__name__)

# ==========================================
# 1. ARQUITECTURA DE LA RED
# ==========================================
class RedPredictiva(nn.Module):
    def __init__(self, input_dim=3):
        super(RedPredictiva, self).__init__()
        self.capa1 = nn.Linear(input_dim, 16)
        self.capa2 = nn.Linear(16, 8)
        self.salida = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        return self.salida(x)

# ==========================================
# 2. CARGAR MEMORIA DE LA IA
# ==========================================
modelo = RedPredictiva(input_dim=3)
modelo.load_state_dict(torch.load('modelo_obesidad_mlp.pth', map_location=torch.device('cpu'), weights_only=True))
modelo.eval()

with open('escalador.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ==========================================
# 3. RUTAS WEB (API)
# ==========================================
# Ruta principal que muestra tu diseño HTML
@app.route('/')
def home():
    return render_template('index.html')

# Ruta oculta que procesa la matemática cuando JS le manda datos
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 3.1 Recibir datos de JS
    edad = float(data['edad'])
    genero = float(data['genero'])
    sector = float(data['sector'])
    
    # 3.2 Pasar por el escalador (Mismo orden que en Colab)
    datos_crudos = np.array([[edad, genero, sector]])
    datos_escalados = scaler.transform(datos_crudos)
    tensor_entrada = torch.tensor(datos_escalados, dtype=torch.float32)
    
    # 3.3 Predecir con Deep Learning
    with torch.no_grad():
        salida_cruda = modelo(tensor_entrada)
        probabilidad = torch.sigmoid(salida_cruda).item()
        
    # Devolver porcentaje a la web
    return jsonify({
        'probabilidad': probabilidad * 100,
        'alerta': True if probabilidad >= 0.5 else False
    })

if __name__ == '__main__':
    app.run(debug=True)