# Vitalis IA - Predicción de Riesgo Metabólico 🧬

Este proyecto implementa una Red Neuronal Artificial (Perceptrón Multicapa) desarrollada en **PyTorch** para predecir la incidencia de obesidad en la población laboral colombiana, utilizando microdatos de la GEIH (DANE). 

El modelo aborda el desbalance poblacional mediante entropía cruzada binaria ponderada (`BCEWithLogitsLoss`) y se encuentra desplegado en la nube a través de una API construida con **Flask** y un dashboard interactivo en **JavaScript**.

## Arquitectura del Proyecto
* **Machine Learning:** PyTorch, Scikit-Learn, Pandas.
* **Backend:** Python (Flask), Gunicorn.
* **Frontend:** HTML5, TailwindCSS, Plotly.js.

## Características
1. **Dashboard Estadístico:** Análisis demográfico y geográfico interactivo.
2. **Motor de Inferencia:** Formulario clínico para predicción en tiempo real.
3. **Manejo de Desbalances:** Priorización del recall clínico para minimizar falsos negativos.