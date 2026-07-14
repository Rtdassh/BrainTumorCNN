# Brain Tumor CNN Classification Project

Este proyecto implementa un sistema robusto de clasificación de tumores cerebrales en 4 clases a partir de cortes de resonancia magnética (MRI) 2D axiales, utilizando el dataset **Task01_BrainTumour (BraTS)** de MONAI.

La arquitectura se basa en una red **ResNet-18** pre-entrenada, y la metodología de entrenamiento incorpora buenas prácticas del artículo *"Deep Learning–based Identification of Brain MRI Sequences"* (2023) para garantizar la solidez ante la variabilidad de escáneres y la correcta validación sin fugas de datos.

---

## 🛠️ Estructura del Proyecto

El código está estructurado de la siguiente forma:

- **`requirements.txt`**: Listado de dependencias necesarias.
- **`setup_env.ps1`** / **`setup_env.sh`**: Scripts de configuración de entorno para Windows y Linux/Colab.
- **`preprocess_data.py`**: Convierte los volúmenes 3D en rodajas (*slices*) axiales 2D de 3 canales (FLAIR, T1gd, T2w) y las clasifica.
- **`model.py`**: Define la arquitectura ResNet-18 y el soporte para ensambles de múltiples modelos.
- **`crossval.py`**: Divide a los pacientes utilizando una estrategia de **K-Fold Estratificado por paciente** para evitar fugas de datos.
- **`train.py`**: Script de entrenamiento principal con balanceo de clases, aumentación y registro (soporta Weights & Biases).
- **`evaluate.py`**: Evalúa los modelos entrenados, genera matrices de confusión, curvas ROC/PR, análisis de robustez por centro y mapas **Grad-CAM**.
- **`inference.py`**: Ejecuta predicciones sobre nuevos volúmenes NIfTI 3D y genera perfiles de probabilidad a lo largo del eje axial.
- **`external_test.py`**: Valida el modelo frente a un dataset externo independiente de imágenes 2D (`brain_tumor_dataset`).
- **`hanging_protocol_integration.py`**: Demostración de integración clínica PACS/Hanging Protocol basada en las predicciones.

---

## 🚀 Guía de Uso Paso a Paso

### 1. Preparar el Entorno

En Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

En Linux / Google Colab / macOS (Bash):
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Preprocesar los Datos

El preprocesamiento carga los archivos `.nii.gz` 3D, filtra las zonas sin tejido cerebral relevante y extrae cortes axiales 2D que contienen las clases:
- **Clase 0**: Sano / Fondo (Healthy/Bg)
- **Clase 1**: Edema (Edema)
- **Clase 2**: Tumor no realzado (Non-enhancing tumor)
- **Clase 3**: Tumor realzado (Enhancing tumor)

Para ejecutar el preprocesamiento:
```bash
# Activar entorno virtual si es necesario: .\venv\Scripts\activate o source venv/bin/activate
python preprocess_data.py
```
Esto creará una carpeta en `data/processed_2d/` separando las rodajas por carpetas de clase y guardando un archivo `metadata.json` con la distribución por pacientes.

### 3. Entrenar el Modelo (con Cross-Validation)

Para entrenar el modelo en el Fold 1 (por defecto se ejecutan 15 épocas):
```bash
python train.py --fold 1 --epochs 15 --batch_size 32
```
Puedes usar opciones adicionales como:
- `--device cpu` o `--device cuda`
- `--wandb` para activar el registro en Weights & Biases.
- Cambiar el fold con `--fold 2` hasta `--fold 5`.

Los checkpoints del mejor modelo basado en F1-Macro de validación se guardarán en la carpeta `checkpoints/`.

### 4. Evaluación Completa y Grad-CAM

Una vez entrenado el modelo para un fold, evalúa su rendimiento global y por centro clínico simulado (basado en el paper):
```bash
python evaluate.py --fold 1
```
Este script producirá en la carpeta `results/`:
1. **`confusion_matrix.png`**: Matriz de confusión detallada.
2. **`roc_curves.png`** y **`precision_recall_curves.png`**: Rendimiento ROC y PR por clase.
3. **`gradcam_samples.png`**: Explicabilidad visual **Grad-CAM** superpuesta sobre el canal FLAIR, localizando las áreas que la red considera clave para su decisión.
4. **Análisis de Robustez por Centro**: Desglose de precisión y F1 por subestudios/escáneres de origen.

### 5. Inferencia en Nuevos Volúmenes 3D

Para clasificar cada corte axial de un paciente completo y ver su perfil de evolución a lo largo del cerebro:
```bash
python inference.py --image_path data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --folds 1
```
*(Puedes pasar múltiples folds separados por comas para usar un **Ensemble** de predicción, p. ej. `--folds 1,2,3`)*

Esto guardará un gráfico de perfil de probabilidad `results/volume_prediction_profile.png` que ayuda al radiólogo a ver en qué coordenadas axiales se localiza cada componente tumoral.

### 6. Validación Externa (Generalización)

Valida la robustez y capacidad de generalización del modelo frente al dataset 2D externo (`brain_tumor_dataset`):
```bash
python external_test.py --folds 1
```
Este script mapea las predicciones de 4 clases a una clasificación binaria (Sano vs Tumor) y evalúa la **Sensibilidad** y **Especificidad** en imágenes reales no provenientes de la distribución de BraTS. Guarda el reporte en `results/external_test_report.txt`.

### 7. Integración Clínica (Hanging Protocol)

Simula un sistema PACS clínico que organiza de manera inteligente los monitores del especialista a partir de los hallazgos automáticos del modelo:
```bash
python hanging_protocol_integration.py --image_path data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --folds 1
```
Generará un archivo JSON de configuración de visualización `results/hanging_protocol_config.json` especificando qué secuencia colocar en cada monitor, en qué cortes enfocar la vista y el nivel de zoom idóneo.

---

## 📊 Buenas Prácticas del Artículo Incorporadas

1. **Manejo de Variabilidad de Escáneres (Scanner/Center Analysis)**: Evaluamos y desglosamos las métricas por centros independientes para detectar posibles pérdidas de generalización debido al fabricante del escáner.
2. **K-Fold Estratificado por Paciente**: Se evita rigurosamente cualquier fuga de datos (data leakage) asegurando que ninguna rodaja de un paciente de validación se use en el entrenamiento.
3. **Normalización por Canales**: Cada secuencia (FLAIR, T1gd, T2w) se escala independientemente de 0 a 1 por rodaja, minimizando el impacto del contraste relativo entre distintas máquinas.
4. **Pérdida Balanceada (Weighted Cross Entropy)**: Ajustamos el gradiente según la frecuencia de clases para compensar el desbalance natural de las áreas de tumor en el cerebro.
