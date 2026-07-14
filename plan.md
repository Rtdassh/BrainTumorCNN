# Detectar 4 tipos de tumores cerebrales con CNN (actualizado)

## Goal Description
Desarrollar un modelo de red neuronal convolucional (CNN) capaz de clasificar imágenes de resonancia magnética cerebral en **4 tipos de tumores** utilizando el dataset *Task01_BrainTumour* de MONAI. El plan incorpora buenas prácticas extraídas del artículo "Deep Learning–based Identification of Brain MRI Sequences" (2023) para mejorar la robustez y generalización.

## User Review Required
> [!IMPORTANT]
> Confirme las decisiones siguientes antes de iniciar la implementación:
> - **Framework**: PyTorch + MONAI (ya confirmado).
> - **Recursos**: GPU en la nube (Google Colab / AWS) (confirmado).
> - **Tipo de modelo**: 2D (cortes axiales) (confirmado).
> - **Backbone**: ResNet‑18 pre‑entrenado (confirmado).
> - **Manejo de desbalance**: No aplicar técnicas de balanceo (confirmado). *Nota*: el artículo sugiere considerar técnicas para clases poco representadas; podemos añadirlas si lo desea.
> - **Métricas**: Todas (accuracy, macro‑F1, AUC, matriz de confusión) (confirmado).
> - **Entorno**: Jupyter Notebook (confirmado).
> - **Seguimiento**: Weights & Biases (confirmado).
> - **Exportación**: No exportar (confirmado).
> - **Validación externa**: ¿Desea reservar un conjunto externo (p. ej., otro centro) para probar la generalización?
> - **Ensemble**: ¿Le interesa entrenar un ensemble de varios modelos ResNet‑18 y combinar sus predicciones?
> - **Grad‑CAM**: ¿Desea incluir visualizaciones Grad‑CAM en la fase de evaluación?
> - **Integración en hanging protocols**: ¿Planea integrar el modelo en sistemas de organización automática de imágenes (hanging protocols) para uso clínico?

## Open Questions
> [!WARNING]
> - ¿Qué nivel de detalle necesita en la documentación y visualizaciones (p.ej., Grad‑CAM, curvas de aprendizaje)?
> - ¿Quiere incorporar búsqueda de hiperparámetros (Optuna) para optimizar el entrenamiento?
> - ¿Necesita soporte para clases subrepresentadas (p.ej., usar weighted loss o oversampling) pese a que inicialmente no lo aplicará?
> - ¿Desea incluir un conjunto de datos adicional (p.ej., imágenes sanas) para evaluar la generalización fuera de tumores?

## Proposed Changes
---
### 1. Preparar el entorno
- **[NEW] requirements.txt** – Listado actualizado de dependencias (torch, monai, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, torchvision, nibabel, torchmetrics, wandb, optuna).
- **[NEW] setup_env.sh** – Script para crear entorno virtual e instalar dependencias.
---
### 2. Descarga y organización de datos
- **[NEW] download_data.py** – Descarga y extracción de `Task01_BrainTumour.tar`.
---
### 3. Exploración y pre‑procesamiento
- **[NEW] data_exploration.ipynb** – Visualización de ejemplos, análisis de distribución de clases, y generación de histogramas de áreas tumorales (inspirado en Fig 3 del artículo).
---
### 4. Definición del modelo
- **[NEW] model.py** – Arquitectura basada en **ResNet‑18** pre‑entrenado (ImageNet). Incluye opción para crear un **ensemble** de N modelos con diferentes seeds.
---
### 5. Estrategia de validación
- **[NEW] crossval.py** – Implementa **estratificado 5‑fold cross‑validation** tal como el paper, garantizando balance por institución/paciente/etiqueta.
---
### 6. Entrenamiento
- **[NEW] train.py** – Script que:
  - Usa `CacheDataset`/`DataLoader` con augmentaciones (flip, rotate, intensity jitter).
  - Optimizador **AdamW** y scheduler **CosineAnnealingLR**.
  - Registra métricas y curvas en **Weights & Biases**.
  - Opcional: habilitar **WeightedCrossEntropy** o **FocalLoss** para clases minoritarias.
---
### 7. Evaluación y visualización
- **[NEW] evaluate.py** – Calcula accuracy, macro‑F1, AUC, matriz de confusión y genera:
  - Curvas ROC/PR por clase.
  - **Grad‑CAM** visualizaciones (siguiendo Fig 2 del artículo).
  - Reporte de desempeño por centro/hospedador (similar al análisis de scanner en el artículo).
---
### 8. Inferencia y exportación
- **[NEW] inference.py** – Carga modelo y predice nuevas imágenes.
- **[NEW] export_model.py** – Opcional (no requerido por el usuario).
---
### 9. Validación externa (opcional)
- **[NEW] external_test.py** – Permite evaluar el modelo en un conjunto externo no usado en el entrenamiento (p.ej., otro estudio clínico).
---
### 10. Integración clínica (opcional)
- **[NEW] hanging_protocol_integration.py** – Demo de cómo conectar el modelo a un flujo de trabajo de hanging protocol para organización automática de imágenes.
---
### 11. Documentación
- **[NEW] README.md** – Guía paso‑a‑paso, instrucciones de entrenamiento, evaluación y despliegue.
- **[NEW] docs/** – Diagramas de arquitectura (mermaid) y ejemplos de resultados.
---
### 12. Opcionales (según respuestas del usuario)
- Hyper‑parameter search con **Optuna**.
- Dockerfile para reproducibilidad.
- Script para balanceo avanzado de clases.

## Verification Plan
### Automated Tests
- Unit tests para descarga, carga de datos y arquitectura del modelo.
- Test de integridad del dataset (número de imágenes vs etiquetas).
- Test de la función de estratificado 5‑fold.
### Manual Verification
- El usuario revisará notebooks de exploración y los resultados de métricas en `evaluate.py`.
- Visualización de Grad‑CAM para casos representativos.
- Revisión del reporte de desempeño por centro.

*Una vez aprobada la planificación revisada, procederemos a crear/actualizar los archivos y a iterar con el usuario para refinar cada componente.*
