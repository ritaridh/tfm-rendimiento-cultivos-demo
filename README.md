# Demo web · Predicción de rendimiento de cultivos

Aplicación demostrativa asociada al Trabajo Fin de Máster:

**Predicción regional del rendimiento de cultivos mediante fusión de datos multifuente y técnicas de Machine Learning y Deep Learning**

Autora: **Rita Isabel Duarte Henriques**

## Modelo desplegado

La aplicación utiliza la **LSTM multirrama final** seleccionada en el TFM.

Entradas principales:

- meteorología ERA5-Land;
- índices Sentinel-2;
- propiedades edáficas;
- variables estructurales de EuroCrops;
- cultivo (trigo blando o cebada).

Los artefactos de inferencia incluidos son:

- `LSTM_multirrama_final.keras`;
- scalers e imputers;
- `config_inferencia_LSTM.json`;
- referencias históricas 2017–2022;
- rangos de entrenamiento.

## Aplicación

https://tfm-rendimiento-cultivos.streamlit.app/

## Repositorio principal del TFM

Código, notebooks, dataset procesado, resultados y documentación:

https://github.com/ritaridh/tfm-crop-yield-prediction

## Alcance

El modelo fue desarrollado y evaluado con rendimientos agregados a **escala provincial**. La aplicación es una demostración del proceso de inferencia y no constituye una herramienta validada para predicciones a escala de parcela.
