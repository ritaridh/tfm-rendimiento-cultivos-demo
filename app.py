import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de rendimiento agrícola",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# CARGAR MODELO, PREPROCESADORES Y CONFIGURACIÓN
# ============================================================

@st.cache_resource
def cargar_modelo_y_preprocesadores():

    modelo = tf.keras.models.load_model(
        os.path.join(
            BASE_DIR,
            "LSTM_multirrama_final.keras"
        )
    )

    scaler_era5 = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler_ERA5_LSTM.joblib"
        )
    )

    imputer_sentinel = joblib.load(
        os.path.join(
            BASE_DIR,
            "imputer_Sentinel_LSTM.joblib"
        )
    )

    scaler_sentinel = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler_Sentinel_LSTM.joblib"
        )
    )

    imputer_static = joblib.load(
        os.path.join(
            BASE_DIR,
            "imputer_Static_LSTM.joblib"
        )
    )

    scaler_static = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler_Static_LSTM.joblib"
        )
    )

    return (
        modelo,
        scaler_era5,
        imputer_sentinel,
        scaler_sentinel,
        imputer_static,
        scaler_static
    )


@st.cache_data
def cargar_configuracion_y_datos():

    with open(
        os.path.join(
            BASE_DIR,
            "config_inferencia_LSTM.json"
        ),
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    referencias = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "referencias_provincia_cultivo_2017_2022.csv"
        )
    )

    rangos = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "rangos_entrenamiento_2017_2022.csv"
        )
    )

    return config, referencias, rangos


(
    modelo_lstm,
    scaler_era5,
    imputer_sentinel,
    scaler_sentinel,
    imputer_static,
    scaler_static
) = cargar_modelo_y_preprocesadores()

config, referencias, rangos = cargar_configuracion_y_datos()

rangos = rangos.set_index("variable")


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

PERIODOS_ERA5 = config["periodos_era5"]
VARIABLES_ERA5 = config["variables_era5"]

PERIODOS_SENTINEL = config["periodos_sentinel"]
INDICES_SENTINEL = config["indices_sentinel"]

VARIABLES_STATIC = config["variables_static"]
MAPA_CULTIVO = config["mapa_cultivo"]

variables_era5_temporales = [
    f"{variable}_{periodo}"
    for periodo in PERIODOS_ERA5
    for variable in VARIABLES_ERA5
]

variables_sentinel = [
    f"{indice}_{periodo}"
    for periodo in PERIODOS_SENTINEL
    for indice in INDICES_SENTINEL
]

variables_suelo = [
    "clay_pct_0_30cm",
    "sand_pct_0_30cm",
    "soc_g_kg_0_30cm",
    "ph_0_30cm",
    "bdod_g_cm3_0_30cm"
]

variables_eurocrops = [
    "n_recintos",
    "superficie_eurocrops_ha",
    "superficie_media_recinto_ha"
]

VARS_CLIMA_RESUMEN = [
    "temp_media_c_campana",
    "temp_rocio_c_campana",
    "precipitacion_mm_campana",
    "radiacion_mj_m2_campana"
]

VARS_CLIMA_DETALLE = variables_era5_temporales

NUM_COLS_MODELO = (
    variables_era5_temporales
    + variables_sentinel
    + VARIABLES_STATIC
)


# ============================================================
# CULTIVOS
# ============================================================

CULTIVOS_UI = {
    "Cebada": "barley",
    "Trigo blando": "common_soft_wheat"
}


# ============================================================
# ETIQUETAS AMIGABLES
# ============================================================

def etiqueta_variable(variable):

    etiquetas = {
        "clay_pct_0_30cm":
            "Arcilla 0–30 cm (%)",

        "sand_pct_0_30cm":
            "Arena 0–30 cm (%)",

        "soc_g_kg_0_30cm":
            "Carbono orgánico del suelo (g/kg)",

        "ph_0_30cm":
            "pH del suelo",

        "bdod_g_cm3_0_30cm":
            "Densidad aparente (g/cm³)"
    }

    if variable in etiquetas:
        return etiquetas[variable]

    nombre = variable

    reemplazos = [
        ("temp_media_c_", "Temperatura media (°C) — "),
        ("temp_rocio_c_", "Temperatura de rocío (°C) — "),
        ("precipitacion_mm_", "Precipitación (mm) — "),
        ("radiacion_mj_m2_", "Radiación (MJ/m²) — "),
        ("NDVI_", "NDVI — "),
        ("EVI_", "EVI — "),
        ("NDMI_", "NDMI — "),
        ("ene_feb", "Enero–Febrero"),
        ("jan_feb", "Enero–Febrero"),
        ("mar_abr", "Marzo–Abril"),
        ("mar_apr", "Marzo–Abril"),
        ("may_jun", "Mayo–Junio"),
        ("jul_ago", "Julio–Agosto"),
        ("sep_dic", "Septiembre–Diciembre"),
        ("nov_dec", "Noviembre–Diciembre"),
        ("campana", "Campaña completa"),
        ("jul", "Julio")
    ]

    for antiguo, nuevo in reemplazos:
        nombre = nombre.replace(
            antiguo,
            nuevo
        )

    return nombre


# ============================================================
# FORMATO NUMÉRICO ESPAÑOL
# ============================================================

def formato_es(numero, decimales=0):

    texto = f"{numero:,.{decimales}f}"

    return (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# ============================================================
# REFERENCIAS HISTÓRICAS
# ============================================================

def obtener_referencia(
    provincia,
    cultivo_modelo
):

    fila = referencias[
        (
            referencias[
                "provincia_estandar"
            ] == provincia
        )
        &
        (
            referencias[
                "cultivo"
            ] == cultivo_modelo
        )
    ]

    if fila.empty:
        return None

    return fila.iloc[0].copy()


# ============================================================
# INPUTS OPCIONALES
# ============================================================

def campo_opcional(
    variable,
    key
):
    """
    Campo vacío = utilizar referencia histórica.
    """

    valor = st.text_input(
        etiqueta_variable(variable),
        value="",
        placeholder="Usar valor histórico de referencia",
        key=key
    )

    if valor.strip() == "":
        return None

    try:
        return float(
            valor.replace(",", ".")
        )

    except ValueError:
        st.warning(
            f"Valor no válido en "
            f"'{etiqueta_variable(variable)}'. "
            f"Se utilizará la referencia histórica."
        )

        return None


# ============================================================
# CONSTRUIR INPUTS LSTM
# ============================================================

def preparar_inputs_lstm(
    entrada
):

    # --------------------------------------------------------
    # ERA5-Land: (1, 5, 4)
    # --------------------------------------------------------

    era5 = np.array(
        [
            [
                entrada[
                    f"{variable}_{periodo}"
                ]
                for variable in VARIABLES_ERA5
            ]
            for periodo in PERIODOS_ERA5
        ],
        dtype=np.float32
    ).reshape(1, 5, 4)

    era5_2d = era5.reshape(-1, 4)

    era5_scaled = (
        scaler_era5
        .transform(era5_2d)
        .reshape(1, 5, 4)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Sentinel-2: (1, 5, 3)
    # --------------------------------------------------------

    sentinel = np.array(
        [
            [
                entrada[
                    f"{indice}_{periodo}"
                ]
                for indice in INDICES_SENTINEL
            ]
            for periodo in PERIODOS_SENTINEL
        ],
        dtype=np.float32
    ).reshape(1, 5, 3)

    sentinel_2d = sentinel.reshape(-1, 3)

    sentinel_imp = (
        imputer_sentinel
        .transform(sentinel_2d)
    )

    sentinel_scaled = (
        scaler_sentinel
        .transform(sentinel_imp)
        .reshape(1, 5, 3)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Variables estáticas/agregadas: (1, 12)
    # --------------------------------------------------------

    static = np.array(
        [
            entrada[variable]
            for variable in VARIABLES_STATIC
        ],
        dtype=np.float32
    ).reshape(1, 12)

    static_imp = (
        imputer_static
        .transform(static)
    )

    static_scaled = (
        scaler_static
        .transform(static_imp)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Cultivo: (1, 1)
    # --------------------------------------------------------

    crop = np.array(
        [[
            MAPA_CULTIVO[
                entrada["cultivo"]
            ]
        ]],
        dtype=np.float32
    )

    return [
        era5_scaled,
        sentinel_scaled,
        static_scaled,
        crop
    ]


# ============================================================
# CABECERA
# ============================================================

st.title(
    "🌾 Predicción de rendimiento agrícola"
)

st.subheader(
    "Demo experimental para trigo blando y cebada"
)

st.write(
    """
Para obtener una estimación básica solo es necesario indicar
el **cultivo, la provincia y la superficie de la finca**.

La aplicación completa automáticamente la información restante
con valores históricos de referencia del periodo **2017–2022**.

Si dispone de información propia sobre clima, suelo o estado
del cultivo, puede introducirla en los apartados opcionales.
Los campos que queden vacíos se completarán automáticamente.
"""
)

st.info(
    """
Esta herramienta es un prototipo demostrativo desarrollado
en el marco del TFM. El modelo trabaja con información agregada
a escala provincial, por lo que el resultado debe interpretarse
como una estimación orientativa y no como una predicción
parcelaria operacional.
"""
)


# ============================================================
# DATOS BÁSICOS
# ============================================================

st.header(
    "📍 Datos básicos de la finca"
)

col1, col2, col3 = st.columns(3)

with col1:

    cultivo_ui = st.selectbox(
        "Cultivo",
        list(
            CULTIVOS_UI.keys()
        )
    )

cultivo_modelo = (
    CULTIVOS_UI[cultivo_ui]
)

provincias_disponibles = sorted(
    referencias.loc[
        referencias["cultivo"]
        == cultivo_modelo,
        "provincia_estandar"
    ]
    .dropna()
    .unique()
)

with col2:

    provincia = st.selectbox(
        "Provincia",
        provincias_disponibles
    )

with col3:

    superficie_ha = st.number_input(
        "Superficie de la finca (ha)",
        min_value=0.01,
        value=10.0,
        step=0.1
    )


# ============================================================
# INFORMACIÓN OPCIONAL
# ============================================================

st.markdown(
    """
### ¿Dispone de información adicional?

Los siguientes apartados son **opcionales**.

Introduzca únicamente los datos disponibles.
Los campos vacíos utilizarán automáticamente los valores
históricos de referencia de la provincia y cultivo seleccionados.
"""
)

datos_usuario = {}


# ------------------------------------------------------------
# CLIMA GENERAL
# ------------------------------------------------------------

with st.expander(
    "🌦️ Datos meteorológicos generales — opcional"
):

    st.write(
        """
Introduzca los datos generales de la campaña si dispone
de ellos. No es necesario completar todos los campos.
"""
    )

    for variable in VARS_CLIMA_RESUMEN:

        datos_usuario[variable] = (
            campo_opcional(
                variable,
                f"general_{variable}"
            )
        )


# ------------------------------------------------------------
# CLIMA DETALLADO
# ------------------------------------------------------------

with st.expander(
    "🌦️ Meteorología detallada por periodos — avanzado"
):

    st.write(
        """
Utilice este apartado únicamente si dispone de información
meteorológica para los diferentes periodos de la campaña.
"""
    )

    for variable in VARS_CLIMA_DETALLE:

        datos_usuario[variable] = (
            campo_opcional(
                variable,
                f"meteo_{variable}"
            )
        )


# ------------------------------------------------------------
# SUELO
# ------------------------------------------------------------

with st.expander(
    "🌱 Propiedades del suelo — opcional"
):

    st.write(
        """
Si dispone de un análisis de suelo, introduzca los valores
disponibles. No es necesario completar todas las propiedades.
"""
    )

    for variable in variables_suelo:

        datos_usuario[variable] = (
            campo_opcional(
                variable,
                f"suelo_{variable}"
            )
        )


# ------------------------------------------------------------
# SATÉLITE
# ------------------------------------------------------------

with st.expander(
    "🛰️ Datos satelitales / estado del cultivo — avanzado"
):

    st.write(
        """
Introduzca estos valores únicamente si dispone de información
de índices de vegetación. En caso contrario, deje los campos
vacíos y se utilizarán las referencias históricas.
"""
    )

    for variable in variables_sentinel:

        datos_usuario[variable] = (
            campo_opcional(
                variable,
                f"sat_{variable}"
            )
        )


# ------------------------------------------------------------
# INFORMACIÓN TERRITORIAL / EUROCROPS
# ------------------------------------------------------------

with st.expander(
    "🗺️ Información territorial de referencia"
):

    st.write(
        """
La aplicación incorpora automáticamente las variables
estructurales de EuroCrops asociadas a la provincia y cultivo.
No se solicitan como datos de entrada al usuario.
"""
    )


# ============================================================
# BOTÓN
# ============================================================

st.header(
    "🌾 Obtener estimación"
)

calcular = st.button(
    "CALCULAR RENDIMIENTO",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICCIÓN
# ============================================================

if calcular:

    referencia = obtener_referencia(
        provincia,
        cultivo_modelo
    )

    if referencia is None:

        st.error(
            "No existen datos de referencia para "
            "la combinación seleccionada."
        )

        st.stop()


    # --------------------------------------------------------
    # Crear entrada base con referencias históricas
    # --------------------------------------------------------

    entrada = {
        "cultivo": cultivo_modelo
    }

    for variable in NUM_COLS_MODELO:

        if variable not in referencia.index:

            st.error(
                f"No se encuentra la variable "
                f"'{variable}' en el archivo de referencias."
            )

            st.stop()

        entrada[variable] = float(
            referencia[variable]
        )


    # --------------------------------------------------------
    # Sustituir datos personalizados
    # --------------------------------------------------------

    variables_personalizadas = []

    for variable, valor in (
        datos_usuario.items()
    ):

        if valor is not None:

            entrada[variable] = float(
                valor
            )

            variables_personalizadas.append(
                variable
            )


    # --------------------------------------------------------
    # Preparar las cuatro entradas de la LSTM
    # --------------------------------------------------------

    X_lstm = preparar_inputs_lstm(
        entrada
    )


    # --------------------------------------------------------
    # Predicción LSTM
    # --------------------------------------------------------

    pred_lstm = float(
        modelo_lstm.predict(
            X_lstm,
            verbose=0
        )
        .reshape(-1)[0]
    )

    pred_lstm = max(
        0.0,
        pred_lstm
    )

    rendimiento_t_ha = (
        pred_lstm / 1000
    )

    produccion_total_t = (
        pred_lstm
        * superficie_ha
        / 1000
    )


    # --------------------------------------------------------
    # RESULTADO PRINCIPAL
    # --------------------------------------------------------

    st.success(
        "Estimación calculada correctamente"
    )

    st.subheader(
        "🌾 Rendimiento estimado"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Rendimiento",
            f"{formato_es(pred_lstm, 0)} kg/ha"
        )

    with r2:

        st.metric(
            "Rendimiento",
            f"{formato_es(rendimiento_t_ha, 2)} t/ha"
        )

    with r3:

        st.metric(
            "Producción total estimada",
            f"{formato_es(produccion_total_t, 2)} t"
        )


    # --------------------------------------------------------
    # DATOS UTILIZADOS
    # --------------------------------------------------------

    with st.expander(
        "📊 Ver datos utilizados para la estimación"
    ):

        n_personalizadas = len(
            variables_personalizadas
        )

        n_referencia = (
            len(NUM_COLS_MODELO)
            - n_personalizadas
        )

        if n_personalizadas == 0:

            st.write(
                """
No se han introducido datos adicionales.

La estimación utiliza valores históricos de referencia
del periodo **2017–2022** para la provincia y cultivo
seleccionados.
"""
            )

        else:

            st.write(
                f"""
**Datos introducidos por el usuario:** {n_personalizadas}

**Variables completadas automáticamente con valores
de referencia:** {n_referencia}
"""
            )

        st.markdown(
            "#### Principales valores históricos de referencia"
        )

        st.write(
            f"""
- Temperatura media de campaña:
  **{formato_es(float(referencia["temp_media_c_campana"]), 1)} °C**
- Precipitación de campaña:
  **{formato_es(float(referencia["precipitacion_mm_campana"]), 1)} mm**
- Radiación de campaña:
  **{formato_es(float(referencia["radiacion_mj_m2_campana"]), 1)} MJ/m²**
- Arena del suelo:
  **{formato_es(float(referencia["sand_pct_0_30cm"]), 1)} %**
- Arcilla del suelo:
  **{formato_es(float(referencia["clay_pct_0_30cm"]), 1)} %**
- pH del suelo:
  **{formato_es(float(referencia["ph_0_30cm"]), 2)}**
- NDVI mayo–junio:
  **{formato_es(float(referencia["NDVI_may_jun"]), 3)}**
- NDMI mayo–junio:
  **{formato_es(float(referencia["NDMI_may_jun"]), 3)}**
"""
        )


    # --------------------------------------------------------
    # CONTROL DE RANGO
    # --------------------------------------------------------

    fuera_rango = []

    for variable in NUM_COLS_MODELO:

        if variable not in rangos.index:
            continue

        valor = float(
            entrada[variable]
        )

        minimo = float(
            rangos.loc[
                variable,
                "minimo"
            ]
        )

        maximo = float(
            rangos.loc[
                variable,
                "maximo"
            ]
        )

        if (
            valor < minimo
            or valor > maximo
        ):

            fuera_rango.append(
                variable
            )

    if len(fuera_rango) == 0:

        st.success(
            "✅ Los datos utilizados están dentro del "
            "rango observado durante el entrenamiento."
        )

    else:

        st.warning(
            f"⚠️ Se han detectado {len(fuera_rango)} "
            "valores fuera del rango observado durante "
            "el entrenamiento. La estimación debe "
            "interpretarse con mayor precaución."
        )

        with st.expander(
            "Ver valores fuera del rango"
        ):

            for variable in fuera_rango:

                st.write(
                    f"- {etiqueta_variable(variable)}"
                )


    # --------------------------------------------------------
    # INFORMACIÓN DEL MODELO
    # --------------------------------------------------------

    with st.expander(
        "🔬 Ver información del modelo"
    ):

        st.write(
            f"""
**Modelo:** LSTM multirrama de Deep Learning

**Predicción:** {formato_es(pred_lstm, 0)} kg/ha

La arquitectura integra de forma independiente:

- series temporales meteorológicas ERA5-Land;
- series temporales de índices Sentinel-2;
- propiedades del suelo y variables estructurales/agregadas;
- tipo de cultivo.

La arquitectura LSTM fue seleccionada mediante validación
temporal sobre 2020–2022 y posteriormente evaluada sobre
el año 2023 como test independiente.

**Rendimiento en test 2023:**

- MAE: **572 kg/ha**
- RMSE: **715 kg/ha**
- R²: **0,527**
"""
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.subheader(
    "ℹ️ Alcance de la herramienta"
)

st.write(
    """
Esta demo muestra cómo puede integrarse el modelo desarrollado
en el TFM dentro de una herramienta sencilla orientada a
usuarios finales.

El modelo fue desarrollado con datos agregados a escala
provincial. Por ello, la aplicación constituye una demostración
tecnológica y sus resultados no deben interpretarse como una
predicción parcelaria operacional.

Una aplicación operacional requeriría ampliar el número de
campañas, trabajar con datos a escala de parcela y automatizar
la incorporación de información meteorológica, edáfica y
satelital actualizada.
"""
)
