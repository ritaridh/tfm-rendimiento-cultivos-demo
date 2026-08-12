
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


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
# CARGAR MODELOS Y DATOS
# ============================================================

@st.cache_resource
def cargar_modelos():

    modelo_rf = joblib.load(
        os.path.join(
            BASE_DIR,
            "RF_tuned_TFM.joblib"
        )
    )

    modelo_xgb = joblib.load(
        os.path.join(
            BASE_DIR,
            "XGB_tuned_TFM.joblib"
        )
    )

    return modelo_rf, modelo_xgb


@st.cache_data
def cargar_datos():

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

    return referencias, rangos


modelo_rf, modelo_xgb = cargar_modelos()
referencias, rangos = cargar_datos()


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

features = list(
    modelo_rf.feature_names_in_
)

num_cols = [
    c for c in features
    if c != "cultivo"
]

rangos = rangos.set_index(
    "variable"
)


# ============================================================
# CULTIVOS
# ============================================================

CULTIVOS_UI = {
    "Cebada": "barley",
    "Trigo blando": "common_soft_wheat"
}


# ============================================================
# GRUPOS DE VARIABLES
# ============================================================

variables_suelo = [
    "clay_pct_0_30cm",
    "sand_pct_0_30cm",
    "soc_g_kg_0_30cm",
    "ph_0_30cm",
    "bdod_g_cm3_0_30cm"
]

variables_sentinel = [
    c for c in num_cols
    if c.startswith(
        ("NDVI_", "EVI_", "NDMI_")
    )
]

variables_eurocrops = [
    "n_recintos",
    "superficie_eurocrops_ha",
    "superficie_media_recinto_ha"
]

variables_era5 = [
    c for c in num_cols
    if (
        c.startswith("temp_media_")
        or c.startswith("temp_rocio_")
        or c.startswith("precipitacion_")
        or c.startswith("radiacion_")
    )
]

VARS_CLIMA_RESUMEN = [
    "temp_media_c_campana",
    "temp_rocio_c_campana",
    "precipitacion_mm_campana",
    "radiacion_mj_m2_campana"
]

VARS_CLIMA_DETALLE = [
    variable
    for variable in variables_era5
    if variable not in VARS_CLIMA_RESUMEN
]

VARS_PERSONALIZABLES = (
    VARS_CLIMA_RESUMEN
    + VARS_CLIMA_DETALLE
    + variables_suelo
    + variables_sentinel
)


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
        (
            "temp_media_c_",
            "Temperatura media (°C) — "
        ),
        (
            "temp_rocio_c_",
            "Temperatura de rocío (°C) — "
        ),
        (
            "precipitacion_mm_",
            "Precipitación (mm) — "
        ),
        (
            "radiacion_mj_m2_",
            "Radiación (MJ/m²) — "
        ),
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
a partir del modelo del TFM. El modelo fue entrenado con
datos agregados a escala provincial.
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

# Provincias disponibles para el cultivo seleccionado
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
históricos de referencia.
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
meteorológica más detallada para diferentes periodos
de la campaña.
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
satelital o de índices de vegetación.

En caso contrario, deje los campos vacíos.
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
# INFORMACIÓN TERRITORIAL
# ------------------------------------------------------------

with st.expander(
    "🗺️ Información territorial de referencia"
):

    st.write(
        """
La aplicación incorpora automáticamente información
territorial asociada a la provincia y al cultivo seleccionados.
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
    # Crear entrada base con referencias
    # --------------------------------------------------------

    entrada = {}

    for variable in features:

        if variable == "cultivo":

            entrada[variable] = (
                cultivo_modelo
            )

        else:

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

    X_usuario = pd.DataFrame(
        [entrada],
        columns=features
    )

    # --------------------------------------------------------
    # Modelos
    # --------------------------------------------------------

    pred_rf = float(
        modelo_rf.predict(
            X_usuario
        )[0]
    )

    pred_xgb = float(
        modelo_xgb.predict(
            X_usuario
        )[0]
    )

    pred_ensemble = (
        0.50 * pred_rf
        + 0.50 * pred_xgb
    )

    pred_ensemble = max(
        0.0,
        pred_ensemble
    )

    rendimiento_t_ha = (
        pred_ensemble / 1000
    )

    produccion_total_t = (
        pred_ensemble
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
            f"{formato_es(pred_ensemble, 0)} kg/ha"
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
            len(num_cols)
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

    for variable in num_cols:

        if variable not in rangos.index:
            continue

        valor = float(
            X_usuario.iloc[0][variable]
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
            "rango habitual del modelo."
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
    # MODELOS
    # --------------------------------------------------------

    with st.expander(
        "🔬 Ver información del modelo"
    ):

        st.write(
            f"""
**Random Forest:** {formato_es(pred_rf, 0)} kg/ha

**XGBoost:** {formato_es(pred_xgb, 0)} kg/ha

La estimación principal combina ambos modelos
al **50 %**.
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

Para avanzar hacia una aplicación operacional sería necesario
incorporar más campañas, datos a escala de parcela y conexiones
automáticas con fuentes meteorológicas, edáficas y satelitales.
"""
)
