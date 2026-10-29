import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import pandas as pd

st.set_page_config(
    page_title="Digit Arcade",
    page_icon="🕹️",
    layout="wide"
)

# ============================================================
# ESTILOS (TEMÁTICA: ARCADE RETRO NEÓN)
# ============================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Orbitron:wght@500;700&display=swap');

    :root {
        --fondo: #0D0221;
        --fondo-claro: #1B0A3C;
        --fucsia: #FF2E97;
        --cian: #00F0FF;
        --amarillo: #FFE14D;
        --texto: #E8E1FF;
    }

    /* Fondo general */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(255, 46, 151, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(0, 240, 255, 0.15) 0%, transparent 45%),
            var(--fondo);
        font-family: 'Orbitron', sans-serif;
    }

    /* Texto general */
    .stMarkdown p, .stMarkdown li, label, .stCaption, span {
        color: var(--texto);
        font-family: 'Orbitron', sans-serif;
    }

    /* Título principal */
    h1 {
        font-family: 'Press Start 2P', cursive !important;
        color: var(--cian) !important;
        text-align: center;
        font-size: 30px !important;
        line-height: 1.5 !important;
        text-shadow: 0 0 8px var(--cian), 0 0 20px var(--fucsia);
    }

    /* Subtítulos */
    h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: var(--fucsia) !important;
        letter-spacing: 1px;
        text-shadow: 0 0 8px rgba(255, 46, 151, 0.6);
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B0A3C 0%, #0D0221 100%);
        border-right: 3px solid var(--fucsia);
        box-shadow: 4px 0 20px rgba(255, 46, 151, 0.35);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--cian) !important;
    }

    /* Botones */
    .stButton > button {
        background: transparent;
        color: var(--cian);
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: 2px solid var(--cian);
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: var(--cian);
        color: var(--fondo);
        box-shadow: 0 0 22px var(--cian);
    }

    .stButton > button[kind="primary"] {
        color: var(--fucsia);
        border-color: var(--fucsia);
        box-shadow: 0 0 10px rgba(255, 46, 151, 0.6);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--fucsia);
        color: white;
        box-shadow: 0 0 22px var(--fucsia);
    }

    /* Tarjetas de métricas */
    [data-testid="stMetric"] {
        background: var(--fondo-claro);
        border: 2px solid var(--fucsia);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 14px rgba(255, 46, 151, 0.35);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--cian) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--amarillo) !important;
        font-family: 'Press Start 2P', cursive !important;
        font-size: 28px !important;
        text-shadow: 0 0 10px var(--amarillo);
    }

    /* Alertas */
    [data-testid="stAlert"] {
        background: var(--fondo-claro);
        border: 1px solid var(--cian);
        border-radius: 12px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--fondo-claro);
        border: 1px solid var(--fucsia);
        border-radius: 12px;
    }

    /* Canvas: marco de pantalla arcade */
    iframe {
        border: 4px solid var(--fucsia) !important;
        border-radius: 14px;
        box-shadow: 0 0 25px rgba(255, 46, 151, 0.7), inset 0 0 25px rgba(0, 240, 255, 0.3);
    }

    hr { border-color: var(--fucsia) !important; opacity: 0.5; }

    /* Decoración */
    .subtitulo {
        text-align: center;
        color: var(--amarillo);
        font-size: 16px;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .adorno {
        text-align: center;
        color: var(--fucsia);
        font-size: 22px;
        letter-spacing: 14px;
        margin: 6px 0 20px 0;
        text-shadow: 0 0 10px var(--fucsia);
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.markdown("### 🕹️ Panel del jugador")
    st.write(f"**TensorFlow:** {tf.__version__}")
    st.write(f"**Keras:** {keras.__version__}")
    st.divider()
    st.write("🎮 **Cómo jugar**")
    st.write("1. Dibuja un número del 0 al 9")
    st.write("2. Pulsa **Predecir**")
    st.write("3. ¡Mira si la IA acierta!")

# ============================================================
# TÍTULO
# ============================================================

st.title("🕹️ DIGIT ARCADE")
st.markdown(
    "<p class='subtitulo'>¡Dibuja y la IA adivina!</p>"
    "<div class='adorno'>◆ ◇ ◆</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Una red neuronal CNN entrenada con MNIST "
    "intentará descubrir qué número escribiste a mano.</p>",
    unsafe_allow_html=True
)

# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras"""

    model_files = ['mnist_model.keras', 'mnist_model.h5']

    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                st.info(f"📂 Intentando cargar: {model_path}")
                model = keras.models.load_model(model_path)
                st.success(f"✅ Modelo cargado exitosamente desde: `{model_path}`")
                return model
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
                continue

    st.error("❌ No se encontró ningún modelo")
    st.info("""
    **Instrucciones:**
    1. Entrena el modelo en Google Colab
    2. Descarga `mnist_model.keras`
    3. Súbelo a la raíz de tu repositorio
    4. Reinicia la aplicación
    """)
    st.stop()

model = load_mnist_model()

# ============================================================
# INTERFAZ
# ============================================================

st.write("### 🎨 Dibuja un dígito del 0 al 9")

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### 🎮 Controles")
    predict_btn = st.button("🔍 **Predecir**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar", use_container_width=True)

    if clear_btn:
        st.rerun()

# ============================================================
# PREDICCIÓN
# ============================================================

if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ Insert coin: dibuja un dígito primero")
        else:
            with st.spinner("🤖 La IA está pensando..."):
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")

                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

            # Resultados
            st.markdown("<div class='adorno'>◆ ◇ ◆</div>", unsafe_allow_html=True)
            st.success(f"## 🏆 ¡Dígito detectado: **{digit}**!")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Predicción", digit)
            with col_m2:
                st.metric("Confianza", f"{confidence:.1f}%")
            with col_m3:
                alternative = np.argsort(prediction[0])[-2]
                st.metric("2ª opción", alternative)

            # Imágenes
            st.write("### 🖼️ Así vio la IA tu dibujo")
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.write("**Tu dibujo original:**")
                st.image(canvas_result.image_data, width=200)

            with col_img2:
                st.write("**Imagen procesada (28x28):**")
                st.image(image_resized, width=200)

            # Gráfico
            st.write("### 📊 Puntuación por dígito")
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'), color="#FF2E97")

            with st.expander("🔍 Ver detalles"):
                for i, prob in enumerate(prediction[0]):
                    emoji = "🎯" if i == digit else ""
                    st.write(f"{emoji} **Dígito {i}**: {prob*100:.2f}%")
    else:
        st.warning("⚠️ Insert coin: dibuja un dígito en el canvas")

# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown(
    "<div class='adorno'>◆ ◇ ◆</div>"
    "<p style='text-align:center; font-size:13px;'>DIGIT ARCADE · Press start to draw</p>",
    unsafe_allow_html=True
)
