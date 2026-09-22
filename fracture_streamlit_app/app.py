import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🩻",
    layout="wide"
)


# =========================
# SESSION STATE
# =========================

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

/* Main title */
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #999;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Column headings */
.column-title {
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 15px;
}


/* =========================
   UPLOAD BOX
   ========================= */

[data-testid="stFileUploader"] {
    width: 100% !important;
}

[data-testid="stFileUploaderDropzone"] {
    height: 350px !important;
    min-height: 350px !important;
    width: 100% !important;

    border: 2px dashed #777 !important;
    border-radius: 12px !important;

    background: transparent !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    position: relative !important;
    box-sizing: border-box !important;
}

/* Hide Upload button */
[data-testid="stFileUploaderDropzone"] button {
    display: none !important;
}

/* Hide file information */
[data-testid="stFileUploaderDropzone"] small {
    display: none !important;
}

/* Hide default uploader text */
[data-testid="stFileUploaderDropzone"] > div {
    opacity: 0 !important;
}

/* Plus icon */
[data-testid="stFileUploaderDropzone"]::after {
    content: "+";

    position: absolute;

    left: 50%;
    top: 50%;

    transform: translate(-50%, -50%);

    font-size: 55px;
    font-weight: 300;

    color: #999;

    pointer-events: none;
}


/* =========================
   IMAGE BOX
   ========================= */

.image-box {
    width: 100%;

    border: 2px solid #777;
    border-radius: 12px;

    overflow: hidden;

    line-height: 0;

    box-sizing: border-box;
}

.image-box img {
    width: 100%;
    height: auto;
    display: block;
}


/* =========================
   EMPTY OUTPUT BOX
   ========================= */

.result-placeholder {
    width: 100%;
    height: 350px;

    border: 2px solid #777;
    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    color: #999;

    font-size: 17px;

    box-sizing: border-box;
}


/* =========================
   DISCLAIMER
   ========================= */

.disclaimer {
    text-align: center;
    color: #999;
    font-size: 13px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.pt"
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">🩻 Bone Fracture Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a musculoskeletal X-ray to detect possible fracture regions.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================
# TWO COLUMNS
# =========================

col1, col2 = st.columns(
    2,
    vertical_alignment="top"
)


# =========================
# LEFT - INPUT
# =========================

with col1:

    st.markdown(
        '<div class="column-title">📷 Input X-ray</div>',
        unsafe_allow_html=True
    )

    # =====================
    # SHOW UPLOADER
    # =====================

    if st.session_state.show_uploader:

        uploaded_file = st.file_uploader(
            "Upload X-ray",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded_file is not None:

            st.session_state.image_bytes = uploaded_file.getvalue()

            st.session_state.show_uploader = False

            st.rerun()

    # =====================
    # SHOW IMAGE
    # =====================

    else:

        image = Image.open(
            BytesIO(st.session_state.image_bytes)
        ).convert("RGB")

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG"
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode()

        st.markdown(
            f"""
            <div class="image-box">
                <img src="data:image/jpeg;base64,{image_base64}">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔄 Change Image",
            use_container_width=True
        ):

            st.session_state.image_bytes = None

            st.session_state.show_uploader = True

            st.session_state.uploader_key += 1

            st.rerun()


# =========================
# RIGHT - OUTPUT
# =========================

with col2:

    st.markdown(
        '<div class="column-title">🔍 Detection Result</div>',
        unsafe_allow_html=True
    )

    # =====================
    # NO IMAGE
    # =====================

    if st.session_state.image_bytes is None:

        st.markdown(
            """
            <div class="result-placeholder">
                Detection result will appear here
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================
    # IMAGE AVAILABLE
    # =====================

    else:

        image = Image.open(
            BytesIO(st.session_state.image_bytes)
        ).convert("RGB")

        with st.spinner("Analyzing X-ray..."):

            results = model.predict(
                image,
                conf=0.25,
                verbose=False
            )

        result = results[0]

        annotated_image = result.plot()

        annotated_image = Image.fromarray(
            annotated_image
        )

        buffer = BytesIO()

        annotated_image.save(
            buffer,
            format="JPEG"
        )

        output_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode()

        st.markdown(
            f"""
            <div class="image-box">
                <img src="data:image/jpeg;base64,{output_base64}">
            </div>
            """,
            unsafe_allow_html=True
        )

        fracture_count = len(result.boxes)

        if fracture_count > 0:

            st.success(
                f"Fracture Detected — {fracture_count} region(s)"
            )

        else:

            st.info(
                "No fracture detected."
            )


# =========================
# DISCLAIMER
# =========================

st.divider()

st.markdown(
    """
    <div class="disclaimer">
    This AI demonstration is trained on a limited dataset and is intended
    for educational and research purposes only. It should not be used for
    medical diagnosis or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)
