import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64
import io
import html
from datetime import datetime


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_DIR = "fundburo_data"
DATA_PATH = os.path.join(DATA_DIR, "fundstuecke.json")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# CLEAN UI
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #f7f7f8;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Hide unnecessary Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------------- HEADER ---------------- */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #111827;
        color: white;
        font-size: 1.25rem;
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.02em;
    }

    .brand-small {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 2px;
    }

    /* ---------------- HERO ---------------- */

    .hero-clean {
        background: #111827;
        border-radius: 30px;
        padding: 3.2rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .hero-clean::before {
        content: "";
        position: absolute;
        width: 330px;
        height: 330px;
        border-radius: 50%;
        background: #312e81;
        right: -120px;
        top: -160px;
        opacity: 0.7;
    }

    .hero-clean::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: #0f766e;
        right: 130px;
        bottom: -180px;
        opacity: 0.35;
    }

    .hero-kicker {
        position: relative;
        z-index: 2;
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title-clean {
        position: relative;
        z-index: 2;
        color: white;
        font-size: clamp(2.3rem, 5vw, 4.3rem);
        font-weight: 800;
        letter-spacing: -0.065em;
        line-height: 0.98;
        max-width: 700px;
    }

    .hero-description-clean {
        position: relative;
        z-index: 2;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 1.1rem;
        max-width: 500px;
        line-height: 1.6;
    }

    /* ---------------- STATS ---------------- */

    .mini-stat {
        background: white;
        border: 1px solid #e8e8eb;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        height: 100%;
    }

    .mini-stat-number {
        font-size: 1.65rem;
        font-weight: 800;
        color: #111827;
    }

    .mini-stat-label {
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* ---------------- SECTION ---------------- */

    .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        margin: 2.3rem 0 1rem 0;
    }

    .section-head-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.03em;
    }

    .section-head-small {
        color: #9ca3af;
        font-size: 0.78rem;
    }

    /* ---------------- CARD ---------------- */

    .fund-card {
        background: white;
        border: 1px solid #e8e8eb;
        border-radius: 22px;
        overflow: hidden;
        margin-bottom: 0.65rem;
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease;
    }

    .fund-card:hover {
        transform: translateY(-6px);
        border-color: #c7d2fe;
        box-shadow: 0 20px 45px rgba(17,24,39,0.10);
    }

    .fund-image {
        width: 100%;
        height: 235px;
        object-fit: cover;
        display: block;
        background: #f1f5f9;
    }

    .fund-no-image {
        height: 235px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f1f5f9;
        color: #cbd5e1;
        font-size: 3rem;
    }

    .fund-content {
        padding: 1rem 1.05rem 1.1rem;
    }

    .fund-name {
        font-size: 1rem;
        font-weight: 750;
        color: #111827;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .fund-meta {
        margin-top: 0.45rem;
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        background: #f3f4f6;
        color: #6b7280;
        padding: 0.3rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .pill-dark {
        background: #eef2ff;
        color: #4338ca;
    }

    /* ---------------- EMPTY ---------------- */

    .empty {
        background: white;
        border: 1px dashed #d1d5db;
        border-radius: 22px;
        padding: 4rem 2rem;
        text-align: center;
    }

    .empty-icon {
        font-size: 2.5rem;
        margin-bottom: 0.7rem;
    }

    .empty-title {
        font-size: 1rem;
        font-weight: 750;
        color: #111827;
    }

    .empty-text {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }

    /* ---------------- DETAIL ---------------- */

    .detail-card {
        background: white;
        border: 1px solid #e8e8eb;
        border-radius: 25px;
        padding: 1.5rem;
    }

    .detail-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.05em;
    }

    .detail-muted {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .description {
        color: #4b5563;
        line-height: 1.7;
        font-size: 0.92rem;
    }

    /* ---------------- BUTTONS ---------------- */

    .stButton > button {
        border-radius: 12px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #e5e7eb;
        background: white;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #a5b4fc;
        box-shadow: 0 8px 20px rgba(79,70,229,0.10);
    }

    /* ---------------- INPUTS ---------------- */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        border-radius: 12px !important;
    }

    /* ---------------- MOBILE ---------------- */

    @media (max-width: 700px) {

        .hero-clean {
            padding: 2rem;
            border-radius: 22px;
        }

        .hero-title-clean {
            font-size: 2.5rem;
        }

        .fund-image,
        .fund-no-image {
            height: 200px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA
# =========================================================

def load_data():
    if not os.path.exists(DATA_PATH):
        return []

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_data(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


fundstuecke = load_data()


# =========================================================
# IMAGE HELPERS
# =========================================================

def image_to_base64(image):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(data):
    try:
        raw = base64.b64decode(data)
        return Image.open(
            io.BytesIO(raw)
        ).convert("RGB")
    except Exception:
        return None


def get_image(item):
    if item.get("image_data"):
        return base64_to_image(item["image_data"])

    old_path = item.get("image_path")

    if old_path and os.path.exists(old_path):
        try:
            return Image.open(old_path).convert("RGB")
        except Exception:
            pass

    return None


def image_as_html(image):
    if image is None:
        return """
        <div class="fund-no-image">
            📦
        </div>
        """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return f"""
    <img
        class="fund-image"
        src="data:image/jpeg;base64,{encoded}"
    >
    """


# =========================================================
# AI
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )
    except Exception:
        return None


@st.cache_data
def load_labels():

    if not os.path.exists(LABELS_PATH):
        return []

    labels = []

    try:
        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(" ", 1)

                if len(parts) == 2:
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)

    except Exception:
        return []

    return labels


model = load_model()
labels = load_labels()


def classify_image(image):

    if model is None:
        return "Unbekannt", 0.0

    try:

        image = (
            image
            .convert("RGB")
            .resize((224, 224))
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array = (
            array / 127.5
        ) - 1.0

        array = np.expand_dims(
            array,
            axis=0,
        )

        prediction = model.predict(
            array,
            verbose=0,
        )[0]

        index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[index]
        )

        if index < len(labels):
            label = labels[index]
        else:
            label = f"Klasse {index}"

        return label, confidence

    except Exception:
        return "Unbekannt", 0.0


# =========================================================
# SESSION
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected" not in st.session_state:
    st.session_state.selected = None


# =========================================================
# TOPBAR
# =========================================================

st.markdown(
    """
    <div class="topbar">

        <div class="brand">

            <div class="brand-icon">
                🔎
            </div>

            <div>
                <div class="brand-name">
                    Fundbüro
                </div>

                <div class="brand-small">
                    Digital
                </div>
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero-clean">

            <div class="hero-kicker">
                Digitales Fundbüro
            </div>

            <div class="hero-title-clean">
                Gefunden.<br>
                Gespeichert.
            </div>

            <div class="hero-description-clean">
                Fundstücke fotografieren, automatisch erkennen
                und später wiederfinden.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats
    total = len(fundstuecke)

    categories = len(
        set(
            item.get(
                "category",
                "Unbekannt"
            )
            for item in fundstuecke
        )
    )

    locations = len(
        set(
            item.get(
                "location",
                ""
            )
            for item in fundstuecke
            if item.get("location")
        )
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-number">
                    {total}
                </div>
                <div class="mini-stat-label">
                    Fundstücke
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-number">
                    {categories}
                </div>
                <div class="mini-stat-label">
                    Kategorien
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-number">
                    {locations}
                </div>
                <div class="mini-stat-label">
                    Fundorte
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actions
    st.write("")

    a1, a2 = st.columns([1, 1])

    with a1:
        if st.button(
            "＋ Fundstück hinzufügen",
            use_container_width=True,
        ):
            st.session_state.page = "add"
            st.rerun()

    with a2:
        if st.button(
            "⌕ Suchen",
            use_container_width=True,
        ):
            st.session_state.page = "search"
            st.rerun()

    # Latest
    st.markdown(
        """
        <div class="section-head">

            <div>
                <div class="section-head-title">
                    Zuletzt
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fundstuecke:

        st.markdown(
            """
            <div class="empty">

                <div class="empty-icon">
                    📦
                </div>

                <div class="empty-title">
                    Noch leer
                </div>

                <div class="empty-text">
                    Dein erstes Fundstück erscheint hier.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        latest = list(
            reversed(
                fundstuecke[-6:]
            )
        )

        cols = st.columns(3)

        for number, item in enumerate(latest):

            with cols[number % 3]:

                name = html.escape(
                    str(
                        item.get(
                            "name",
                            "Fundstück",
                        )
                    )
                )

                category = html.escape(
                    str(
                        item.get(
                            "category",
                            "Unbekannt",
                        )
                    )
                )

                location = html.escape(
                    str(
                        item.get(
                            "location",
                            "Kein Ort",
                        )
                    )
                )

                image = get_image(item)

                st.markdown(
                    f"""
                    <div class="fund-card">

                        {image_as_html(image)}

                        <div class="fund-content">

                            <div class="fund-name">
                                {name}
                            </div>

                            <div class="fund-meta">

                                <span class="pill pill-dark">
                                    {category}
                                </span>

                                <span class="pill">
                                    📍 {location}
                                </span>

                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                original_index = fundstuecke.index(item)

                if st.button(
                    "Ansehen",
                    key=f"home_{original_index}",
                    use_container_width=True,
                ):
                    st.session_state.selected = original_index
                    st.session_state.page = "detail"
                    st.rerun()


# =========================================================
# SEARCH
# =========================================================

elif st.session_state.page == "search":

    if st.button("← Zurück"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        """
        <div class="section-head">

            <div class="section-head-title">
                Suche
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Suche",
        placeholder="Was suchst du?",
        label_visibility="collapsed",
    )

    categories = sorted(
        set(
            item.get(
                "category",
                "Unbekannt",
            )
            for item in fundstuecke
        )
    )

    selected_category = st.selectbox(
        "Kategorie",
        ["Alle"] + categories,
    )

    results = []

    query = search.lower().strip()

    for index, item in enumerate(fundstuecke):

        searchable = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("description", "")),
                str(item.get("location", "")),
                str(item.get("category", "")),
            ]
        ).lower()

        if query and query not in searchable:
            continue

        if (
            selected_category != "Alle"
            and item.get("category")
            != selected_category
        ):
            continue

        results.append(
            (index, item)
        )

    st.caption(
        f"{len(results)} Ergebnisse"
    )

    if not results:

        st.markdown(
            """
            <div class="empty">

                <div class="empty-icon">
                    ⌕
                </div>

                <div class="empty-title">
                    Nichts gefunden
                </div>

                <div class="empty-text">
                    Versuche einen anderen Suchbegriff.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        cols = st.columns(3)

        for number, (index, item) in enumerate(results):

            with cols[number % 3]:

                name = html.escape(
                    str(
                        item.get(
                            "name",
                            "Fundstück",
                        )
                    )
                )

                category = html.escape(
                    str(
                        item.get(
                            "category",
                            "Unbekannt",
                        )
                    )
                )

                location = html.escape(
                    str(
                        item.get(
                            "location",
                            "Kein Ort",
                        )
                    )
                )

                image = get_image(item)

                st.markdown(
                    f"""
                    <div class="fund-card">

                        {image_as_html(image)}

                        <div class="fund-content">

                            <div class="fund-name">
                                {name}
                            </div>

                            <div class="fund-meta">

                                <span class="pill pill-dark">
                                    {category}
                                </span>

                                <span class="pill">
                                    📍 {location}
                                </span>

                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Ansehen",
                    key=f"search_{index}",
                    use_container_width=True,
                ):
                    st.session_state.selected = index
                    st.session_state.page = "detail"
                    st.rerun()


# =========================================================
# ADD
# =========================================================

elif st.session_state.page == "add":

    if st.button("← Zurück"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        """
        <div class="section-head">
            <div class="section-head-title">
                Neues Fundstück
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:

        name = st.text_input(
            "Name",
            placeholder="z. B. Schwarzer Rucksack",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Kurze Beschreibung...",
            height=120,
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle",
        )

        date = st.date_input(
            "Datum",
            value=datetime.now().date(),
        )

    with right:

        source = st.radio(
            "Foto",
            [
                "Datei",
                "Kamera",
            ],
            horizontal=True,
        )

        if source == "Datei":

            uploaded = st.file_uploader(
                "Bild auswählen",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],
            )

        else:

            uploaded = st.camera_input(
                "Foto aufnehmen"
            )

        image = None

        if uploaded is not None:

            try:
                image = Image.open(
                    uploaded
                ).convert("RGB")

                st.image(
                    image,
                    caption="Vorschau",
                )

            except Exception:
                st.error(
                    "Bild konnte nicht gelesen werden."
                )

    if image is not None:

        with st.spinner("KI erkennt das Fundstück..."):

            predicted, confidence = classify_image(
                image
            )

        if predicted != "Unbekannt":

            st.success(
                f"Erkannt: {predicted} · "
                f"{confidence * 100:.0f}%"
            )

    st.write("")

    if st.button(
        "Speichern",
        type="primary",
        use_container_width=True,
    ):

        if not name.strip():

            st.warning(
                "Bitte einen Namen eingeben."
            )

        elif image is None:

            st.warning(
                "Bitte ein Foto hinzufügen."
            )

        else:

            new_item = {
                "name": name.strip(),
                "description": description.strip(),
                "location": location.strip(),
                "date": str(date),
                "category": (
                    predicted
                    if image is not None
                    else "Unbekannt"
                ),
                "confidence": (
                    confidence
                    if image is not None
                    else 0.0
                ),
                "image_data": image_to_base64(
                    image
                ),
            }

            fundstuecke.append(
                new_item
            )

            save_data(
                fundstuecke
            )

            st.session_state.page = "home"

            st.rerun()


# =========================================================
# DETAIL
# =========================================================

elif st.session_state.page == "detail":

    index = st.session_state.selected

    if (
        index is None
        or index < 0
        or index >= len(fundstuecke)
    ):

        st.session_state.page = "home"
        st.rerun()

    item = fundstuecke[index]

    if st.button("← Zurück"):

        st.session_state.page = "home"
        st.rerun()

    st.write("")

    left, right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with left:

        image = get_image(item)

        if image is not None:

            st.image(
                image,
                caption="Fundstück",
            )

        else:

            st.markdown(
                """
                <div class="empty">
                    <div class="empty-icon">
                        📦
                    </div>
                    Kein Foto
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:

        name = html.escape(
            str(
                item.get(
                    "name",
                    "Fundstück",
                )
            )
        )

        category = html.escape(
            str(
                item.get(
                    "category",
                    "Unbekannt",
                )
            )
        )

        location = html.escape(
            str(
                item.get(
                    "location",
                    "Kein Fundort",
                )
            )
        )

        date = html.escape(
            str(
                item.get(
                    "date",
                    "Kein Datum",
                )
            )
        )

        st.markdown(
            f"""
            <div class="detail-card">

                <div class="detail-title">
                    {name}
                </div>

                <div class="fund-meta" style="margin-top:1rem;">

                    <span class="pill pill-dark">
                        {category}
                    </span>

                    <span class="pill">
                        📍 {location}
                    </span>

                    <span class="pill">
                        {date}
                    </span>

                </div>

                <div style="height:1.5rem;"></div>

                <div class="detail-muted">
                    BESCHREIBUNG
                </div>

                <div class="description" style="margin-top:0.5rem;">
                    {html.escape(
                        str(
                            item.get(
                                "description",
                                "Keine Beschreibung.",
                            )
                        )
                    )}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        confidence = item.get(
            "confidence",
            0,
        )

        if confidence:

            st.caption(
                f"KI-Erkennung · "
                f"{float(confidence) * 100:.0f}%"
            )

        st.write("")

        if st.button(
            "Fundstück löschen",
            use_container_width=True,
        ):

            del fundstuecke[index]

            save_data(
                fundstuecke
            )

            st.session_state.selected = None
            st.session_state.page = "home"

            st.rerun()
