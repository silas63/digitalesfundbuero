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
# KONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_DIR = "fundburo_data"
DATA_PATH = os.path.join(DATA_DIR, "fundstuecke.json")


# =========================================================
# MODERNES DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       GRUNDLAYOUT
    ----------------------------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,0.08), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,0.08), transparent 28%),
            #f6f8fc;
        color: #172033;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.65rem 0.7rem;
        border-radius: 10px;
        transition: 0.2s ease;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }

    /* -----------------------------
       HERO
    ----------------------------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.6rem 2.8rem;
        border-radius: 28px;
        margin-bottom: 1.8rem;

        background:
            radial-gradient(circle at 90% 10%, rgba(129,140,248,0.30), transparent 30%),
            linear-gradient(135deg, #111827 0%, #1e293b 52%, #312e81 100%);

        box-shadow:
            0 18px 50px rgba(15,23,42,0.18);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -80px;
        bottom: -100px;
        border-radius: 50%;
        background: rgba(255,255,255,0.07);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #c7d2fe;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero-title {
        color: white;
        font-size: clamp(2rem, 4vw, 3.3rem);
        font-weight: 850;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 720px;
        line-height: 1.7;
        margin-top: 0.9rem;
    }

    /* -----------------------------
       SECTION HEADINGS
    ----------------------------- */

    .section-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    .section-subtitle {
        color: #64748b;
        margin-bottom: 1.3rem;
    }

    /* -----------------------------
       STAT CARDS
    ----------------------------- */

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 8px 28px rgba(15,23,42,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(15,23,42,0.10);
    }

    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.7rem;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 850;
        color: #111827;
        line-height: 1;
    }

    .stat-label {
        margin-top: 0.45rem;
        color: #64748b;
        font-size: 0.9rem;
    }

    /* -----------------------------
       FUNDSTÜCK-KARTEN
    ----------------------------- */

    .item-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        overflow: hidden;
        margin-bottom: 0.6rem;

        box-shadow:
            0 7px 25px rgba(15,23,42,0.055);

        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease,
            border-color 0.22s ease;
    }

    .item-card:hover {
        transform: translateY(-7px);
        border-color: #a5b4fc;
        box-shadow:
            0 20px 45px rgba(79,70,229,0.13);
    }

    .item-image {
        width: 100%;
        height: 210px;
        object-fit: cover;
        display: block;
        background: #eef2f7;
    }

    .item-content {
        padding: 1.15rem;
    }

    .item-title {
        color: #111827;
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .item-description {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
        min-height: 44px;
        margin-bottom: 0.9rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #f1f5f9;
        color: #475569;
        border-radius: 999px;
        padding: 0.34rem 0.62rem;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .chip-accent {
        background: #eef2ff;
        color: #4338ca;
    }

    /* -----------------------------
       EMPTY STATE
    ----------------------------- */

    .empty-state {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        color: #64748b;
        margin-top: 1rem;
    }

    .empty-icon {
        font-size: 3rem;
        margin-bottom: 0.7rem;
    }

    .empty-title {
        color: #1e293b;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    /* -----------------------------
       FORMULAR
    ----------------------------- */

    .form-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 8px 28px rgba(15,23,42,0.05);
        margin-bottom: 1.2rem;
    }

    /* -----------------------------
       BUTTONS
    ----------------------------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #dbe1ea;
        font-weight: 700;
        min-height: 42px;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #818cf8;
        box-shadow: 0 7px 18px rgba(79,70,229,0.12);
    }

    /* -----------------------------
       INPUTS
    ----------------------------- */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        border-radius: 12px !important;
    }

    /* -----------------------------
       DETAIL HEADER
    ----------------------------- */

    .detail-header {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
    }

    .detail-title {
        font-size: 2rem;
        font-weight: 850;
        color: #111827;
        letter-spacing: -0.035em;
    }

    .detail-meta {
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* -----------------------------
       RESPONSIVE
    ----------------------------- */

    @media (max-width: 900px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .hero {
            padding: 2rem;
        }
    }

    @media (max-width: 600px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero {
            border-radius: 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATENBANK
# =========================================================

os.makedirs(DATA_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(DATA_PATH):
        return []

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_data(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


fundstuecke = load_data()


# =========================================================
# BILDER
# =========================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
    )
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(data):
    try:
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def get_item_image(item):
    # Neue Variante: Base64
    if item.get("image_data"):
        return base64_to_image(item["image_data"])

    # Alte Variante: Datei
    image_path = item.get("image_path")

    if image_path and os.path.exists(image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    return None


# =========================================================
# MODELL
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
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
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
        image = image.convert("RGB").resize((224, 224))

        array = np.asarray(image, dtype=np.float32)

        array = (array / 127.5) - 1.0
        array = np.expand_dims(array, axis=0)

        prediction = model.predict(array, verbose=0)[0]

        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        if index < len(labels):
            label = labels[index]
        else:
            label = f"Klasse {index}"

        return label, confidence

    except Exception:
        return "Unbekannt", 0.0


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None


def go_to(page):
    st.session_state.page = page


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 0.6rem 0.3rem 1.5rem 0.3rem;
        ">
            <div style="
                font-size: 1.45rem;
                font-weight: 850;
                color: white;
                letter-spacing: -0.03em;
            ">
                🔎 Fundbüro
            </div>

            <div style="
                color: #94a3b8;
                font-size: 0.85rem;
                margin-top: 0.35rem;
            ">
                Digital. Schnell. Übersichtlich.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Übersicht",
            "➕ Fundstück hinzufügen",
            "🔍 Fundstücke suchen",
        ],
        label_visibility="collapsed",
    )

    if page == "Übersicht":
        st.session_state.page = "Übersicht"

    elif page == "➕ Fundstück hinzufügen":
        st.session_state.page = "Hinzufügen"

    elif page == "🔍 Fundstücke suchen":
        st.session_state.page = "Suchen"

    st.divider()

    st.markdown(
        f"""
        <div style="
            color:#94a3b8;
            font-size:0.78rem;
            line-height:1.5;
        ">
            <b>{len(fundstuecke)}</b> Fundstück(e) gespeichert
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ DIGITALES FUNDBÜRO
        </div>

        <div class="hero-title">
            Fundstücke einfach verwalten.
        </div>

        <div class="hero-subtitle">
            Fundstücke fotografieren, automatisch erkennen lassen,
            speichern und später blitzschnell wiederfinden.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# ÜBERSICHT
# =========================================================

if st.session_state.page == "Übersicht":

    st.markdown(
        '<div class="section-title">Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Alles Wichtige auf einen Blick.</div>',
        unsafe_allow_html=True,
    )

    total = len(fundstuecke)

    categories = set()

    for item in fundstuecke:
        category = item.get("category", "")

        if category:
            categories.add(category)

    with_location = sum(
        1
        for item in fundstuecke
        if item.get("location")
    )

    with_image = sum(
        1
        for item in fundstuecke
        if item.get("image_data") or item.get("image_path")
    )

    st.markdown(
        f"""
        <div class="stats-grid">

            <div class="stat-card">
                <div class="stat-icon">📦</div>
                <div class="stat-number">{total}</div>
                <div class="stat-label">Fundstücke</div>
            </div>

            <div class="stat-card">
                <div class="stat-icon">🏷️</div>
                <div class="stat-number">{len(categories)}</div>
                <div class="stat-label">Kategorien</div>
            </div>

            <div class="stat-card">
                <div class="stat-icon">📍</div>
                <div class="stat-number">{with_location}</div>
                <div class="stat-label">Mit Fundort</div>
            </div>

            <div class="stat-card">
                <div class="stat-icon">📸</div>
                <div class="stat-number">{with_image}</div>
                <div class="stat-label">Mit Foto</div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "➕ Neues Fundstück",
            use_container_width=True,
        ):
            st.session_state.page = "Hinzufügen"
            st.rerun()

    with col2:
        if st.button(
            "🔍 Fundstücke durchsuchen",
            use_container_width=True,
        ):
            st.session_state.page = "Suchen"
            st.rerun()

    st.markdown(
        '<div class="section-title" style="margin-top:2rem;">Zuletzt hinzugefügt</div>',
        unsafe_allow_html=True,
    )

    if not fundstuecke:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">📦</div>

                <div class="empty-title">
                    Noch keine Fundstücke
                </div>

                <div>
                    Füge dein erstes Fundstück hinzu und starte
                    dein digitales Fundbüro.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        recent_items = list(reversed(fundstuecke[-6:]))

        cols = st.columns(3)

        for i, item in enumerate(recent_items):

            with cols[i % 3]:

                image = get_item_image(item)

                if image is not None:
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=85)
                    image_bytes = buffer.getvalue()

                    st.markdown(
                        f"""
                        <div class="item-card">

                            <img
                                class="item-image"
                                src="data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
                            />

                            <div class="item-content">

                                <div class="item-title">
                                    {html.escape(str(item.get("name", "Unbekanntes Fundstück")))}
                                </div>

                                <div class="item-description">
                                    {html.escape(str(item.get("description", "Keine Beschreibung vorhanden.")))[:150]}
                                </div>

                                <div class="chip-row">

                                    <span class="chip chip-accent">
                                        🏷️ {html.escape(str(item.get("category", "Unbekannt")))}
                                    </span>

                                    <span class="chip">
                                        📍 {html.escape(str(item.get("location", "Unbekannt")))}
                                    </span>

                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="item-card">

                            <div style="
                                height:210px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                background:#f1f5f9;
                                font-size:4rem;
                            ">
                                📦
                            </div>

                            <div class="item-content">

                                <div class="item-title">
                                    {html.escape(str(item.get("name", "Unbekanntes Fundstück")))}
                                </div>

                                <div class="item-description">
                                    {html.escape(str(item.get("description", "Keine Beschreibung vorhanden.")))[:150]}
                                </div>

                                <div class="chip-row">

                                    <span class="chip chip-accent">
                                        🏷️ {html.escape(str(item.get("category", "Unbekannt")))}
                                    </span>

                                    <span class="chip">
                                        📍 {html.escape(str(item.get("location", "Unbekannt")))}
                                    </span>

                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "Details ansehen →",
                    key=f"details_home_{i}",
                    use_container_width=True,
                ):
                    st.session_state.selected_item = fundstuecke.index(item)
                    st.session_state.page = "Details"
                    st.rerun()


# =========================================================
# FUNDSTÜCK HINZUFÜGEN
# =========================================================

elif st.session_state.page == "Hinzufügen":

    st.markdown(
        '<div class="section-title">➕ Fundstück hinzufügen</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Erfasse das Fundstück in wenigen Sekunden.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:

        st.markdown(
            '<div class="form-card">',
            unsafe_allow_html=True,
        )

        name = st.text_input(
            "Name des Fundstücks",
            placeholder="z. B. Schwarzer Rucksack",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Beschreibe das Fundstück möglichst genau...",
            height=130,
        )

        location = st.text_input(
            "📍 Fundort",
            placeholder="z. B. Sporthalle, Raum 12",
        )

        date = st.date_input(
            "📅 Funddatum",
            value=datetime.now().date(),
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="form-card">',
            unsafe_allow_html=True,
        )

        st.markdown("### 📸 Foto")

        image_source = st.radio(
            "Fotoquelle",
            ["Datei hochladen", "Kamera"],
            horizontal=True,
        )

        if image_source == "Datei hochladen":
            uploaded_file = st.file_uploader(
                "Bild auswählen",
                type=["jpg", "jpeg", "png"],
            )
        else:
            uploaded_file = st.camera_input(
                "Fundstück fotografieren"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    image = None

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown(
                "### 👀 Vorschau"
            )

            # Absichtlich ohne use_container_width,
            # damit die alte st.image-Problematik nicht wieder auftritt.
            st.image(
                image,
                caption="Ausgewähltes Foto",
            )

        except Exception:
            st.error("Das Bild konnte nicht gelesen werden.")

    # ---------------------------------------------
    # KI ERKENNUNG
    # ---------------------------------------------

    predicted_label = ""
    confidence = 0.0

    if image is not None:

        with st.spinner("🤖 KI analysiert das Fundstück..."):
            predicted_label, confidence = classify_image(image)

        if predicted_label != "Unbekannt":

            st.success(
                f"🤖 Erkennung: **{predicted_label}** "
                f"({confidence * 100:.1f} % Sicherheit)"
            )

    # ---------------------------------------------
    # SPEICHERN
    # ---------------------------------------------

    if st.button(
        "💾 Fundstück speichern",
        type="primary",
        use_container_width=True,
    ):

        if not name.strip():

            st.warning("Bitte gib dem Fundstück einen Namen.")

        elif image is None:

            st.warning("Bitte füge ein Foto hinzu.")

        else:

            new_item = {
                "name": name.strip(),
                "description": description.strip(),
                "location": location.strip(),
                "date": str(date),
                "category": predicted_label or "Unbekannt",
                "confidence": confidence,
                "image_data": image_to_base64(image),
            }

            fundstuecke.append(new_item)
            save_data(fundstuecke)

            st.success("✅ Fundstück wurde gespeichert!")

            st.session_state.page = "Übersicht"

            st.rerun()


# =========================================================
# SUCHEN
# =========================================================

elif st.session_state.page == "Suchen":

    st.markdown(
        '<div class="section-title">🔍 Fundstücke suchen</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Finde gespeicherte Fundstücke mit wenigen Klicks.</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Suche",
        placeholder="Name, Beschreibung, Fundort oder Kategorie...",
    )

    all_categories = sorted(
        set(
            item.get("category", "Unbekannt")
            for item in fundstuecke
        )
    )

    category_filter = st.selectbox(
        "Kategorie",
        ["Alle"] + all_categories,
    )

    filtered = []

    search_lower = search.lower().strip()

    for index, item in enumerate(fundstuecke):

        searchable = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("description", "")),
                str(item.get("location", "")),
                str(item.get("category", "")),
            ]
        ).lower()

        matches_search = (
            not search_lower
            or search_lower in searchable
        )

        matches_category = (
            category_filter == "Alle"
            or item.get("category", "Unbekannt") == category_filter
        )

        if matches_search and matches_category:
            filtered.append((index, item))

    st.markdown(
        f"**{len(filtered)}** Fundstück(e) gefunden"
    )

    if not filtered:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">🔎</div>

                <div class="empty-title">
                    Nichts gefunden
                </div>

                <div>
                    Versuch es mit einem anderen Suchbegriff
                    oder einer anderen Kategorie.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        cols = st.columns(3)

        for i, (original_index, item) in enumerate(filtered):

            with cols[i % 3]:

                image = get_item_image(item)

                if image is not None:

                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=85)
                    image_bytes = buffer.getvalue()

                    encoded = base64.b64encode(
                        image_bytes
                    ).decode()

                    st.markdown(
                        f"""
                        <div class="item-card">

                            <img
                                class="item-image"
                                src="data:image/jpeg;base64,{encoded}"
                            />

                            <div class="item-content">

                                <div class="item-title">
                                    {html.escape(str(item.get("name", "Unbekanntes Fundstück")))}
                                </div>

                                <div class="item-description">
                                    {html.escape(str(item.get("description", "Keine Beschreibung vorhanden.")))[:150]}
                                </div>

                                <div class="chip-row">

                                    <span class="chip chip-accent">
                                        🏷️ {html.escape(str(item.get("category", "Unbekannt")))}
                                    </span>

                                    <span class="chip">
                                        📍 {html.escape(str(item.get("location", "Unbekannt")))}
                                    </span>

                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="item-card">

                            <div style="
                                height:210px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                background:#f1f5f9;
                                font-size:4rem;
                            ">
                                📦
                            </div>

                            <div class="item-content">

                                <div class="item-title">
                                    {html.escape(str(item.get("name", "Unbekanntes Fundstück")))}
                                </div>

                                <div class="item-description">
                                    {html.escape(str(item.get("description", "Keine Beschreibung vorhanden.")))[:150]}
                                </div>

                                <div class="chip-row">

                                    <span class="chip chip-accent">
                                        🏷️ {html.escape(str(item.get("category", "Unbekannt")))}
                                    </span>

                                    <span class="chip">
                                        📍 {html.escape(str(item.get("location", "Unbekannt")))}
                                    </span>

                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "Details ansehen →",
                    key=f"details_search_{original_index}",
                    use_container_width=True,
                ):
                    st.session_state.selected_item = original_index
                    st.session_state.page = "Details"
                    st.rerun()


# =========================================================
# DETAILANSICHT
# =========================================================

elif st.session_state.page == "Details":

    index = st.session_state.selected_item

    if index is None or index >= len(fundstuecke):

        st.warning("Fundstück wurde nicht gefunden.")

        if st.button("← Zurück"):
            st.session_state.page = "Übersicht"
            st.rerun()

    else:

        item = fundstuecke[index]

        if st.button("← Zurück zu den Fundstücken"):
            st.session_state.page = "Übersicht"
            st.rerun()

        st.markdown(
            f"""
            <div class="detail-header">

                <div class="detail-title">
                    {html.escape(str(item.get("name", "Unbekanntes Fundstück")))}
                </div>

                <div class="detail-meta">
                    📅 {html.escape(str(item.get("date", "Kein Datum")))}
                    &nbsp;&nbsp; • &nbsp;&nbsp;
                    📍 {html.escape(str(item.get("location", "Kein Fundort")))}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.1, 0.9], gap="large")

        with left:

            image = get_item_image(item)

            if image is not None:

                st.image(
                    image,
                    caption="Foto des Fundstücks",
                )

            else:

                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="empty-icon">📦</div>
                        Kein Foto vorhanden.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:

            st.markdown("### 📝 Beschreibung")

            # Wichtig:
            # Die Beschreibung wird NICHT als HTML eingefügt.
            # Dadurch können < > & usw. die Seite nicht kaputt machen.
            st.write(
                item.get(
                    "description",
                    "Keine Beschreibung vorhanden.",
                )
                or "Keine Beschreibung vorhanden."
            )

            st.markdown("### 🏷️ Informationen")

            st.markdown(
                f"""
                <div class="chip-row">

                    <span class="chip chip-accent">
                        🏷️ {html.escape(str(item.get("category", "Unbekannt")))}
                    </span>

                    <span class="chip">
                        📍 {html.escape(str(item.get("location", "Unbekannt")))}
                    </span>

                    <span class="chip">
                        📅 {html.escape(str(item.get("date", "Unbekannt")))}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

            confidence = item.get("confidence", 0)

            if confidence:
                st.markdown("### 🤖 KI-Erkennung")

                st.progress(
                    min(max(float(confidence), 0.0), 1.0)
                )

                st.caption(
                    f"Erkennungssicherheit: {float(confidence) * 100:.1f} %"
                )

            st.divider()

            if st.button(
                "🗑️ Fundstück löschen",
                use_container_width=True,
            ):

                del fundstuecke[index]

                save_data(fundstuecke)

                st.session_state.selected_item = None
                st.session_state.page = "Übersicht"

                st.success("Fundstück wurde gelöscht.")

                st.rerun()
