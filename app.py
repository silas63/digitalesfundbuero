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

# ============================================================
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = "fundburo_data"
DATA_FILE = os.path.join(DATA_DIR, "fundstuecke.json")
MODEL_FILE = "keras_model.h5"
LABELS_FILE = "labels.txt"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f7f5;
    color: #171717;
}

/* Hauptbereich */
.block-container {
    max-width: 1180px;
    padding-top: 28px;
    padding-bottom: 80px;
}

/* Streamlit Header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------------------------------------------------
   BRAND
--------------------------------------------------------- */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 35px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: #171717;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
}

.brand-name {
    font-size: 18px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.5px;
}

.brand-small {
    font-size: 11px;
    color: #8b8b8b;
    margin-top: 4px;
    font-weight: 500;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero {
    background: #ffffff;
    border: 1px solid #e9e9e7;
    border-radius: 30px;
    padding: 55px 58px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: #f1f1ef;
    right: -90px;
    top: -110px;
}

.hero-kicker {
    position: relative;
    z-index: 2;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.7px;
    font-weight: 700;
    color: #777;
    margin-bottom: 17px;
}

.hero-title {
    position: relative;
    z-index: 2;
    font-size: clamp(46px, 6vw, 78px);
    line-height: 0.94;
    letter-spacing: -4px;
    font-weight: 800;
    color: #111;
    margin-bottom: 25px;
}

.hero-description {
    position: relative;
    z-index: 2;
    max-width: 540px;
    color: #707070;
    font-size: 16px;
    line-height: 1.65;
}


/* ---------------------------------------------------------
   STATS
--------------------------------------------------------- */

.stat-card {
    background: white;
    border: 1px solid #e8e8e5;
    border-radius: 22px;
    padding: 25px 27px;
    min-height: 125px;
    transition: all 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    border-color: #d5d5d2;
    box-shadow: 0 12px 30px rgba(0,0,0,0.06);
}

.stat-number {
    font-size: 36px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 10px;
}

.stat-label {
    color: #8a8a87;
    font-size: 13px;
    font-weight: 500;
}


/* ---------------------------------------------------------
   SECTION HEADER
--------------------------------------------------------- */

.section-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-top: 58px;
    margin-bottom: 22px;
}

.section-kicker {
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 6px;
}

.section-title {
    font-size: 31px;
    font-weight: 800;
    letter-spacing: -1.5px;
}


/* ---------------------------------------------------------
   FUND CARD
--------------------------------------------------------- */

.fund-card {
    background: #fff;
    border: 1px solid #e7e7e4;
    border-radius: 23px;
    overflow: hidden;
    margin-bottom: 10px;
    transition:
        transform 0.22s ease,
        box-shadow 0.22s ease,
        border-color 0.22s ease;
}

.fund-card:hover {
    transform: translateY(-7px);
    border-color: #d2d2ce;
    box-shadow: 0 20px 45px rgba(0,0,0,0.09);
}

.fund-image-wrap {
    width: 100%;
    height: 235px;
    overflow: hidden;
    background: #f0f0ee;
}

.fund-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.35s ease;
}

.fund-card:hover .fund-image {
    transform: scale(1.045);
}

.fund-info {
    padding: 20px 20px 22px 20px;
}

.fund-name {
    font-size: 18px;
    font-weight: 750;
    letter-spacing: -0.4px;
    margin-bottom: 13px;
    color: #171717;
}

.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.pill {
    display: inline-flex;
    align-items: center;
    border: 1px solid #e5e5e2;
    border-radius: 999px;
    padding: 6px 10px;
    color: #666;
    background: #fafaf9;
    font-size: 11px;
    font-weight: 600;
}

.pill-main {
    background: #171717;
    border-color: #171717;
    color: white;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #dededb !important;
    background: white !important;
    color: #171717 !important;
    font-weight: 600 !important;
    min-height: 43px;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    border-color: #171717 !important;
    background: #171717 !important;
    color: white !important;
    transform: translateY(-1px);
}


/* ---------------------------------------------------------
   INPUTS
--------------------------------------------------------- */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    border-radius: 13px !important;
    border-color: #dededb !important;
    background: white !important;
}

label {
    font-weight: 600 !important;
    color: #333 !important;
}


/* ---------------------------------------------------------
   UPLOAD
--------------------------------------------------------- */

section[data-testid="stFileUploaderDropzone"] {
    border-radius: 18px !important;
    border: 1px dashed #cfcfcb !important;
    background: #fff !important;
}


/* ---------------------------------------------------------
   DETAIL
--------------------------------------------------------- */

.detail-card {
    background: white;
    border: 1px solid #e7e7e4;
    border-radius: 25px;
    padding: 28px;
    margin-top: 15px;
}

.detail-title {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 18px;
}

.detail-label {
    color: #999;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 6px;
}

.detail-value {
    font-size: 15px;
    color: #444;
    line-height: 1.6;
}


/* ---------------------------------------------------------
   EMPTY STATE
--------------------------------------------------------- */

.empty {
    background: white;
    border: 1px dashed #d6d6d2;
    border-radius: 24px;
    padding: 55px 30px;
    text-align: center;
}

.empty-icon {
    font-size: 35px;
    margin-bottom: 12px;
}

.empty-title {
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 7px;
}

.empty-text {
    color: #888;
    font-size: 14px;
}


/* ---------------------------------------------------------
   NAVIGATION
--------------------------------------------------------- */

.nav-space {
    height: 5px;
}

div[role="radiogroup"] {
    gap: 7px;
}

div[role="radiogroup"] label {
    background: transparent !important;
    border-radius: 999px !important;
    padding: 7px 14px !important;
}

div[role="radiogroup"] label:hover {
    background: #ededeb !important;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 700px) {

    .block-container {
        padding-left: 17px;
        padding-right: 17px;
    }

    .hero {
        padding: 35px 27px;
        border-radius: 24px;
    }

    .hero-title {
        font-size: 48px;
        letter-spacing: -2.8px;
    }

    .fund-image-wrap {
        height: 210px;
    }

    .section-title {
        font-size: 27px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATEN
# ============================================================

def load_items():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


items = load_items()


# ============================================================
# BILDER
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=88,
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
    image_data = item.get("image_data")

    if image_data:
        return base64_to_image(image_data)

    old_path = item.get("image_path")

    if old_path and os.path.exists(old_path):
        try:
            return Image.open(old_path).convert("RGB")
        except Exception:
            pass

    return None


def image_as_data_uri(item):
    image_data = item.get("image_data")

    if image_data:
        return "data:image/jpeg;base64," + image_data

    old_path = item.get("image_path")

    if old_path and os.path.exists(old_path):
        try:
            with open(old_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            return "data:image/jpeg;base64," + encoded

        except Exception:
            pass

    return None


# ============================================================
# KI MODELL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    return tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )


@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_FILE):
        return []

    labels = []

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return labels


def classify_image(image):
    try:
        model = load_model()
        labels = load_labels()

        if model is None or not labels:
            return "Unbekannt", 0.0

        image = image.convert("RGB").resize((224, 224))

        array = np.asarray(image, dtype=np.float32)
        array = (array / 127.5) - 1.0
        array = np.expand_dims(array, axis=0)

        prediction = model.predict(array, verbose=0)[0]

        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        if index >= len(labels):
            return "Unbekannt", confidence

        return labels[index], confidence

    except Exception:
        return "Unbekannt", 0.0


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="brand">
    <div class="brand-icon">🔎</div>

    <div>
        <div class="brand-name">Fundbüro</div>
        <div class="brand-small">Digital</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    ["Übersicht", "Neues Fundstück", "Suche"],
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.page = page


# ============================================================
# DETAILANSICHT
# ============================================================

if st.session_state.selected_id is not None:

    selected = None

    for item in items:
        if str(item.get("id")) == str(st.session_state.selected_id):
            selected = item
            break

    if selected is not None:

        st.markdown(
            """
<div class="section-header">
    <div>
        <div class="section-kicker">Fundstück</div>
        <div class="section-title">Details</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.15, 0.85], gap="large")

        with col1:
            image = get_item_image(selected)

            if image is not None:
                st.image(image)

            else:
                st.info("Für dieses Fundstück ist kein Bild gespeichert.")

        with col2:

            safe_name = html.escape(
                str(selected.get("name") or "Fundstück")
            )

            safe_category = html.escape(
                str(selected.get("category") or "Unbekannt")
            )

            safe_location = html.escape(
                str(selected.get("location") or "Unbekannt")
            )

            safe_date = html.escape(
                str(selected.get("date") or "Unbekannt")
            )

            st.markdown(
                f"""
<div class="detail-card">

    <div class="detail-title">
        {safe_name}
    </div>

    <div class="pills">
        <span class="pill pill-main">{safe_category}</span>
        <span class="pill">📍 {safe_location}</span>
    </div>

    <div class="detail-label">
        Gefunden am
    </div>

    <div class="detail-value">
        {safe_date}
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="detail-label">Beschreibung</div>',
                unsafe_allow_html=True,
            )

            description = str(
                selected.get("description") or
                "Keine Beschreibung vorhanden."
            )

            # Bewusst als normaler Streamlit-Text:
            # User-Text kann dadurch kein HTML kaputt machen.
            st.write(description)

            st.write("")

            back_col, delete_col = st.columns(2)

            with back_col:
                if st.button(
                    "← Zur Übersicht",
                    key="back_details",
                    use_container_width=True,
                ):
                    st.session_state.selected_id = None
                    st.rerun()

            with delete_col:
                if st.button(
                    "Fundstück löschen",
                    key="delete_details",
                    use_container_width=True,
                ):
                    items = [
                        x for x in items
                        if str(x.get("id")) != str(selected.get("id"))
                    ]

                    save_items(items)
                    st.session_state.selected_id = None
                    st.rerun()

        st.stop()


# ============================================================
# ÜBERSICHT
# ============================================================

if page == "Übersicht":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(
        """
<div class="hero">

    <div class="hero-kicker">
        Digitales Fundbüro
    </div>

    <div class="hero-title">
        Gefunden.<br>
        Gespeichert.
    </div>

    <div class="hero-description">
        Fundstücke fotografieren, automatisch erkennen
        und später schnell wiederfinden.
    </div>

</div>
""",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    categories = {
        str(x.get("category"))
        for x in items
        if x.get("category")
    }

    locations = {
        str(x.get("location"))
        for x in items
        if x.get("location")
    }

    stat_cols = st.columns(3, gap="medium")

    stats = [
        (len(items), "Fundstücke"),
        (len(categories), "Kategorien"),
        (len(locations), "Fundorte"),
    ]

    for col, (number, label) in zip(stat_cols, stats):

        with col:
            st.markdown(
                f"""
<div class="stat-card">

    <div class="stat-number">
        {number}
    </div>

    <div class="stat-label">
        {label}
    </div>

</div>
""",
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # ZULETZT
    # --------------------------------------------------------

    st.markdown(
        """
<div class="section-header">

    <div>
        <div class="section-kicker">
            Fundstücke
        </div>

        <div class="section-title">
            Zuletzt
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )


    if not items:

        st.markdown(
            """
<div class="empty">

    <div class="empty-icon">🔎</div>

    <div class="empty-title">
        Noch keine Fundstücke
    </div>

    <div class="empty-text">
        Füge dein erstes Fundstück hinzu und es erscheint hier.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    else:

        recent_items = list(reversed(items[-6:]))

        for row_start in range(0, len(recent_items), 3):

            row = recent_items[row_start:row_start + 3]

            columns = st.columns(3, gap="medium")

            for col, item in zip(columns, row):

                with col:

                    image_uri = image_as_data_uri(item)

                    if image_uri:
                        image_html = (
                            '<div class="fund-image-wrap">'
                            '<img class="fund-image" src="'
                            + image_uri
                            + '">'
                            '</div>'
                        )
                    else:
                        image_html = (
                            '<div class="fund-image-wrap">'
                            '<div style="height:100%;display:flex;'
                            'align-items:center;justify-content:center;'
                            'font-size:35px;">🔎</div>'
                            '</div>'
                        )

                    name = html.escape(
                        str(item.get("name") or "Fundstück")
                    )

                    category = html.escape(
                        str(item.get("category") or "Unbekannt")
                    )

                    location = html.escape(
                        str(item.get("location") or "Unbekannt")
                    )

                    st.markdown(
                        f"""
<div class="fund-card">

    {image_html}

    <div class="fund-info">

        <div class="fund-name">
            {name}
        </div>

        <div class="pills">

            <span class="pill pill-main">
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
                        "Details →",
                        key=f"details_{item.get('id')}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = item.get("id")
                        st.rerun()


# ============================================================
# NEUES FUNDSTÜCK
# ============================================================

elif page == "Neues Fundstück":

    st.markdown(
        """
<div class="section-header">

    <div>
        <div class="section-kicker">
            Neu
        </div>

        <div class="section-title">
            Fundstück hinzufügen
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:

        st.markdown("### 1. Foto")

        uploaded = st.file_uploader(
            "Bild hochladen",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        camera = st.camera_input(
            "Oder direkt fotografieren",
        )

        image_source = camera if camera is not None else uploaded

        current_image = None

        if image_source is not None:

            try:
                current_image = Image.open(image_source).convert("RGB")

                st.image(current_image)

            except Exception:
                st.error("Das Bild konnte nicht gelesen werden.")

    with right:

        st.markdown("### 2. Informationen")

        name = st.text_input(
            "Name",
            placeholder="z. B. Schwarzer Rucksack",
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Weitere Merkmale oder Hinweise …",
            height=120,
        )

        st.markdown("")

        if st.button(
            "Fundstück speichern",
            use_container_width=True,
        ):

            if current_image is None:
                st.warning("Bitte zuerst ein Foto auswählen.")

            elif not location.strip():
                st.warning("Bitte einen Fundort eintragen.")

            else:

                with st.spinner("Bild wird analysiert …"):
                    category, confidence = classify_image(current_image)

                if not name.strip():
                    name = "Fundstück"

                new_item = {
                    "id": datetime.now().strftime(
                        "%Y%m%d%H%M%S%f"
                    ),
                    "name": name.strip(),
                    "category": category,
                    "location": location.strip(),
                    "date": datetime.now().strftime("%d.%m.%Y"),
                    "description": description.strip(),
                    "image_data": image_to_base64(current_image),
                    "confidence": round(confidence, 4),
                }

                items.append(new_item)
                save_items(items)

                st.success(
                    f"Gespeichert · erkannt als „{category}“"
                )

                st.session_state.page = "Übersicht"

                st.rerun()


# ============================================================
# SUCHE
# ============================================================

elif page == "Suche":

    st.markdown(
        """
<div class="section-header">

    <div>
        <div class="section-kicker">
            Finden
        </div>

        <div class="section-title">
            Fundstücke suchen
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Suche",
        placeholder="Name, Kategorie oder Fundort …",
        label_visibility="collapsed",
    )

    search_clean = search.lower().strip()

    if search_clean:

        results = []

        for item in items:

            searchable = " ".join([
                str(item.get("name", "")),
                str(item.get("category", "")),
                str(item.get("location", "")),
                str(item.get("description", "")),
            ]).lower()

            if search_clean in searchable:
                results.append(item)

    else:

        results = items


    if not results:

        st.markdown(
            """
<div class="empty">

    <div class="empty-icon">⌕</div>

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

        st.caption(
            f"{len(results)} Fundstück"
            + ("" if len(results) == 1 else "e")
            + " gefunden"
        )

        for row_start in range(0, len(results), 3):

            row = results[row_start:row_start + 3]

            columns = st.columns(3, gap="medium")

            for col, item in zip(columns, row):

                with col:

                    image_uri = image_as_data_uri(item)

                    if image_uri:

                        image_html = (
                            '<div class="fund-image-wrap">'
                            '<img class="fund-image" src="'
                            + image_uri
                            + '">'
                            '</div>'
                        )

                    else:

                        image_html = (
                            '<div class="fund-image-wrap">'
                            '<div style="height:100%;display:flex;'
                            'align-items:center;justify-content:center;'
                            'font-size:35px;">🔎</div>'
                            '</div>'
                        )

                    name = html.escape(
                        str(item.get("name") or "Fundstück")
                    )

                    category = html.escape(
                        str(item.get("category") or "Unbekannt")
                    )

                    location = html.escape(
                        str(item.get("location") or "Unbekannt")
                    )

                    st.markdown(
                        f"""
<div class="fund-card">

    {image_html}

    <div class="fund-info">

        <div class="fund-name">
            {name}
        </div>

        <div class="pills">

            <span class="pill pill-main">
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
                        "Details →",
                        key=f"search_details_{item.get('id')}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = item.get("id")
                        st.rerun()
