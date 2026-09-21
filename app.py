import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64
import io
from datetime import datetime
import html

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
    .stApp {
        background: #f7f7f5;
        color: #171717;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 28px;
        padding-bottom: 70px;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* BRAND */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
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
        color: #888;
        margin-top: 4px;
        font-weight: 500;
    }

    /* NAVIGATION */
    div[role="radiogroup"] {
        gap: 6px;
        margin-bottom: 25px;
    }

    div[role="radiogroup"] label {
        border-radius: 999px !important;
        padding: 7px 14px !important;
    }

    /* HERO */
    .hero {
        background: white;
        border: 1px solid #e8e8e5;
        border-radius: 28px;
        padding: 48px 52px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: #f0f0ed;
        right: -90px;
        top: -110px;
    }

    .hero-kicker,
    .section-kicker {
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 10px;
        font-weight: 700;
    }

    .hero-kicker {
        margin-bottom: 15px;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        font-size: clamp(44px, 6vw, 76px);
        line-height: .95;
        letter-spacing: -4px;
        font-weight: 800;
        margin-bottom: 22px;
        color: #111;
    }

    .hero-description {
        position: relative;
        z-index: 1;
        max-width: 540px;
        color: #707070;
        font-size: 16px;
        line-height: 1.6;
    }

    /* STATS */
    .stat {
        background: white;
        border: 1px solid #e8e8e5;
        border-radius: 20px;
        padding: 22px 24px;
        min-height: 112px;
    }

    .stat-number {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -2px;
    }

    .stat-label {
        color: #888;
        font-size: 13px;
        margin-top: 7px;
    }

    /* SECTIONS */
    .section {
        margin-top: 48px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-top: 5px;
    }

    /* CARDS */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-color: #e7e7e4 !important;
        border-radius: 22px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #cfcfcb !important;
        box-shadow: 0 14px 34px rgba(0,0,0,.07);
    }

    /* BUTTONS */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid #dededb !important;
        background: white !important;
        color: #171717 !important;
        font-weight: 600 !important;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #171717 !important;
        background: #171717 !important;
        color: white !important;
    }

    /* INPUTS */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border-color: #dededb !important;
        background: white !important;
    }

    /* EMPTY */
    .empty {
        background: white;
        border: 1px dashed #d5d5d1;
        border-radius: 22px;
        padding: 48px 25px;
        text-align: center;
    }

    .empty-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }

    .empty-title {
        font-size: 20px;
        font-weight: 800;
    }

    .empty-text {
        color: #888;
        margin-top: 6px;
    }

    /* PILLS */
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin: 10px 0 4px 0;
    }

    .pill {
        display: inline-block;
        border: 1px solid #e4e4e1;
        background: #fafaf9;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        color: #666;
        font-weight: 600;
    }

    .pill-main {
        background: #171717;
        border-color: #171717;
        color: white;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 16px;
            padding-right: 16px;
        }

        .hero {
            padding: 34px 27px;
            border-radius: 23px;
        }

        .hero-title {
            font-size: 48px;
            letter-spacing: -3px;
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

        return data if isinstance(data, list) else []

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
    data = item.get("image_data")

    if data:
        image = base64_to_image(data)

        if image is not None:
            return image

    old_path = item.get("image_path")

    if old_path and os.path.exists(old_path):
        try:
            return Image.open(old_path).convert("RGB")
        except Exception:
            pass

    return None


# ============================================================
# KI
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_FILE,
            compile=False,
        )
    except Exception:
        return None


@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_FILE):
        return []

    labels = []

    try:
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

    except Exception:
        return []

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

        prediction = model.predict(
            array,
            verbose=0
        )[0]

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
    [
        "Übersicht",
        "Neues Fundstück",
        "Suche"
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.page = page


# ============================================================
# DETAILANSICHT
# ============================================================

if st.session_state.selected_id is not None:

    selected = next(
        (
            item
            for item in items
            if str(item.get("id"))
            == str(st.session_state.selected_id)
        ),
        None,
    )

    if selected is not None:

        st.markdown(
            """
            <div class="section">
                <div class="section-kicker">
                    Fundstück
                </div>

                <div class="section-title">
                    Details
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(
            [1.15, 0.85],
            gap="large"
        )

        with left:

            image = get_item_image(selected)

            if image is not None:
                st.image(
                    image,
                    use_container_width=True
                )
            else:
                st.info(
                    "Für dieses Fundstück ist kein Bild gespeichert."
                )

        with right:

            name = str(
                selected.get("name")
                or "Fundstück"
            )

            category = str(
                selected.get("category")
                or "Unbekannt"
            )

            location = str(
                selected.get("location")
                or "Unbekannt"
            )

            date = str(
                selected.get("date")
                or "Unbekannt"
            )

            description = str(
                selected.get("description")
                or "Keine Beschreibung vorhanden."
            )

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #e7e7e4;
                    border-radius:22px;
                    padding:25px;
                ">

                    <div style="
                        font-size:34px;
                        font-weight:800;
                        letter-spacing:-1.5px;
                        margin-bottom:15px;
                    ">
                        {html.escape(name)}
                    </div>

                    <div class="pill-row">

                        <span class="pill pill-main">
                            {html.escape(category)}
                        </span>

                        <span class="pill">
                            📍 {html.escape(location)}
                        </span>

                    </div>

                    <div style="
                        margin-top:24px;
                        color:#999;
                        font-size:11px;
                        text-transform:uppercase;
                        letter-spacing:1.2px;
                        font-weight:700;
                    ">
                        Gefunden am
                    </div>

                    <div style="
                        margin-top:5px;
                        color:#444;
                    ">
                        {html.escape(date)}
                    </div>

                    <div style="
                        margin-top:24px;
                        color:#999;
                        font-size:11px;
                        text-transform:uppercase;
                        letter-spacing:1.2px;
                        font-weight:700;
                    ">
                        Beschreibung
                    </div>

                    <div style="
                        margin-top:5px;
                        color:#444;
                        line-height:1.6;
                    ">
                        {html.escape(description)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            back, delete = st.columns(2)

            with back:

                if st.button(
                    "← Zur Übersicht",
                    key="back_details",
                    use_container_width=True,
                ):

                    st.session_state.selected_id = None
                    st.session_state.page = "Übersicht"
                    st.rerun()

            with delete:

                if st.button(
                    "Fundstück löschen",
                    key="delete_details",
                    use_container_width=True,
                ):

                    items = [
                        x
                        for x in items
                        if str(x.get("id"))
                        != str(selected.get("id"))
                    ]

                    save_items(items)

                    st.session_state.selected_id = None
                    st.session_state.page = "Übersicht"

                    st.rerun()

    else:
        st.session_state.selected_id = None

    st.stop()


# ============================================================
# ÜBERSICHT
# ============================================================

if page == "Übersicht":

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

    stats = [
        (len(items), "Fundstücke"),
        (len(categories), "Kategorien"),
        (len(locations), "Fundorte"),
    ]

    stat_cols = st.columns(3)

    for col, (number, label) in zip(
        stat_cols,
        stats
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat">

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

    st.markdown(
        """
        <div class="section">

            <div class="section-kicker">
                Fundstücke
            </div>

            <div class="section-title">
                Zuletzt
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not items:

        st.markdown(
            """
            <div class="empty">

                <div class="empty-icon">
                    🔎
                </div>

                <div class="empty-title">
                    Noch keine Fundstücke
                </div>

                <div class="empty-text">
                    Füge dein erstes Fundstück hinzu
                    und es erscheint hier.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        recent_items = list(
            reversed(items[-6:])
        )

        for start in range(
            0,
            len(recent_items),
            3
        ):

            row = recent_items[
                start:start + 3
            ]

            columns = st.columns(
                3,
                gap="medium"
            )

            for col, item in zip(
                columns,
                row
            ):

                with col:

                    with st.container(
                        border=True
                    ):

                        image = get_item_image(item)

                        if image is not None:

                            st.image(
                                image,
                                use_container_width=True
                            )

                        else:

                            st.markdown(
                                """
                                <div style="
                                    height:180px;
                                    display:flex;
                                    align-items:center;
                                    justify-content:center;
                                    background:#f1f1ef;
                                    border-radius:15px;
                                    font-size:34px;
                                ">
                                    🔎
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        name = str(
                            item.get("name")
                            or "Fundstück"
                        )

                        category = str(
                            item.get("category")
                            or "Unbekannt"
                        )

                        location = str(
                            item.get("location")
                            or "Unbekannt"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                font-size:18px;
                                font-weight:800;
                                margin-top:12px;
                                margin-bottom:7px;
                            ">
                                {html.escape(name)}
                            </div>

                            <div class="pill-row">

                                <span class="pill pill-main">
                                    {html.escape(category)}
                                </span>

                                <span class="pill">
                                    📍 {html.escape(location)}
                                </span>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "Details →",
                            key=f"details_{item.get('id')}",
                            use_container_width=True,
                        ):

                            st.session_state.selected_id = item.get(
                                "id"
                            )

                            st.rerun()


# ============================================================
# NEUES FUNDSTÜCK
# ============================================================

elif page == "Neues Fundstück":

    st.markdown(
        """
        <div class="section">

            <div class="section-kicker">
                Neu
            </div>

            <div class="section-title">
                Fundstück hinzufügen
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        2,
        gap="large"
    )

    with left:

        st.subheader("1. Foto")

        uploaded = st.file_uploader(
            "Bild hochladen",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
        )

        camera = st.camera_input(
            "Oder direkt fotografieren"
        )

        image_source = (
            camera
            if camera is not None
            else uploaded
        )

        current_image = None

        if image_source is not None:

            try:

                current_image = Image.open(
                    image_source
                ).convert("RGB")

                st.image(
                    current_image,
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "Das Bild konnte nicht gelesen werden."
                )

    with right:

        st.subheader("2. Informationen")

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

        if st.button(
            "Fundstück speichern",
            use_container_width=True,
        ):

            if current_image is None:

                st.warning(
                    "Bitte zuerst ein Foto auswählen."
                )

            elif not location.strip():

                st.warning(
                    "Bitte einen Fundort eintragen."
                )

            else:

                with st.spinner(
                    "Bild wird analysiert …"
                ):

                    category, confidence = classify_image(
                        current_image
                    )

                new_item = {
                    "id": datetime.now().strftime(
                        "%Y%m%d%H%M%S%f"
                    ),

                    "name": (
                        name.strip()
                        or "Fundstück"
                    ),

                    "category": category,

                    "location": location.strip(),

                    "date": datetime.now().strftime(
                        "%d.%m.%Y"
                    ),

                    "description": (
                        description.strip()
                    ),

                    "image_data": image_to_base64(
                        current_image
                    ),

                    "confidence": round(
                        confidence,
                        4
                    ),
                }

                items.append(new_item)

                save_items(items)

                st.session_state.page = "Übersicht"

                st.success(
                    f"Gespeichert · erkannt als „{category}“"
                )

                st.rerun()


# ============================================================
# SUCHE
# ============================================================

elif page == "Suche":

    st.markdown(
        """
        <div class="section">

            <div class="section-kicker">
                Finden
            </div>

            <div class="section-title">
                Fundstücke suchen
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Suche",
        placeholder=(
            "Name, Kategorie, Fundort "
            "oder Beschreibung …"
        ),
        label_visibility="collapsed",
    )

    query = search.lower().strip()

    if query:

        results = [
            item
            for item in items
            if query in " ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("category", "")),
                    str(item.get("location", "")),
                    str(item.get("description", "")),
                ]
            ).lower()
        ]

    else:

        results = items

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

        st.caption(
            f"{len(results)} Fundstück"
            + (
                ""
                if len(results) == 1
                else "e"
            )
            + " gefunden"
        )

        for start in range(
            0,
            len(results),
            3
        ):

            row = results[
                start:start + 3
            ]

            columns = st.columns(
                3,
                gap="medium"
            )

            for col, item in zip(
                columns,
                row
            ):

                with col:

                    with st.container(
                        border=True
                    ):

                        image = get_item_image(item)

                        if image is not None:

                            st.image(
                                image,
                                use_container_width=True
                            )

                        else:

                            st.markdown(
                                """
                                <div style="
                                    height:180px;
                                    display:flex;
                                    align-items:center;
                                    justify-content:center;
                                    background:#f1f1ef;
                                    border-radius:15px;
                                    font-size:34px;
                                ">
                                    🔎
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        name = str(
                            item.get("name")
                            or "Fundstück"
                        )

                        category = str(
                            item.get("category")
                            or "Unbekannt"
                        )

                        location = str(
                            item.get("location")
                            or "Unbekannt"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                font-size:18px;
                                font-weight:800;
                                margin-top:12px;
                                margin-bottom:7px;
                            ">
                                {html.escape(name)}
                            </div>

                            <div class="pill-row">

                                <span class="pill pill-main">
                                    {html.escape(category)}
                                </span>

                                <span class="pill">
                                    📍 {html.escape(location)}
                                </span>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "Details →",
                            key=f"search_details_{item.get('id')}",
                            use_container_width=True,
                        ):

                            st.session_state.selected_id = item.get(
                                "id"
                            )

                            st.rerun()
