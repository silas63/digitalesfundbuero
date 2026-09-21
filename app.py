import base64
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# KONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

DATA_DIR = Path("fundburo_data")
ITEMS_FILE = DATA_DIR / "fundstuecke.json"

DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# DESIGN
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(99, 102, 241, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(14, 165, 233, 0.12),
                transparent 25%
            ),
            #f7f8fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-box {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #312e81 50%,
            #075985 100%
        );

        border-radius: 28px;
        padding: 40px;
        margin-bottom: 25px;

        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.20);

        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 44px !important;
        font-weight: 900 !important;
        line-height: 1.15 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .hero-subtitle {
        color: #e0e7ff !important;
        font-size: 18px !important;
        line-height: 1.5 !important;
        margin-top: 12px !important;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 20px;
        min-height: 110px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.06);
    }

    .stat-number {
        color: #111827 !important;
        font-size: 30px !important;
        font-weight: 900 !important;
    }

    .stat-label {
        color: #6b7280 !important;
        font-size: 14px !important;
        margin-top: 5px;
    }

    .item-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3 !important;
        font-size: 13px;
        font-weight: 800;
    }

    .muted {
        color: #6b7280 !important;
        font-size: 14px;
        line-height: 1.8;
    }

    .info-box {
        padding: 18px;
        border-radius: 17px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a !important;
    }

    .footer {
        text-align: center;
        color: #9ca3af !important;
        margin-top: 50px;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 13px;
        font-weight: 750;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATENBANK
# =========================================================

def load_items():
    if not ITEMS_FILE.exists():
        return []

    try:
        with open(
            ITEMS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    DATA_DIR.mkdir(exist_ok=True)

    temp_file = DATA_DIR / "fundstuecke.tmp"

    try:
        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                items,
                file,
                ensure_ascii=False,
                indent=2
            )

        temp_file.replace(ITEMS_FILE)

    except Exception as error:
        st.error(
            f"❌ Fehler beim Speichern: {error}"
        )


# =========================================================
# BILDER
# =========================================================

def image_to_base64(image):
    try:
        image = image.convert("RGB")

        image.thumbnail((1200, 1200))

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=82,
            optimize=True
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except Exception:
        return None


def base64_to_image(data):
    if not data:
        return None

    try:
        raw = base64.b64decode(data)

        image = Image.open(
            BytesIO(raw)
        )

        return image.convert("RGB")

    except Exception:
        return None


def show_image(image, caption=None):
    if image is None:
        st.info("📷 Kein Bild vorhanden.")
        return

    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        image_bytes = buffer.getvalue()

        if caption:
            st.image(
                image_bytes,
                caption=caption
            )
        else:
            st.image(image_bytes)

    except Exception:
        st.info(
            "📷 Bild konnte nicht angezeigt werden."
        )


def get_image_for_item(item):
    image_data = item.get("image_data")

    if image_data:
        image = base64_to_image(image_data)

        if image is not None:
            return image

    old_path = item.get("image_path")

    if old_path:
        path = Path(old_path)

        if path.exists():
            try:
                return Image.open(
                    path
                ).convert("RGB")
            except Exception:
                pass

    return None


# =========================================================
# KI
# =========================================================

def load_labels():
    if not LABELS_PATH.exists():
        return []

    try:
        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            labels = []

            for line in file:
                value = line.strip()

                if not value:
                    continue

                parts = value.split(
                    maxsplit=1
                )

                if (
                    len(parts) == 2
                    and parts[0].isdigit()
                ):
                    value = parts[1]

                labels.append(value)

            return labels

    except Exception:
        return []


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
    except Exception:
        return None


def classify_image(image):
    model = load_model()
    labels = load_labels()

    if model is None:
        return (
            "Modell nicht gefunden",
            0.0
        )

    try:
        image = image.convert("RGB")

        image = image.resize(
            (224, 224)
        )

        array = np.asarray(
            image
        ).astype(np.float32)

        array = (
            array / 127.5
        ) - 1.0

        array = np.expand_dims(
            array,
            axis=0
        )

        prediction = model.predict(
            array,
            verbose=0
        )

        probabilities = prediction[0]

        index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[index]
        )

        if index < len(labels):
            label = labels[index]
        else:
            label = f"Klasse {index + 1}"

        return (
            label,
            confidence
        )

    except Exception:
        return (
            "Erkennung fehlgeschlagen",
            0.0
        )


# =========================================================
# KATEGORIEN
# =========================================================

def get_category(label):
    text = str(label).lower()

    if any(
        word in text
        for word in [
            "rucksack",
            "tasche",
            "beutel",
            "handtasche",
            "schulranzen"
        ]
    ):
        return "🎒 Tasche / Rucksack"

    if any(
        word in text
        for word in [
            "flasche",
            "trinkflasche",
            "wasserflasche"
        ]
    ):
        return "🥤 Flasche"

    if any(
        word in text
        for word in [
            "schlüssel",
            "key"
        ]
    ):
        return "🔑 Schlüssel"

    if any(
        word in text
        for word in [
            "jacke",
            "mantel",
            "hoodie",
            "pullover",
            "shirt",
            "tshirt",
            "kleidung"
        ]
    ):
        return "👕 Kleidung"

    if any(
        word in text
        for word in [
            "schuh",
            "sneaker",
            "turnschuh"
        ]
    ):
        return "👟 Schuhe"

    if any(
        word in text
        for word in [
            "handy",
            "smartphone",
            "telefon"
        ]
    ):
        return "📱 Handy"

    if any(
        word in text
        for word in [
            "kopfhörer",
            "headset",
            "airpods"
        ]
    ):
        return "🎧 Kopfhörer"

    if any(
        word in text
        for word in [
            "brille",
            "sonnenbrille"
        ]
    ):
        return "👓 Brille"

    if any(
        word in text
        for word in [
            "buch",
            "heft",
            "ordner",
            "block"
        ]
    ):
        return "📚 Schule"

    return "📦 Sonstiges"


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def format_date(value):
    if not value:
        return "Unbekannt"

    try:
        date = datetime.fromisoformat(
            value
        )

        return date.strftime(
            "%d.%m.%Y, %H:%M Uhr"
        )

    except Exception:
        return str(value)


def item_matches_search(
    item,
    search
):
    search = search.lower().strip()

    if not search:
        return True

    fields = [
        item.get("category", ""),
        item.get("ai_label", ""),
        item.get("location", ""),
        item.get("size", ""),
        item.get("color", ""),
        item.get("description", "")
    ]

    combined = " ".join(
        str(field)
        for field in fields
    ).lower()

    return search in combined


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

if "ai_label" not in st.session_state:
    st.session_state.ai_label = ""

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = 0.0

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False


items = load_items()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-title">
            🔎 Digitales Fundbüro
        </div>

        <div class="hero-subtitle">
            Fundstücke digital erfassen, automatisch erkennen
            und schnell wiederfinden.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button(
        "🏠 Übersicht",
        use_container_width=True
    ):
        st.session_state.page = "Übersicht"
        st.rerun()

with nav2:
    if st.button(
        "➕ Neuer Fund",
        use_container_width=True
    ):
        st.session_state.page = "Neuer Fund"
        st.rerun()

with nav3:
    if st.button(
        "🔍 Suchen",
        use_container_width=True
    ):
        st.session_state.page = "Suchen"
        st.rerun()

with nav4:
    if st.button(
        "📋 Details",
        use_container_width=True
    ):
        st.session_state.page = "Details"
        st.rerun()


st.divider()


# =========================================================
# ÜBERSICHT
# =========================================================

if st.session_state.page == "Übersicht":

    st.subheader("📊 Übersicht")

    total = len(items)

    categories = set()

    for item in items:
        category = item.get("category")

        if category:
            categories.add(category)

    model_active = load_model() is not None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {total}
                </div>
                <div class="stat-label">
                    Gespeicherte Fundstücke
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {len(categories)}
                </div>
                <div class="stat-label">
                    Kategorien
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        status = (
            "🟢 Aktiv"
            if model_active
            else "🔴 Nicht gefunden"
        )

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {status}
                </div>
                <div class="stat-label">
                    KI-Erkennung
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    if not items:

        st.markdown(
            """
            <div class="info-box">
                <strong>👋 Noch keine Fundstücke!</strong><br><br>
                Klicke auf „➕ Neuer Fund“, um dein
                erstes Fundstück einzutragen.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.subheader(
            "🆕 Zuletzt hinzugefügt"
        )

        recent_items = list(
            reversed(items[-8:])
        )

        for item in recent_items:

            item_id = item.get("id")

            category = item.get(
                "category",
                "📦 Sonstiges"
            )

            location = item.get(
                "location",
                "Unbekannt"
            )

            description = item.get(
                "description",
                ""
            )

            date = format_date(
                item.get("date")
            )

            st.markdown(
                '<div class="item-card">',
                unsafe_allow_html=True
            )

            left, right = st.columns(
                [1, 2]
            )

            with left:
                image = get_image_for_item(
                    item
                )

                show_image(image)

            with right:

                st.markdown(
                    f'<span class="badge">{category}</span>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"### {category}"
                )

                if description:
                    st.write(description)
                else:
                    st.write(
                        "Keine Beschreibung angegeben."
                    )

                st.markdown(
                    f"""
                    <div class="muted">
                        📍 {location}<br>
                        🕒 {date}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "📋 Details anzeigen",
                    key=f"overview_{item_id}"
                ):

                    st.session_state.selected_item_id = (
                        item_id
                    )

                    st.session_state.page = (
                        "Details"
                    )

                    st.rerun()

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# =========================================================
# NEUER FUND
# =========================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader(
        "➕ Neues Fundstück"
    )

    st.write(
        "Foto aufnehmen oder hochladen und anschließend "
        "von der KI erkennen lassen."
    )

    upload_tab, camera_tab = st.tabs(
        [
            "📁 Bild hochladen",
            "📷 Kamera"
        ]
    )

    uploaded_file = None
    camera_image = None

    with upload_tab:

        uploaded_file = st.file_uploader(
            "Bild auswählen",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

    with camera_tab:

        camera_image = st.camera_input(
            "Foto aufnehmen"
        )

    image_source = (
        uploaded_file
        if uploaded_file is not None
        else camera_image
    )

    selected_image = None

    if image_source is not None:

        try:

            selected_image = Image.open(
                image_source
            ).convert("RGB")

            st.markdown(
                "### 🖼️ Bildvorschau"
            )

            show_image(
                selected_image,
                caption="Ausgewähltes Fundstück"
            )

        except Exception:

            st.error(
                "❌ Das Bild konnte nicht gelesen werden."
            )

    if selected_image is not None:

        st.markdown(
            "### 🤖 KI-Erkennung"
        )

        if st.button(
            "✨ Fundstück erkennen",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🔎 KI analysiert das Bild ..."
            ):

                label, confidence = (
                    classify_image(
                        selected_image
                    )
                )

            st.session_state.ai_label = label

            st.session_state.ai_confidence = (
                confidence
            )

            st.success(
                "✅ Analyse abgeschlossen!"
            )

        ai_label = st.session_state.ai_label

        ai_confidence = (
            st.session_state.ai_confidence
        )

        if ai_label:

            category = get_category(
                ai_label
            )

            result1, result2 = st.columns(2)

            with result1:
                st.info(
                    f"🤖 Erkennung: **{ai_label}**"
                )

            with result2:
                st.info(
                    "🎯 Sicherheit: "
                    f"**{ai_confidence * 100:.1f}%**"
                )

            st.success(
                f"📦 Kategorie: **{category}**"
            )

        st.divider()

        st.markdown(
            "### 📝 Angaben"
        )

        location = st.text_input(
            "📍 Fundort",
            placeholder=(
                "z. B. Sporthalle, Schulhof, Raum 204"
            )
        )

        size = st.text_input(
            "📏 Größe",
            placeholder=(
                "z. B. klein, mittel, groß, M"
            )
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder=(
                "z. B. schwarz, blau, rot"
            )
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder=(
                "Weitere Merkmale oder Besonderheiten ..."
            ),
            height=120
        )

        st.write("")

        if st.button(
            "💾 Fundstück speichern",
            type="primary",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "⚠️ Bitte gib den Fundort an."
                )

            else:

                image_data = image_to_base64(
                    selected_image
                )

                if image_data is None:

                    st.error(
                        "❌ Bild konnte nicht gespeichert werden."
                    )

                else:

                    final_label = (
                        st.session_state.ai_label
                        if st.session_state.ai_label
                        else "Nicht erkannt"
                    )

                    final_category = get_category(
                        final_label
                    )

                    new_item = {
                        "id": str(
                            uuid.uuid4()
                        ),
                        "category": final_category,
                        "ai_label": final_label,
                        "ai_confidence": (
                            st.session_state.ai_confidence
                        ),
                        "location": location.strip(),
                        "size": size.strip(),
                        "color": color.strip(),
                        "description": description.strip(),
                        "image_data": image_data,
                        "date": datetime.now().isoformat()
                    }

                    items.append(
                        new_item
                    )

                    save_items(items)

                    st.session_state.selected_item_id = (
                        new_item["id"]
                    )

                    st.session_state.ai_label = ""
                    st.session_state.ai_confidence = 0.0

                    st.session_state.page = "Details"

                    st.balloons()

                    st.rerun()


# =========================================================
# SUCHE
# =========================================================

elif st.session_state.page == "Suchen":

    st.subheader(
        "🔍 Fundstücke suchen"
    )

    search = st.text_input(
        "Suchbegriff",
        placeholder=(
            "Rucksack, schwarz, Sporthalle, Schlüssel ..."
        )
    )

    all_categories = sorted(
        set(
            item.get(
                "category",
                "📦 Sonstiges"
            )
            for item in items
        )
    )

    category_filter = st.selectbox(
        "📦 Kategorie",
        [
            "Alle Kategorien",
            *all_categories
        ]
    )

    results = []

    for item in items:

        if not item_matches_search(
            item,
            search
        ):
            continue

        if (
            category_filter != "Alle Kategorien"
            and item.get("category")
            != category_filter
        ):
            continue

        results.append(item)

    st.write(
        f"**{len(results)} Fundstück(e) gefunden.**"
    )

    if not results:

        st.info(
            "🔎 Keine passenden Fundstücke gefunden."
        )

    else:

        for item in reversed(results):

            item_id = item.get("id")

            category = item.get(
                "category",
                "📦 Sonstiges"
            )

            location = item.get(
                "location",
                "Unbekannt"
            )

            color = item.get(
                "color",
                ""
            )

            size = item.get(
                "size",
                ""
            )

            description = item.get(
                "description",
                ""
            )

            st.markdown(
                '<div class="item-card">',
                unsafe_allow_html=True
            )

            left, right = st.columns(
                [1, 2]
            )

            with left:

                image = get_image_for_item(
                    item
                )

                show_image(image)

            with right:

                st.markdown(
                    f'<span class="badge">{category}</span>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"### {category}"
                )

                if description:
                    st.write(description)

                details = []

                if location:
                    details.append(
                        f"📍 {location}"
                    )

                if color:
                    details.append(
                        f"🎨 {color}"
                    )

                if size:
                    details.append(
                        f"📏 {size}"
                    )

                if details:
                    st.write(
                        " • ".join(details)
                    )

                if st.button(
                    "📋 Details",
                    key=f"search_{item_id}"
                ):

                    st.session_state.selected_item_id = (
                        item_id
                    )

                    st.session_state.page = (
                        "Details"
                    )

                    st.rerun()

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# =========================================================
# DETAILS
# =========================================================

elif st.session_state.page == "Details":

    st.subheader(
        "📋 Fundstück-Details"
    )

    selected_item = None

    selected_id = (
        st.session_state.selected_item_id
    )

    if selected_id:

        for item in items:

            if item.get("id") == selected_id:

                selected_item = item
                break

    if selected_item is None:

        if not items:

            st.info(
                "ℹ️ Es gibt noch keine Fundstücke."
            )

        else:

            options = {}

            for item in items:

                item_id = item.get("id")

                category = item.get(
                    "category",
                    "📦 Sonstiges"
                )

                location = item.get(
                    "location",
                    "Unbekannt"
                )

                label = (
                    f"{category} — {location}"
                )

                options[label] = item_id

            selected_label = st.selectbox(
                "Fundstück auswählen",
                list(options.keys())
            )

            if st.button(
                "📋 Fundstück öffnen",
                type="primary"
            ):

                st.session_state.selected_item_id = (
                    options[selected_label]
                )

                st.rerun()

    else:

        category = selected_item.get(
            "category",
            "📦 Sonstiges"
        )

        ai_label = selected_item.get(
            "ai_label",
            "Nicht erkannt"
        )

        confidence = float(
            selected_item.get(
                "ai_confidence",
                0.0
            ) or 0.0
        )

        location = selected_item.get(
            "location",
            "Unbekannt"
        )

        size = selected_item.get(
            "size",
            ""
        )

        color = selected_item.get(
            "color",
            ""
        )

        description = selected_item.get(
            "description",
            ""
        )

        date = format_date(
            selected_item.get(
                "date"
            )
        )

        left, right = st.columns(
            [1, 1]
        )

        with left:

            image = get_image_for_item(
                selected_item
            )

            show_image(
                image,
                caption="Fundstück"
            )

        with right:

            st.markdown(
                f'<span class="badge">{category}</span>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"## {category}"
            )

            st.markdown(
                f"""
                <div class="item-card">

                    <strong>🤖 KI-Erkennung</strong><br>
                    {ai_label}

                    <br><br>

                    <strong>🎯 Sicherheit</strong><br>
                    {confidence * 100:.1f}%

                    <br><br>

                    <strong>📍 Fundort</strong><br>
                    {location}

                    <br><br>

                    <strong>📏 Größe</strong><br>
                    {size if size else "Keine Angabe"}

                    <br><br>

                    <strong>🎨 Farbe</strong><br>
                    {color if color else "Keine Angabe"}

                    <br><br>

                    <strong>🕒 Eingetragen</strong><br>
                    {date}

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "### 📝 Beschreibung"
        )

        if description:
            st.write(description)
        else:
            st.write(
                "Keine Beschreibung vorhanden."
            )

        st.divider()

        back_col, delete_col = st.columns(2)

        with back_col:

            if st.button(
                "⬅️ Zur Übersicht",
                use_container_width=True
            ):

                st.session_state.page = (
                    "Übersicht"
                )

                st.rerun()

        with delete_col:

            if st.button(
                "🗑️ Fundstück löschen",
                use_container_width=True
            ):

                st.session_state.confirm_delete = True

        if st.session_state.confirm_delete:

            st.warning(
                "⚠️ Möchtest du dieses Fundstück wirklich löschen?"
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:

                if st.button(
                    "Ja, endgültig löschen",
                    type="primary",
                    use_container_width=True
                ):

                    delete_id = selected_item.get(
                        "id"
                    )

                    items = [
                        item
                        for item in items
                        if item.get("id")
                        != delete_id
                    ]

                    save_items(items)

                    st.session_state.selected_item_id = None
                    st.session_state.confirm_delete = False
                    st.session_state.page = "Übersicht"

                    st.rerun()

            with cancel_col:

                if st.button(
                    "Abbrechen",
                    use_container_width=True
                ):

                    st.session_state.confirm_delete = False

                    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🔎 Digitales Fundbüro
        • KI-gestützte Fundstück-Erkennung
    </div>
    """,
    unsafe_allow_html=True
)
