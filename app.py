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
# APP-KONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DATEIEN
# =========================================================

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

    /* ------------------------------
       GRUNDLAYOUT
    ------------------------------ */

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


    /* ------------------------------
       HERO
    ------------------------------ */

    .hero-box {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e1b4b 45%,
            #075985 100%
        );

        border-radius: 28px;
        padding: 2.5rem 2.6rem;
        margin-bottom: 1.6rem;

        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.20);

        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 0;
        padding: 0;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        color: #e0e7ff !important;
        font-size: 1.08rem;
        line-height: 1.5;
        margin-top: 0.75rem;
        margin-bottom: 0;
    }


    /* ------------------------------
       STATISTIK
    ------------------------------ */

    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.25rem;
        min-height: 115px;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.06);
    }

    .stat-number {
        color: #111827;
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.2;
    }

    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }


    /* ------------------------------
       FUNDSTÜCK-KARTEN
    ------------------------------ */

    .item-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 1.15rem;
        margin-bottom: 1rem;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .badge {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;

        background: #eef2ff;
        color: #3730a3;

        font-size: 0.82rem;
        font-weight: 800;
    }

    .muted {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.7;
    }


    /* ------------------------------
       INFO-BOXEN
    ------------------------------ */

    .info-box {
        padding: 1.1rem 1.25rem;
        border-radius: 17px;

        background: #eff6ff;
        border: 1px solid #bfdbfe;

        color: #1e3a8a;
    }

    .success-box {
        padding: 1.1rem 1.25rem;
        border-radius: 17px;

        background: #ecfdf5;
        border: 1px solid #a7f3d0;

        color: #065f46;
    }


    /* ------------------------------
       BUTTONS
    ------------------------------ */

    div.stButton > button {
        border-radius: 13px;
        font-weight: 750;
        min-height: 45px;
    }


    /* ------------------------------
       FOOTER
    ------------------------------ */

    .footer {
        text-align: center;
        color: #9ca3af;
        margin-top: 3rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATENBANK
# =========================================================

def load_items():
    """Lädt alle Fundstücke."""

    if not ITEMS_FILE.exists():
        return []

    try:
        with open(
            ITEMS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    """Speichert Fundstücke sicher."""

    DATA_DIR.mkdir(exist_ok=True)

    temporary_file = DATA_DIR / "fundstuecke.tmp"

    try:
        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                items,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_file.replace(ITEMS_FILE)

    except Exception as error:
        st.error(
            f"❌ Fehler beim Speichern: {error}"
        )


# =========================================================
# BILDER SPEICHERN
# =========================================================

def image_to_base64(image):
    """
    Speichert das Bild komprimiert als Base64.
    """

    try:
        image = image.convert("RGB")

        # Verhindert extrem große JSON-Dateien.
        image.thumbnail((1200, 1200))

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=82,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        )

        return encoded.decode("utf-8")

    except Exception:
        return None


def base64_to_image(data):
    """Lädt ein Base64-Bild."""

    if not data:
        return None

    try:
        raw_data = base64.b64decode(data)

        image = Image.open(
            BytesIO(raw_data)
        )

        return image.convert("RGB")

    except Exception:
        return None


def show_image(image, caption=None):
    """
    Sehr robuste Bildanzeige.

    PIL -> JPEG Bytes -> Streamlit
    """

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
            quality=90,
        )

        image_bytes = buffer.getvalue()

        if caption:
            st.image(
                image_bytes,
                caption=caption,
            )
        else:
            st.image(image_bytes)

    except Exception:
        st.info(
            "📷 Dieses Bild konnte nicht angezeigt werden."
        )


def get_image_for_item(item):
    """
    Holt das Bild eines Fundstücks.

    Neue Datensätze:
    image_data

    Alte Datensätze:
    image_path
    """

    image_data = item.get("image_data")

    if image_data:
        image = base64_to_image(image_data)

        if image is not None:
            return image

    # Unterstützung für alte Datensätze.
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
# KI-MODELL
# =========================================================

def load_labels():

    if not LABELS_PATH.exists():
        return []

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            labels = []

            for line in file:

                value = line.strip()

                if not value:
                    continue

                parts = value.split(
                    maxsplit=1
                )

                # Teachable Machine:
                # 0 Rucksack
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

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        return model

    except Exception:
        return None


def classify_image(image):
    """
    Teachable-Machine-Bilderkennung.
    """

    model = load_model()
    labels = load_labels()

    if model is None:
        return (
            "Modell nicht gefunden",
            0.0,
        )

    try:

        image = image.convert("RGB")

        # Teachable Machine Standardgröße.
        image = image.resize(
            (224, 224)
        )

        image_array = np.asarray(
            image
        ).astype(np.float32)

        # Normalisierung.
        image_array = (
            image_array / 127.5
        ) - 1.0

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        prediction = model.predict(
            image_array,
            verbose=0,
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
            confidence,
        )

    except Exception:
        return (
            "Erkennung fehlgeschlagen",
            0.0,
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
            "schulranzen",
        ]
    ):
        return "🎒 Tasche / Rucksack"

    if any(
        word in text
        for word in [
            "flasche",
            "trinkflasche",
            "wasserflasche",
        ]
    ):
        return "🥤 Flasche"

    if any(
        word in text
        for word in [
            "schlüssel",
            "key",
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
            "kleidung",
        ]
    ):
        return "👕 Kleidung"

    if any(
        word in text
        for word in [
            "schuh",
            "sneaker",
            "turnschuh",
        ]
    ):
        return "👟 Schuhe"

    if any(
        word in text
        for word in [
            "handy",
            "smartphone",
            "telefon",
        ]
    ):
        return "📱 Handy"

    if any(
        word in text
        for word in [
            "kopfhörer",
            "headset",
            "airpods",
        ]
    ):
        return "🎧 Kopfhörer"

    if any(
        word in text
        for word in [
            "brille",
            "sonnenbrille",
        ]
    ):
        return "👓 Brille"

    if any(
        word in text
        for word in [
            "buch",
            "heft",
            "ordner",
            "block",
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
    search,
):

    search = search.lower().strip()

    if not search:
        return True

    values = [
        item.get("category", ""),
        item.get("ai_label", ""),
        item.get("location", ""),
        item.get("size", ""),
        item.get("color", ""),
        item.get("description", ""),
    ]

    combined = " ".join(
        str(value)
        for value in values
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


# =========================================================
# DATEN LADEN
# =========================================================

items = load_items()


# =========================================================
# HERO
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
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:

    if st.button(
        "🏠 Übersicht",
        use_container_width=True,
    ):
        st.session_state.page = "Übersicht"
        st.rerun()


with nav2:

    if st.button(
        "➕ Neuer Fund",
        use_container_width=True,
    ):
        st.session_state.page = "Neuer Fund"
        st.rerun()


with nav3:

    if st.button(
        "🔍 Suchen",
        use_container_width=True,
    ):
        st.session_state.page = "Suchen"
        st.rerun()


with nav4:

    if st.button(
        "📋 Details",
        use_container_width=True,
    ):
        st.session_state.page = "Details"
        st.rerun()


st.divider()


# =========================================================
# ÜBERSICHT
# =========================================================

if st.session_state.page == "Übersicht":

    st.subheader("📊 Übersicht")

    total_items = len(items)

    categories = set()

    for item in items:

        category = item.get(
            "category"
        )

        if category:
            categories.add(category)

    model_active = (
        load_model() is not None
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {total_items}
                </div>

                <div class="stat-label">
                    Gespeicherte Fundstücke
                </div>
            </div>
            """,
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
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
            unsafe_allow_html=True,
        )

    else:

        st.subheader("🆕 Zuletzt hinzugefügt")

        recent_items = list(
            reversed(items[-8:])
        )

        for item in recent_items:

            item_id = item.get("id")

            category = item.get(
                "category",
                "📦 Sonstiges",
            )

            location = item.get(
                "location",
                "Unbekannt",
            )

            description = item.get(
                "description",
                "",
            )

            date = format_date(
                item.get("date")
            )

            st.markdown(
                '<div class="item-card">',
                unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
                )

                if st.button(
                    "📋 Details anzeigen",
                    key=f"overview_{item_id}",
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
                unsafe_allow_html=True,
            )


# =========================================================
# NEUER FUND
# =========================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("➕ Neues Fundstück")

    st.write(
        "Foto aufnehmen oder hochladen und anschließend "
        "von der KI erkennen lassen."
    )

    upload_tab, camera_tab = st.tabs(
        [
            "📁 Bild hochladen",
            "📷 Kamera",
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
                "webp",
            ],
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
                caption="Ausgewähltes Fundstück",
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
            use_container_width=True,
        ):

            with st.spinner(
                "🔎 KI analysiert das Bild ..."
            ):

                label, confidence = (
                    classify_image(
                        selected_image
                    )
                )

            st.session_state.ai_label = (
                label
            )

            st.session_state.ai_confidence = (
                confidence
            )

            st.success(
                "✅ Analyse abgeschlossen!"
            )

        ai_label = (
            st.session_state.ai_label
        )

        ai_confidence = (
            st.session_state.ai_confidence
        )

        if ai_label:

            category = get_category(
                ai_label
            )

            result1, result2 = st.columns(
                2
            )

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
            ),
        )

        size = st.text_input(
            "📏 Größe",
            placeholder=(
                "z. B. klein, mittel, groß, M"
            ),
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder=(
                "z. B. schwarz, blau, rot"
            ),
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder=(
                "Weitere Merkmale oder Besonderheiten ..."
            ),
            height=120,
        )

        st.write("")

        if st.button(
            "💾 Fundstück speichern",
            type="primary",
            use_container_width=True,
        ):

            if not location.strip():

                st.warning(
                    "⚠️ Bitte gib den Fundort an."
                )

            else:

                image_data = (
                    image_to_base64(
                        selected_image
                    )
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

                    final_category = (
                        get_category(
                            final_label
                        )
                    )

                    new_item = {
                        "id": str(
                            uuid.uuid4()
                        ),

                        "category": (
                            final_category
                        ),

                        "ai_label": (
                            final_label
                        ),

                        "ai_confidence": (
                            st.session_state.ai_confidence
                        ),

                        "location": (
                            location.strip()
                        ),

                        "size": (
                            size.strip()
                        ),

                        "color": (
                            color.strip()
                        ),

                        "description": (
                            description.strip()
                        ),

                        "image_data": (
                            image_data
                        ),

                        "date": (
                            datetime.now().isoformat()
                        ),
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

                    st.session_state.page = (
                        "Details"
                    )

                    st.success(
                        "🎉 Fundstück gespeichert!"
                    )

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
        ),
    )

    all_categories = sorted(
        set(
            item.get(
                "category",
                "📦 Sonstiges",
            )
            for item in items
        )
    )

    category_filter = st.selectbox(
        "📦 Kategorie",
        [
            "Alle Kategorien",
            *all_categories,
        ],
    )

    results = []

    for item in items:

        if not item_matches_search(
            item,
            search,
        ):
            continue

        if (
            category_filter
            != "Alle Kategorien"
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

        for item in reversed(
            results
        ):

            item_id = item.get("id")

            category = item.get(
                "category",
                "📦 Sonstiges",
            )

            location = item.get(
                "location",
                "Unbekannt",
            )

            color = item.get(
                "color",
                "",
            )

            size = item.get(
                "size",
                "",
            )

            description = item.get(
                "description",
                "",
            )

            st.markdown(
                '<div class="item-card">',
                unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
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
                    key=f"search_{item_id}",
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
                unsafe_allow_html=True,
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

                item_id = item.get(
                    "id"
                )

                category = item.get(
                    "category",
                    "📦 Sonstiges",
                )

                location = item.get(
                    "location",
                    "Unbekannt",
                )

                label = (
                    f"{category} — {location}"
                )

                options[label] = item_id

            selected_label = st.selectbox(
                "Fundstück auswählen",
                list(options.keys()),
            )

            if st.button(
                "📋 Fundstück öffnen",
                type="primary",
            ):

                st.session_state.selected_item_id = (
                    options[selected_label]
                )

                st.rerun()

    else:

        category = selected_item.get(
            "category",
            "📦 Sonstiges",
        )

        ai_label = selected_item.get(
            "ai_label",
            "Nicht erkannt",
        )

        confidence = float(
            selected_item.get(
                "ai_confidence",
                0.0,
            )
            or 0.0
        )

        location = selected_item.get(
            "location",
            "Unbekannt",
        )

        size = selected_item.get(
            "size",
            "",
        )

        color = selected_item.get(
            "color",
            "",
        )

        description = selected_item.get(
            "description",
            "",
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
                caption="Fundstück",
            )

        with right:

            st.markdown(
                f'<span class="badge">{category}</span>',
                unsafe_allow_html=True,
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
                unsafe_allow_html=True,
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

        back_col, delete_col = st.columns(
            2
        )

        with back_col:

            if st.button(
                "⬅️ Zur Übersicht",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "Übersicht"
                )

                st.rerun()

        with delete_col:

            if st.button(
                "🗑️ Fundstück löschen",
                use_container_width=True,
            ):

                st.session_state.confirm_delete = (
                    True
                )

        if st.session_state.confirm_delete:

            st.warning(
                "⚠️ Möchtest du dieses Fundstück wirklich löschen?"
            )

            confirm_col, cancel_col = st.columns(
                2
            )

            with confirm_col:

                if st.button(
                    "Ja, endgültig löschen",
                    type="primary",
                    use_container_width=True,
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

                    st.session_state.selected_item_id = (
                        None
                    )

                    st.session_state.confirm_delete = (
                        False
                    )

                    st.session_state.page = (
                        "Übersicht"
                    )

                    st.rerun()

            with cancel_col:

                if st.button(
                    "Abbrechen",
                    use_container_width=True,
                ):

                    st.session_state.confirm_delete = (
                        False
                    )

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
    unsafe_allow_html=True,
)
