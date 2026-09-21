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

    /* -----------------------------------------------------
       GRUNDLAYOUT
    ----------------------------------------------------- */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    /* -----------------------------------------------------
       NORMALE TEXTE
    ----------------------------------------------------- */

    p,
    label,
    .stMarkdown,
    .stText,
    .stCaption {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;
    }

    h1,
    h2,
    h3 {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;

        letter-spacing: -0.025em;
    }

    /* -----------------------------------------------------
       HAUPTTITEL
    ----------------------------------------------------- */

    .app-header {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 24px;
        padding: 30px 34px;
        margin-bottom: 26px;
        box-shadow:
            0 8px 30px rgba(15, 23, 42, 0.06);
    }

    .app-header-title {
        font-size: 36px;
        font-weight: 800;
        line-height: 1.15;
        color: #111827;
        margin: 0;
    }

    .app-header-subtitle {
        font-size: 16px;
        line-height: 1.6;
        color: #64748b;
        margin-top: 10px;
    }

    /* -----------------------------------------------------
       STATISTIK-KARTEN
    ----------------------------------------------------- */

    .stat-card {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 20px;
        padding: 22px;
        min-height: 118px;

        box-shadow:
            0 6px 22px rgba(15, 23, 42, 0.05);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);

        box-shadow:
            0 14px 32px rgba(15, 23, 42, 0.10);

        border-color: #cbd5e1;
    }

    .stat-number {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
    }

    .stat-label {
        color: #64748b;
        font-size: 14px;
        margin-top: 8px;
    }

    /* -----------------------------------------------------
       FUNDSTÜCK-KARTEN
    ----------------------------------------------------- */

    .item-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 18px;

        box-shadow:
            0 5px 18px rgba(15, 23, 42, 0.045);

        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease,
            border-color 0.22s ease,
            background-color 0.22s ease;
    }

    .item-card:hover {
        transform: translateY(-5px);

        border-color: #b8c3d4;

        background: #ffffff;

        box-shadow:
            0 18px 40px rgba(15, 23, 42, 0.12);
    }

    /* -----------------------------------------------------
       BADGES
    ----------------------------------------------------- */

    .badge {
        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: #eef2ff;
        color: #3730a3;

        font-size: 13px;
        font-weight: 700;

        margin-bottom: 8px;
    }

    /* -----------------------------------------------------
       DETAIL-BEREICH
    ----------------------------------------------------- */

    .detail-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 24px;

        box-shadow:
            0 7px 24px rgba(15, 23, 42, 0.06);
    }

    .detail-row {
        padding: 13px 0;
        border-bottom: 1px solid #eef0f4;
    }

    .detail-row:last-child {
        border-bottom: none;
    }

    .detail-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 3px;
    }

    .detail-value {
        color: #111827;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.5;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    /* -----------------------------------------------------
       BESCHREIBUNGEN
    ----------------------------------------------------- */

    .description-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;

        color: #334155;

        font-size: 15px;
        line-height: 1.65;

        overflow-wrap: anywhere;
        word-break: break-word;

        white-space: pre-wrap;
    }

    /* -----------------------------------------------------
       BILD-BEREICH
    ----------------------------------------------------- */

    .image-box {
        border-radius: 18px;
        overflow: hidden;
    }

    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 48px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
        font-size: 13px;
    }

    /* -----------------------------------------------------
       STREAMLIT BUTTONS
    ----------------------------------------------------- */

    .stButton > button {
        border-radius: 12px;
        min-height: 44px;

        font-weight: 650;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 7px 18px rgba(15, 23, 42, 0.10);
    }

    /* -----------------------------------------------------
       INPUTS
    ----------------------------------------------------- */

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px;
    }

    /* -----------------------------------------------------
       TABS
    ----------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    /* -----------------------------------------------------
       TRENNLINIE
    ----------------------------------------------------- */

    hr {
        border-color: #e5e7eb;
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
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
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
# BILDER
# =========================================================

def image_to_base64(image):
    try:
        image = image.convert("RGB")

        image.thumbnail(
            (1200, 1200)
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=82,
            optimize=True,
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
        raw_data = base64.b64decode(
            data
        )

        image = Image.open(
            BytesIO(raw_data)
        )

        return image.convert("RGB")

    except Exception:
        return None


def show_image(
    image,
    caption=None,
):
    if image is None:
        st.info(
            "📷 Kein Bild vorhanden."
        )
        return

    try:
        image = image.convert("RGB")

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

        if caption:
            st.image(
                buffer.getvalue(),
                caption=caption,
            )
        else:
            st.image(
                buffer.getvalue()
            )

    except Exception:
        st.info(
            "📷 Bild konnte nicht angezeigt werden."
        )


def get_image_for_item(item):
    image_data = item.get(
        "image_data"
    )

    if image_data:
        image = base64_to_image(
            image_data
        )

        if image is not None:
            return image

    old_path = item.get(
        "image_path"
    )

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
            compile=False,
        )

    except Exception:
        return None


def classify_image(image):
    model = load_model()
    labels = load_labels()

    if model is None:
        return (
            "Modell nicht gefunden",
            0.0,
        )

    try:
        image = image.convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = np.asarray(
            image
        ).astype(np.float32)

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
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[index]
        )

        if index < len(labels):
            label = labels[index]
        else:
            label = (
                f"Klasse {index + 1}"
            )

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
    search = (
        search.lower().strip()
    )

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

    text = " ".join(
        str(value)
        for value in values
    ).lower()

    return search in text


def safe_confidence(value):
    try:
        value = float(value or 0)

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    except Exception:
        return 0.0


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = (
        "Übersicht"
    )

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
    <div class="app-header">
        <div class="app-header-title">
            🔎 Digitales Fundbüro
        </div>

        <div class="app-header-subtitle">
            Fundstücke digital erfassen,
            automatisch erkennen und schnell wiederfinden.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

nav1, nav2, nav3, nav4 = st.columns(
    4
)

with nav1:

    if st.button(
        "🏠 Übersicht",
        use_container_width=True,
        key="nav_overview",
    ):

        st.session_state.page = (
            "Übersicht"
        )

        st.rerun()


with nav2:

    if st.button(
        "➕ Neuer Fund",
        use_container_width=True,
        key="nav_new",
    ):

        st.session_state.page = (
            "Neuer Fund"
        )

        st.rerun()


with nav3:

    if st.button(
        "🔍 Suchen",
        use_container_width=True,
        key="nav_search",
    ):

        st.session_state.page = (
            "Suchen"
        )

        st.rerun()


with nav4:

    if st.button(
        "📋 Details",
        use_container_width=True,
        key="nav_details",
    ):

        st.session_state.page = (
            "Details"
        )

        st.rerun()


st.divider()


# =========================================================
# ÜBERSICHT
# =========================================================

if st.session_state.page == "Übersicht":

    st.header(
        "📊 Übersicht"
    )

    st.write(
        "Hier siehst du die wichtigsten Informationen "
        "zu deinen gespeicherten Fundstücken."
    )

    total_items = len(items)

    categories = set()

    for item in items:

        category = item.get(
            "category"
        )

        if category:
            categories.add(
                category
            )

    model_active = (
        load_model() is not None
    )

    col1, col2, col3 = st.columns(
        3
    )

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
                    Verschiedene Kategorien
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

        st.info(
            "👋 Noch keine Fundstücke vorhanden. "
            "Erstelle dein erstes Fundstück über „➕ Neuer Fund“."
        )

    else:

        st.subheader(
            "🆕 Zuletzt hinzugefügt"
        )

        recent_items = list(
            reversed(
                items[-8:]
            )
        )

        for item in recent_items:

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

                st.subheader(
                    category
                )

                if description:

                    st.markdown(
                        '<div class="description-box">'
                        + description
                        + '</div>',
                        unsafe_allow_html=True,
                    )

                else:

                    st.caption(
                        "Keine Beschreibung angegeben."
                    )

                st.write(
                    f"📍 **Fundort:** {location}"
                )

                st.write(
                    f"🕒 **Eingetragen:** {date}"
                )

                if st.button(
                    "📋 Details anzeigen",
                    key=f"overview_{item_id}",
                    use_container_width=True,
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

    st.header(
        "➕ Neues Fundstück"
    )

    st.write(
        "Lade ein Bild hoch oder benutze die Kamera. "
        "Danach kann die KI versuchen, das Fundstück zu erkennen."
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

            st.subheader(
                "🖼️ Bildvorschau"
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

        st.subheader(
            "🤖 KI-Erkennung"
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

        st.subheader(
            "📝 Angaben zum Fundstück"
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
                "z. B. klein, mittel, groß"
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
            height=130,
        )

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

                image_data = image_to_base64(
                    selected_image
                )

                if image_data is None:

                    st.error(
                        "❌ Das Bild konnte nicht gespeichert werden."
                    )

                else:

                    final_label = (
                        st.session_state.ai_label
                        if st.session_state.ai_label
                        else "Nicht erkannt"
                    )

                    new_item = {
                        "id": str(
                            uuid.uuid4()
                        ),
                        "category": get_category(
                            final_label
                        ),
                        "ai_label": final_label,
                        "ai_confidence": (
                            st.session_state.ai_confidence
                        ),
                        "location": location.strip(),
                        "size": size.strip(),
                        "color": color.strip(),
                        "description": description.strip(),
                        "image_data": image_data,
                        "date": datetime.now().isoformat(),
                    }

                    items.append(
                        new_item
                    )

                    save_items(
                        items
                    )

                    st.session_state.selected_item_id = (
                        new_item["id"]
                    )

                    st.session_state.ai_label = ""
                    st.session_state.ai_confidence = 0.0

                    st.session_state.page = (
                        "Details"
                    )

                    st.success(
                        "🎉 Fundstück erfolgreich gespeichert!"
                    )

                    st.rerun()


# =========================================================
# SUCHE
# =========================================================

elif st.session_state.page == "Suchen":

    st.header(
        "🔍 Fundstücke suchen"
    )

    st.write(
        "Durchsuche alle gespeicherten Fundstücke nach "
        "Kategorie, Fundort, Farbe, Größe oder Beschreibung."
    )

    search = st.text_input(
        "Suchbegriff",
        placeholder=(
            "z. B. Rucksack, schwarz, Sporthalle ..."
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

        results.append(
            item
        )

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

                st.subheader(
                    category
                )

                if description:

                    st.markdown(
                        '<div class="description-box">'
                        + description
                        + '</div>',
                        unsafe_allow_html=True,
                    )

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
                        " • ".join(
                            details
                        )
                    )

                if st.button(
                    "📋 Details öffnen",
                    key=f"search_{item_id}",
                    use_container_width=True,
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

    st.header(
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

                options[
                    label
                ] = item_id

            selected_label = st.selectbox(
                "Fundstück auswählen",
                list(
                    options.keys()
                ),
            )

            if st.button(
                "📋 Fundstück öffnen",
                type="primary",
            ):

                st.session_state.selected_item_id = (
                    options[
                        selected_label
                    ]
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

        confidence = safe_confidence(
            selected_item.get(
                "ai_confidence",
                0.0,
            )
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

            st.subheader(
                category
            )

            st.markdown(
                f"""
                <div class="detail-card">

                    <div class="detail-row">
                        <div class="detail-label">
                            🤖 KI-Erkennung
                        </div>

                        <div class="detail-value">
                            {ai_label}
                        </div>
                    </div>

                    <div class="detail-row">
                        <div class="detail-label">
                            🎯 Erkennungs-Sicherheit
                        </div>

                        <div class="detail-value">
                            {confidence * 100:.1f} %
                        </div>
                    </div>

                    <div class="detail-row">
                        <div class="detail-label">
                            📍 Fundort
                        </div>

                        <div class="detail-value">
                            {location}
                        </div>
                    </div>

                    <div class="detail-row">
                        <div class="detail-label">
                            📏 Größe
                        </div>

                        <div class="detail-value">
                            {size if size else "Keine Angabe"}
                        </div>
                    </div>

                    <div class="detail-row">
                        <div class="detail-label">
                            🎨 Farbe
                        </div>

                        <div class="detail-value">
                            {color if color else "Keine Angabe"}
                        </div>
                    </div>

                    <div class="detail-row">
                        <div class="detail-label">
                            🕒 Eingetragen
                        </div>

                        <div class="detail-value">
                            {date}
                        </div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        st.subheader(
            "📝 Beschreibung"
        )

        if description:

            st.markdown(
                '<div class="description-box">'
                + description
                + '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "Keine Beschreibung vorhanden."
            )

        st.write("")

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

                st.session_state.confirm_delete = True

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

                    delete_id = (
                        selected_item.get(
                            "id"
                        )
                    )

                    items = [
                        item
                        for item in items
                        if item.get("id")
                        != delete_id
                    ]

                    save_items(
                        items
                    )

                    st.session_state.selected_item_id = None

                    st.session_state.confirm_delete = False

                    st.session_state.page = (
                        "Übersicht"
                    )

                    st.rerun()

            with cancel_col:

                if st.button(
                    "Abbrechen",
                    use_container_width=True,
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
        · KI-gestützte Fundstück-Erkennung
    </div>
    """,
    unsafe_allow_html=True,
)
