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


# ============================================================
# KONFIGURATION
# ============================================================

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


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79,70,229,0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
    }

    .hero p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    .stat-card {
        background: white;
        border-radius: 18px;
        padding: 1.2rem;
        border: 1px solid #e8eaf0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        min-height: 120px;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #4f46e5;
    }

    .stat-label {
        color: #6b7280;
        margin-top: 0.2rem;
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.8rem;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATEN LADEN
# ============================================================

def load_items():
    if not ITEMS_FILE.exists():
        return []

    try:
        with open(ITEMS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    try:
        with open(ITEMS_FILE, "w", encoding="utf-8") as file:
            json.dump(
                items,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return True

    except Exception as error:
        st.error(
            f"Die Fundstücke konnten nicht gespeichert werden: {error}"
        )
        return False


# ============================================================
# BILD SPEICHERN
# ============================================================

def image_to_base64(image):
    """
    Speichert das komplette Bild als Base64-Text.
    Dadurch bleibt das Bild direkt mit dem Fundstück verbunden.
    """

    try:
        buffer = BytesIO()

        image = image.convert("RGB")

        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except Exception:
        return None


def base64_to_image(data):
    """
    Wandelt gespeicherte Base64-Bilddaten
    wieder in ein PIL-Bild um.
    """

    try:
        if not data:
            return None

        image_bytes = base64.b64decode(data)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except Exception:
        return None


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():
        return []

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            labels = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

        cleaned_labels = []

        for label in labels:

            parts = label.split(" ", 1)

            if (
                len(parts) == 2
                and parts[0].isdigit()
            ):
                label = parts[1]

            cleaned_labels.append(label)

        return cleaned_labels

    except Exception:
        return []


# ============================================================
# KI-MODELL
# ============================================================

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

    if model is None:
        return None, 0.0

    try:

        input_shape = model.input_shape

        if isinstance(
            input_shape,
            list,
        ):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        image = image.convert("RGB")

        resized = image.resize(
            (width, height)
        )

        array = np.asarray(
            resized
        ).astype(np.float32)

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
        )

        if isinstance(
            prediction,
            list,
        ):
            prediction = prediction[0]

        prediction = np.asarray(
            prediction
        ).flatten()

        if len(prediction) == 0:
            return None, 0.0

        index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[index]
        )

        labels = load_labels()

        if index < len(labels):
            label = labels[index]
        else:
            label = f"Klasse {index + 1}"

        return label, confidence

    except Exception:
        return None, 0.0


# ============================================================
# KATEGORIE ERKENNEN
# ============================================================

def get_category(label):

    if not label:
        return "Sonstiges"

    text = label.lower()

    if any(
        word in text
        for word in [
            "shirt",
            "t-shirt",
            "tshirt",
            "pullover",
            "hoodie",
            "jacke",
            "mantel",
            "hemd",
            "bluse",
            "oberteil",
            "sweater",
            "sweatshirt",
        ]
    ):
        return "Oberteile"

    if any(
        word in text
        for word in [
            "hose",
            "jeans",
            "shorts",
            "rock",
            "leggings",
            "trouser",
        ]
    ):
        return "Hosen"

    if any(
        word in text
        for word in [
            "schuh",
            "schuhe",
            "sneaker",
            "stiefel",
            "sandale",
            "stiefelette",
        ]
    ):
        return "Schuhe"

    return "Sonstiges"


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def format_date(value):

    if not value:
        return "—"

    try:

        date = datetime.fromisoformat(
            value
        )

        return date.strftime(
            "%d.%m.%Y"
        )

    except Exception:
        return value


def item_matches_search(
    item,
    search,
):

    if not search:
        return True

    search = search.lower()

    fields = [
        item.get("category", ""),
        item.get("ai_label", ""),
        item.get("location", ""),
        item.get("size", ""),
        item.get("color", ""),
        item.get("description", ""),
        item.get("date", ""),
    ]

    text = " ".join(
        str(field)
        for field in fields
    ).lower()

    return search in text


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

if "category" not in st.session_state:
    st.session_state.category = "Alle"

if "ai_label" not in st.session_state:
    st.session_state.ai_label = ""

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = 0.0


# ============================================================
# FUNDSTÜCKE LADEN
# ============================================================

items = load_items()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔎 Digitales Fundbüro</h1>
        <p>
            Fundstücke einfach erfassen,
            mit KI erkennen und schnell wiederfinden.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns(3)

with nav1:

    if st.button(
        "🏠 Übersicht",
        use_container_width=True,
    ):

        st.session_state.page = "Übersicht"
        st.session_state.selected_id = None

        st.rerun()


with nav2:

    if st.button(
        "➕ Neuer Fund",
        use_container_width=True,
    ):

        st.session_state.page = "Neuer Fund"
        st.session_state.selected_id = None

        st.rerun()


with nav3:

    if st.button(
        "🔍 Suchen",
        use_container_width=True,
    ):

        st.session_state.page = "Suchen"
        st.session_state.selected_id = None

        st.rerun()


st.divider()


# ============================================================
# ÜBERSICHT
# ============================================================

if st.session_state.page == "Übersicht":

    st.subheader("📊 Übersicht")

    total = len(items)

    clothes = len(
        [
            item
            for item in items
            if item.get("category")
            in [
                "Oberteile",
                "Hosen",
            ]
        ]
    )

    shoes = len(
        [
            item
            for item in items
            if item.get("category")
            == "Schuhe"
        ]
    )

    other = len(
        [
            item
            for item in items
            if item.get("category")
            == "Sonstiges"
        ]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {total}
                </div>
                <div class="stat-label">
                    Fundstücke
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {clothes}
                </div>
                <div class="stat-label">
                    Kleidung
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {shoes}
                </div>
                <div class="stat-label">
                    Schuhe
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {other}
                </div>
                <div class="stat-label">
                    Sonstiges
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.subheader("📦 Fundstücke")

    categories = [
        "Alle",
        "Oberteile",
        "Hosen",
        "Schuhe",
        "Sonstiges",
    ]

    cols = st.columns(
        len(categories)
    )

    for index, category in enumerate(
        categories
    ):

        with cols[index]:

            if st.button(
                category,
                key=f"category_{category}",
                use_container_width=True,
            ):

                st.session_state.category = category
                st.rerun()

    selected_category = (
        st.session_state.category
    )

    if selected_category == "Alle":

        filtered_items = items

    else:

        filtered_items = [
            item
            for item in items
            if item.get("category")
            == selected_category
        ]

    if not filtered_items:

        st.info(
            "Noch keine passenden Fundstücke vorhanden."
        )

    else:

        for item in reversed(
            filtered_items
        ):

            col_image, col_info, col_button = st.columns(
                [1, 2, 0.7]
            )

            with col_image:

                saved_image = base64_to_image(
                    item.get("image_data")
                )

                if saved_image is not None:

                    try:

                        st.image(
                            saved_image,
                            use_container_width=True,
                        )

                    except Exception:

                        st.info(
                            "📷 Bild nicht verfügbar."
                        )

                else:

                    st.info(
                        "📷 Kein Bild"
                    )

            with col_info:

                st.markdown(
                    f"""
                    <span class="badge">
                        {item.get(
                            "category",
                            "Sonstiges"
                        )}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"### {item.get(
                        'ai_label',
                        'Unbekannt'
                    )}"
                )

                st.write(
                    f"📍 **Ort:** "
                    f"{item.get(
                        'location',
                        '—'
                    )}"
                )

                st.write(
                    f"📅 **Gefunden:** "
                    f"{format_date(
                        item.get('date')
                    )}"
                )

            with col_button:

                st.write("")

                if st.button(
                    "Details",
                    key=f"details_{item.get('id')}",
                    use_container_width=True,
                ):

                    st.session_state.selected_id = (
                        item.get("id")
                    )

                    st.session_state.page = (
                        "Details"
                    )

                    st.rerun()

            st.divider()


# ============================================================
# NEUER FUND
# ============================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("➕ Neues Fundstück")

    st.write(
        "Lade ein Foto hoch oder nutze die Kamera. "
        "Die KI versucht anschließend, den Gegenstand zu erkennen."
    )

    uploaded_file = st.file_uploader(
        "📷 Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    camera_file = st.camera_input(
        "Oder direkt mit der Kamera aufnehmen"
    )

    image_source = (
        camera_file
        if camera_file is not None
        else uploaded_file
    )

    image = None

    if image_source is not None:

        try:

            image = Image.open(
                image_source
            ).convert("RGB")

            st.image(
                image,
                caption="Ausgewähltes Bild",
                use_container_width=True,
            )

        except Exception:

            st.error(
                "Das Bild konnte nicht geöffnet werden."
            )

    if image is not None:

        if st.button(
            "🤖 Gegenstand mit KI erkennen",
            use_container_width=True,
        ):

            with st.spinner(
                "KI analysiert das Bild ..."
            ):

                label, confidence = classify_image(
                    image
                )

            st.session_state.ai_label = (
                label or ""
            )

            st.session_state.ai_confidence = (
                confidence
            )

            if label:

                st.success(
                    f"Erkannt: **{label}** "
                    f"({confidence * 100:.1f} %)"
                )

            else:

                st.warning(
                    "Der Gegenstand konnte nicht erkannt werden."
                )

    st.divider()

    st.subheader(
        "📝 Angaben zum Fundstück"
    )

    ai_label = (
        st.session_state.ai_label
    )

    default_category = get_category(
        ai_label
    )

    category_options = [
        "Oberteile",
        "Hosen",
        "Schuhe",
        "Sonstiges",
    ]

    if (
        default_category
        not in category_options
    ):
        default_category = "Sonstiges"

    category = st.selectbox(
        "Kategorie",
        category_options,
        index=category_options.index(
            default_category
        ),
    )

    label_input = st.text_input(
        "Bezeichnung",
        value=ai_label,
        placeholder="z. B. schwarzer Hoodie",
    )

    location = st.text_input(
        "📍 Fundort",
        placeholder="z. B. Sporthalle",
    )

    size = st.text_input(
        "📏 Größe",
        placeholder="z. B. M, 38, 42",
    )

    color = st.text_input(
        "🎨 Farbe",
        placeholder="z. B. Schwarz",
    )

    description = st.text_area(
        "📝 Beschreibung",
        placeholder="Weitere Merkmale oder Hinweise ...",
        height=120,
    )

    if st.button(
        "💾 Fundstück speichern",
        type="primary",
        use_container_width=True,
    ):

        if not label_input.strip():

            st.warning(
                "Bitte gib mindestens eine Bezeichnung ein."
            )

        elif not location.strip():

            st.warning(
                "Bitte gib den Fundort ein."
            )

        else:

            image_data = None

            if image is not None:

                image_data = image_to_base64(
                    image
                )

                if image_data is None:

                    st.warning(
                        "Das Fundstück wurde nicht gespeichert, "
                        "weil das Bild nicht verarbeitet werden konnte."
                    )

                    st.stop()

            new_item = {
                "id": str(uuid.uuid4()),
                "category": category,
                "ai_label": label_input.strip(),
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

            if save_items(items):

                st.success(
                    "✅ Fundstück erfolgreich gespeichert!"
                )

                st.session_state.ai_label = ""
                st.session_state.ai_confidence = 0.0
                st.session_state.page = "Übersicht"

                st.rerun()


# ============================================================
# SUCHEN
# ============================================================

elif st.session_state.page == "Suchen":

    st.subheader(
        "🔍 Fundstücke suchen"
    )

    search = st.text_input(
        "Suchbegriff",
        placeholder=(
            "z. B. Hoodie, Sporthalle, schwarz ..."
        ),
    )

    category_filter = st.selectbox(
        "Kategorie",
        [
            "Alle",
            "Oberteile",
            "Hosen",
            "Schuhe",
            "Sonstiges",
        ],
    )

    filtered = []

    for item in items:

        if not item_matches_search(
            item,
            search,
        ):
            continue

        if (
            category_filter != "Alle"
            and item.get("category")
            != category_filter
        ):
            continue

        filtered.append(item)

    st.write(
        f"**{len(filtered)} Fundstück(e) gefunden**"
    )

    if not filtered:

        st.info(
            "Keine passenden Fundstücke gefunden."
        )

    else:

        for item in reversed(
            filtered
        ):

            col1, col2, col3 = st.columns(
                [1, 2, 0.7]
            )

            with col1:

                saved_image = base64_to_image(
                    item.get("image_data")
                )

                if saved_image is not None:

                    try:

                        st.image(
                            saved_image,
                            use_container_width=True,
                        )

                    except Exception:

                        st.info(
                            "📷 Bild nicht verfügbar."
                        )

                else:

                    st.info(
                        "📷 Kein Bild"
                    )

            with col2:

                st.markdown(
                    f"""
                    <span class="badge">
                        {item.get(
                            "category",
                            "Sonstiges"
                        )}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"### {item.get(
                        'ai_label',
                        'Unbekannt'
                    )}"
                )

                st.write(
                    f"📍 {item.get(
                        'location',
                        '—'
                    )}"
                )

                if item.get("color"):

                    st.write(
                        f"🎨 {item.get('color')}"
                    )

                if item.get("size"):

                    st.write(
                        f"📏 {item.get('size')}"
                    )

            with col3:

                if st.button(
                    "Ansehen",
                    key=f"search_details_{item.get('id')}",
                    use_container_width=True,
                ):

                    st.session_state.selected_id = (
                        item.get("id")
                    )

                    st.session_state.page = (
                        "Details"
                    )

                    st.rerun()

            st.divider()


# ============================================================
# DETAILS
# ============================================================

elif st.session_state.page == "Details":

    selected_item = None

    for item in items:

        if (
            item.get("id")
            == st.session_state.selected_id
        ):

            selected_item = item
            break

    if selected_item is None:

        st.warning(
            "Das Fundstück wurde nicht gefunden."
        )

        if st.button(
            "← Zur Übersicht"
        ):

            st.session_state.page = (
                "Übersicht"
            )

            st.rerun()

    else:

        if st.button(
            "← Zurück"
        ):

            st.session_state.page = (
                "Übersicht"
            )

            st.session_state.selected_id = None

            st.rerun()

        st.subheader(
            f"🔎 {selected_item.get(
                'ai_label',
                'Fundstück'
            )}"
        )

        saved_image = base64_to_image(
            selected_item.get(
                "image_data"
            )
        )

        col1, col2 = st.columns(
            [1, 1.3]
        )

        with col1:

            if saved_image is not None:

                try:

                    st.image(
                        saved_image,
                        use_container_width=True,
                    )

                except Exception:

                    st.info(
                        "📷 Das Bild konnte nicht angezeigt werden."
                    )

            else:

                st.info(
                    "📷 Für dieses Fundstück ist kein Bild vorhanden."
                )

        with col2:

            st.markdown(
                f"""
                <span class="badge">
                    {selected_item.get(
                        "category",
                        "Sonstiges"
                    )}
                </span>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            st.write(
                f"**📍 Fundort:** "
                f"{selected_item.get(
                    'location',
                    '—'
                )}"
            )

            st.write(
                f"**📅 Datum:** "
                f"{format_date(
                    selected_item.get('date')
                )}"
            )

            st.write(
                f"**📏 Größe:** "
                f"{selected_item.get(
                    'size'
                ) or '—'}"
            )

            st.write(
                f"**🎨 Farbe:** "
                f"{selected_item.get(
                    'color'
                ) or '—'}"
            )

            confidence = selected_item.get(
                "ai_confidence",
                0,
            )

            try:

                confidence = float(
                    confidence
                )

            except Exception:

                confidence = 0.0

            if confidence > 0:

                st.write(
                    f"🤖 **KI-Sicherheit:** "
                    f"{confidence * 100:.1f} %"
                )

        st.divider()

        st.subheader(
            "📝 Beschreibung"
        )

        description = selected_item.get(
            "description",
            "",
        )

        if description:

            st.write(description)

        else:

            st.write(
                "Keine Beschreibung vorhanden."
            )

        st.divider()

        st.subheader(
            "⚙️ Verwaltung"
        )

        col_back, col_delete = st.columns(
            2
        )

        with col_back:

            if st.button(
                "← Zur Übersicht",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "Übersicht"
                )

                st.session_state.selected_id = None

                st.rerun()

        with col_delete:

            if st.button(
                "🗑️ Fundstück löschen",
                use_container_width=True,
            ):

                item_id = selected_item.get(
                    "id"
                )

                items = [
                    item
                    for item in items
                    if item.get("id")
                    != item_id
                ]

                if save_items(items):

                    st.success(
                        "Das Fundstück wurde gelöscht."
                    )

                    st.session_state.selected_id = None
                    st.session_state.page = "Übersicht"

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9ca3af;
        margin-top:3rem;
        padding-top:1rem;
        border-top:1px solid #e5e7eb;
    ">
        🔎 Digitales Fundbüro · KI-gestützte Fundstück-Erkennung
    </div>
    """,
    unsafe_allow_html=True,
)
