import base64
import io
import json
import os
import uuid
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# TensorFlow darf fehlen, ohne dass die komplette App abstürzt.
try:
    import tensorflow as tf
except Exception:
    tf = None


# ============================================================
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
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
    }

    .block-container {
        max-width: 1180px;
        padding-top: 28px;
        padding-bottom: 70px;
    }

    .brand {
        font-size: 15px;
        font-weight: 800;
        letter-spacing: -0.2px;
        margin-bottom: 8px;
    }

    .brand-sub {
        color: #8b8b8b;
        font-size: 12px;
        margin-bottom: 34px;
    }

    .hero-box {
        background: #ffffff;
        border: 1px solid #e7e7e4;
        border-radius: 28px;
        padding: 46px 48px;
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: #7d7d79;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #171717;
        font-size: clamp(48px, 7vw, 78px);
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: -4px;
        margin-bottom: 20px;
    }

    .hero-text {
        max-width: 560px;
        color: #707070;
        font-size: 15px;
        line-height: 1.65;
    }

    .section-kicker {
        color: #999;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 34px;
        margin-bottom: 5px;
    }

    .section-title {
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin-bottom: 18px;
    }

    .empty-box {
        background: #fff;
        border: 1px dashed #d3d3cf;
        border-radius: 22px;
        padding: 42px 28px;
        text-align: center;
        color: #666;
    }

    .empty-title {
        color: #222;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 1px dashed #cfcfcb;
        background: #fff;
    }

    .stButton > button {
        border-radius: 12px !important;
        min-height: 42px;
        font-weight: 700 !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 16px;
            padding-right: 16px;
        }

        .hero-box {
            padding: 30px 24px;
            border-radius: 22px;
        }

        .hero-title {
            font-size: 49px;
            letter-spacing: -2.7px;
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
    if not os.path.isfile(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            raw = json.load(file)

        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict) and isinstance(raw.get("fundstuecke"), list):
            data = raw["fundstuecke"]
        else:
            return []

        return [item for item in data if isinstance(item, dict)]

    except Exception:
        return []


def save_items(data):
    os.makedirs(DATA_DIR, exist_ok=True)

    temp_file = DATA_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temp_file, DATA_FILE)


def sort_items(data):
    def sort_key(item):
        return str(
            item.get("created_at")
            or item.get("id")
            or ""
        )

    return sorted(data, key=sort_key, reverse=True)


items = load_items()


# ============================================================
# BILDER
# ============================================================

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.copy()
    image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    return image


def image_to_base64(image):
    image = prepare_image(image)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=84,
        optimize=True,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def base64_to_image(value):
    try:
        raw = base64.b64decode(value, validate=True)

        return Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except Exception:
        return None


def get_item_image(item):
    image_data = item.get("image_data")

    if image_data:
        image = base64_to_image(image_data)

        if image is not None:
            return image

    old_path = item.get("image_path")

    if old_path and os.path.isfile(old_path):
        try:
            return ImageOps.exif_transpose(
                Image.open(old_path)
            ).convert("RGB")

        except Exception:
            return None

    return None


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():
    if tf is None:
        return None

    if not os.path.isfile(MODEL_FILE):
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
    if not os.path.isfile(LABELS_FILE):
        return []

    labels = []

    try:
        with open(
            LABELS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            for raw_line in file:
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split(
                    maxsplit=1
                )

                if (
                    len(parts) == 2
                    and parts[0].isdigit()
                ):
                    labels.append(
                        parts[1].strip()
                    )
                else:
                    labels.append(line)

    except Exception:
        return []

    return labels


def classify_image(image):
    model = load_model()
    labels = load_labels()

    if model is None or not labels:
        return "Unbekannt", 0.0

    try:
        input_shape = model.input_shape

        if isinstance(
            input_shape,
            list,
        ):
            input_shape = input_shape[0]

        height = int(
            input_shape[1] or 224
        )

        width = int(
            input_shape[2] or 224
        )

        channels = int(
            input_shape[3] or 3
        )

        image = (
            image
            .convert("RGB")
            .resize(
                (width, height)
            )
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if channels == 1:
            array = np.mean(
                array,
                axis=2,
                keepdims=True,
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

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        ).reshape(-1)

        if prediction.size == 0:
            return "Unbekannt", 0.0

        index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[index]
        )

        if index >= len(labels):
            return (
                "Unbekannt",
                confidence,
            )

        return (
            labels[index],
            confidence,
        )

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
    '<div class="brand">🔎 Fundbüro</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-sub">Digital</div>',
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
        "Suche",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)

st.session_state.page = page


# ============================================================
# DETAILANSICHT
# ============================================================

selected_id = st.session_state.selected_id

if selected_id is not None:

    selected = next(
        (
            item
            for item in items
            if str(item.get("id"))
            == str(selected_id)
        ),
        None,
    )

    if selected is not None:

        st.markdown(
            '<div class="section-kicker">Fundstück</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Details</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(
            [1.1, 0.9],
            gap="large",
        )

        with left:

            image = get_item_image(
                selected
            )

            if image is not None:
                st.image(image)
            else:
                st.info(
                    "Kein Bild gespeichert."
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
                f"### {name}"
            )

            st.caption(
                f"Kategorie: {category}"
            )

            st.caption(
                f"Fundort: {location}"
            )

            st.caption(
                f"Datum: {date}"
            )

            st.markdown(
                "**Beschreibung**"
            )

            st.write(
                description
            )

            st.write("")

            back_col, delete_col = st.columns(2)

            with back_col:

                if st.button(
                    "← Zur Übersicht",
                    use_container_width=True,
                    key="detail_back",
                ):
                    st.session_state.selected_id = None
                    st.rerun()

            with delete_col:

                if st.button(
                    "Löschen",
                    use_container_width=True,
                    key="detail_delete",
                ):
                    items = [
                        item
                        for item in items
                        if str(item.get("id"))
                        != str(
                            selected.get("id")
                        )
                    ]

                    save_items(items)

                    st.session_state.selected_id = None

                    st.rerun()

        st.stop()

    st.session_state.selected_id = None


# ============================================================
# ÜBERSICHT
# ============================================================

if page == "Übersicht":

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-kicker">
                Digitales Fundbüro
            </div>

            <div class="hero-title">
                Gefunden.<br>
                Gespeichert.
            </div>

            <div class="hero-text">
                Fundstücke fotografieren,
                automatisch erkennen und später
                schnell wiederfinden.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = {
        str(item.get("category"))
        for item in items
        if item.get("category")
    }

    locations = {
        str(item.get("location"))
        for item in items
        if item.get("location")
    }

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fundstücke",
        len(items),
    )

    c2.metric(
        "Kategorien",
        len(categories),
    )

    c3.metric(
        "Fundorte",
        len(locations),
    )

    st.markdown(
        '<div class="section-kicker">Fundstücke</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Zuletzt</div>',
        unsafe_allow_html=True,
    )

    if not items:

        st.markdown(
            """
            <div class="empty-box">
                <div class="empty-title">
                    Noch keine Fundstücke
                </div>

                Füge dein erstes Fundstück hinzu.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        recent = sort_items(items)[:6]

        for start in range(
            0,
            len(recent),
            3,
        ):

            row = recent[
                start:start + 3
            ]

            cols = st.columns(
                len(row),
                gap="large",
            )

            for col, item in zip(
                cols,
                row,
            ):

                with col:

                    image = get_item_image(
                        item
                    )

                    if image is not None:
                        st.image(image)
                    else:
                        st.info("Kein Bild")

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
                        f"**{name}**"
                    )

                    st.caption(
                        f"{category} · 📍 {location}"
                    )

                    if st.button(
                        "Details →",
                        key=(
                            "overview_details_"
                            + str(
                                item.get("id")
                            )
                        ),
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
        '<div class="section-kicker">Neu</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Fundstück hinzufügen</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:

        st.markdown("### Foto")

        uploaded = st.file_uploader(
            "Bild hochladen",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            label_visibility="collapsed",
            key="upload_image",
        )

        camera = st.camera_input(
            "Oder Kamera verwenden",
            key="camera_image",
        )

        source = (
            camera
            if camera is not None
            else uploaded
        )

        current_image = None

        if source is not None:

            try:

                current_image = prepare_image(
                    Image.open(source)
                )

                st.image(
                    current_image
                )

            except Exception:

                current_image = None

                st.error(
                    "Das Bild konnte nicht gelesen werden."
                )

    with right:

        st.markdown(
            "### Informationen"
        )

        name = st.text_input(
            "Name",
            placeholder=(
                "z. B. Schwarzer Rucksack"
            ),
            key="new_name",
        )

        location = st.text_input(
            "Fundort",
            placeholder=(
                "z. B. Sporthalle"
            ),
            key="new_location",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder=(
                "Weitere Merkmale oder Hinweise"
            ),
            height=130,
            key="new_description",
        )

        if st.button(
            "Fundstück speichern",
            use_container_width=True,
            key="save_item",
        ):

            if current_image is None:

                st.warning(
                    "Bitte zuerst ein Bild auswählen."
                )

            elif not location.strip():

                st.warning(
                    "Bitte einen Fundort eintragen."
                )

            else:

                category, confidence = classify_image(
                    current_image
                )

                new_item = {
                    "id": uuid.uuid4().hex,

                    "created_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),

                    "name": (
                        name.strip()
                        or "Fundstück"
                    ),

                    "category": category,

                    "location": (
                        location.strip()
                    ),

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
                        4,
                    ),
                }

                items.append(
                    new_item
                )

                save_items(items)

                st.session_state.selected_id = None
                st.session_state.page = "Übersicht"
                st.session_state.main_navigation = "Übersicht"

                st.rerun()


# ============================================================
# SUCHE
# ============================================================

elif page == "Suche":

    st.markdown(
        '<div class="section-kicker">Finden</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Fundstücke suchen</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Suche",
        placeholder=(
            "Name, Kategorie, Fundort oder Beschreibung"
        ),
        label_visibility="collapsed",
        key="search_input",
    )

    needle = search.strip().casefold()

    if needle:

        results = []

        for item in items:

            searchable = " ".join(
                [
                    str(
                        item.get("name")
                        or ""
                    ),
                    str(
                        item.get("category")
                        or ""
                    ),
                    str(
                        item.get("location")
                        or ""
                    ),
                    str(
                        item.get("description")
                        or ""
                    ),
                ]
            ).casefold()

            if needle in searchable:
                results.append(
                    item
                )

    else:

        results = sort_items(
            items
        )

    if not results:

        st.markdown(
            """
            <div class="empty-box">

                <div class="empty-title">
                    Nichts gefunden
                </div>

                Versuche einen anderen Suchbegriff.

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
        )

        for start in range(
            0,
            len(results),
            3,
        ):

            row = results[
                start:start + 3
            ]

            cols = st.columns(
                len(row),
                gap="large",
            )

            for col, item in zip(
                cols,
                row,
            ):

                with col:

                    image = get_item_image(
                        item
                    )

                    if image is not None:
                        st.image(image)
                    else:
                        st.info("Kein Bild")

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
                        f"**{name}**"
                    )

                    st.caption(
                        f"{category} · 📍 {location}"
                    )

                    if st.button(
                        "Details →",
                        key=(
                            "search_details_"
                            + str(
                                item.get("id")
                            )
                        ),
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = item.get(
                            "id"
                        )

                        st.rerun()
