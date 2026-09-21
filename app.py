import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import uuid


# ============================================================
# EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

DATA_DIR = Path("fundburo_data")
ITEMS_FILE = DATA_DIR / "fundstuecke.json"
IMAGES_DIR = DATA_DIR / "bilder"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>

    /* ---------- Allgemein ---------- */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */

    .top-header {
        background: linear-gradient(135deg, #1976d2, #0d63c7);
        padding: 25px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.18);
    }

    .top-header h1 {
        margin: 0;
        font-size: 31px;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .top-header p {
        margin: 6px 0 0 0;
        opacity: 0.92;
        font-size: 15px;
    }

    /* ---------- Navigation ---------- */

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #dce3ed;
        background: white;
        color: #263238;
        font-weight: 600;
        min-height: 43px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        border-color: #1976d2;
        color: #1976d2;
        background: #f4f9ff;
    }

    /* ---------- Hauptbutton ---------- */

    .primary-info {
        background: linear-gradient(135deg, #eaf4ff, #f4f9ff);
        border: 1px solid #c9e2ff;
        border-radius: 18px;
        padding: 26px;
        margin: 15px 0 25px 0;
    }

    .primary-info h2 {
        margin-top: 0;
        color: #145da0;
    }

    .primary-info p {
        color: #566573;
        margin-bottom: 0;
    }

    /* ---------- Karten ---------- */

    .item-card {
        background: white;
        border: 1px solid #e3e8ef;
        border-radius: 17px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(30, 50, 70, 0.05);
    }

    .item-title {
        font-size: 21px;
        font-weight: 700;
        color: #17202a;
        margin-bottom: 7px;
    }

    .item-meta {
        color: #66727e;
        font-size: 14px;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        background: #eaf4ff;
        color: #1264b5;
        border-radius: 20px;
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* ---------- KI Box ---------- */

    .ai-result {
        background: linear-gradient(135deg, #edf7ff, #f7fbff);
        border: 1px solid #bcdfff;
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
    }

    .ai-result h3 {
        color: #1264b5;
        margin-top: 0;
    }

    /* ---------- Leere Übersicht ---------- */

    .empty-box {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 18px;
        padding: 55px 25px;
        text-align: center;
        margin-top: 15px;
    }

    .empty-box h2 {
        color: #37474f;
        margin-bottom: 8px;
    }

    .empty-box p {
        color: #7a8793;
    }

    /* ---------- Detailbereich ---------- */

    .detail-box {
        background: white;
        border: 1px solid #e3e8ef;
        border-radius: 17px;
        padding: 22px;
        box-shadow: 0 4px 15px rgba(30, 50, 70, 0.05);
    }

    .detail-label {
        color: #7a8793;
        font-size: 13px;
        margin-bottom: 2px;
    }

    .detail-value {
        color: #202b33;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 14px;
    }

    /* ---------- Inputs ---------- */

    input, textarea {
        border-radius: 10px !important;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #8a96a3;
        font-size: 13px;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATENBANK / DATEIEN
# ============================================================

def load_items():

    if not ITEMS_FILE.exists():
        return []

    try:
        with open(ITEMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_items(items):

    DATA_DIR.mkdir(exist_ok=True)

    temp_file = DATA_DIR / "fundstuecke_temp.json"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            items,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(ITEMS_FILE)


def save_image(image):

    filename = f"{uuid.uuid4()}.jpg"

    path = IMAGES_DIR / filename

    image.convert("RGB").save(
        path,
        format="JPEG",
        quality=90
    )

    return str(path)


# ============================================================
# KI
# ============================================================

def load_labels():

    labels = {}

    if not LABELS_PATH.exists():
        return labels

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if len(parts) == 2:

                try:
                    index = int(parts[0])
                    labels[index] = parts[1]

                except ValueError:
                    pass

    return labels


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "❌ keras_model.h5 wurde nicht gefunden."
        )

        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def classify_image(image):

    model = load_model()
    labels = load_labels()

    input_shape = model.input_shape

    width = input_shape[2]
    height = input_shape[1]

    image = image.convert("RGB")
    image = image.resize(
        (width, height)
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array = (
        image_array / 127.5
    ) - 1

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[0][index]
    )

    label = labels.get(
        index,
        f"Unbekannt ({index})"
    )

    return label, confidence


def get_category(label):

    label = label.lower()

    if (
        "pullover" in label
        or "t-shirt" in label
        or "shirt" in label
    ):
        return "Oberteile"

    if (
        "hose" in label
        or "sporthose" in label
    ):
        return "Hosen"

    if "schuh" in label:
        return "Schuhe"

    return "Sonstiges"


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected" not in st.session_state:
    st.session_state.selected = None

if "category" not in st.session_state:
    st.session_state.category = "Alle"

if "ai_label" not in st.session_state:
    st.session_state.ai_label = None

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="top-header">

    <h1>🔎 Digitales Fundbüro</h1>

    <p>
        Gefundene Gegenstände schnell erkennen,
        speichern und wiederfinden.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns([1, 1, 1])

with nav1:

    if st.button(
        "🏠 Übersicht",
        use_container_width=True
    ):

        st.session_state.page = "Übersicht"
        st.rerun()


with nav2:

    if st.button(
        "🔎 Suchen",
        use_container_width=True
    ):

        st.session_state.page = "Suchen"
        st.rerun()


with nav3:

    if st.button(
        "➕ Neuer Fund",
        use_container_width=True
    ):

        st.session_state.page = "Neuer Fund"

        st.session_state.ai_label = None
        st.session_state.ai_confidence = None

        st.rerun()


st.write("")


# ============================================================
# ÜBERSICHT
# ============================================================

if st.session_state.page == "Übersicht":

    items = load_items()

    st.markdown("""
    <div class="primary-info">

        <h2>Neuen Fund entdeckt?</h2>

        <p>
            Fotografiere den Gegenstand und unsere KI erkennt
            automatisch, was gefunden wurde.
        </p>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "➕ Neuen Fund eintragen",
        use_container_width=True
    ):

        st.session_state.page = "Neuer Fund"
        st.session_state.ai_label = None
        st.session_state.ai_confidence = None

        st.rerun()

    st.write("")

    st.subheader("Gefundene Gegenstände")

    # --------------------------------------------------------
    # KATEGORIEN
    # --------------------------------------------------------

    categories = [
        "Alle",
        "Oberteile",
        "Hosen",
        "Schuhe",
        "Sonstiges"
    ]

    cols = st.columns(len(categories))

    for i, category in enumerate(categories):

        with cols[i]:

            if st.button(
                category,
                use_container_width=True,
                key=f"cat_{category}"
            ):

                st.session_state.category = category
                st.rerun()

    # --------------------------------------------------------
    # SUCHFELD
    # --------------------------------------------------------

    search = st.text_input(
        "🔎",
        placeholder="Fundstück suchen ...",
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # FILTERN
    # --------------------------------------------------------

    filtered_items = items.copy()

    if st.session_state.category != "Alle":

        filtered_items = [
            item
            for item in filtered_items
            if item.get("category")
            == st.session_state.category
        ]

    if search:

        search_lower = search.lower()

        filtered_items = [
            item
            for item in filtered_items

            if (
                search_lower
                in item.get(
                    "name",
                    ""
                ).lower()
            )

            or (
                search_lower
                in item.get(
                    "location",
                    ""
                ).lower()
            )

            or (
                search_lower
                in item.get(
                    "category",
                    ""
                ).lower()
            )
        ]

    st.write("")

    # --------------------------------------------------------
    # KEINE FUNDSTÜCKE
    # --------------------------------------------------------

    if not filtered_items:

        st.markdown("""
        <div class="empty-box">

            <h2>📦 Noch keine Fundstücke</h2>

            <p>
                Hier werden gefundene Gegenstände angezeigt.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # FUNDSTÜCKE
    # --------------------------------------------------------

    else:

        st.write(
            f"**{len(filtered_items)} Fundstück"
            + ("e" if len(filtered_items) != 1 else "")
            + " gefunden**"
        )

        for index, item in enumerate(
            filtered_items
        ):

            col_img, col_info, col_button = st.columns(
                [1.2, 3.2, 1]
            )

            with col_img:

                image_path = item.get(
                    "image"
                )

                if (
                    image_path
                    and Path(image_path).exists()
                ):

                    st.image(
                        image_path,
                        use_column_width=True
                    )

                else:

                    st.markdown(
                        "<div style='font-size:60px;text-align:center;'>📦</div>",
                        unsafe_allow_html=True
                    )

            with col_info:

                st.markdown(
                    f"""
                    <div class="item-card">

                        <div class="badge">
                            {item.get("category", "Sonstiges")}
                        </div>

                        <div class="item-title">
                            {item.get("name", "Unbekannt")}
                        </div>

                        <div class="item-meta">
                            📍 {item.get("location", "-")}<br>
                            📅 {item.get("date", "-")}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_button:

                st.write("")

                if st.button(
                    "Details →",
                    key=f"details_{item.get('id', index)}",
                    use_container_width=True
                ):

                    st.session_state.selected = item.get(
                        "id"
                    )

                    st.session_state.page = "Details"

                    st.rerun()


# ============================================================
# NEUER FUND
# ============================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("➕ Neuen Fund eintragen")

    st.write(
        "Fotografiere den Gegenstand oder lade ein Bild hoch."
    )

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    camera_file = st.camera_input(
        "Oder direkt fotografieren"
    )

    image = None

    if camera_file:

        image = Image.open(
            camera_file
        )

    elif uploaded_file:

        image = Image.open(
            uploaded_file
        )

    # --------------------------------------------------------
    # BILD
    # --------------------------------------------------------

    if image is not None:

        st.image(
            image,
            caption="Fundstück",
            use_column_width=True
        )

        if st.button(
            "🤖 KI soll den Gegenstand erkennen",
            use_container_width=True
        ):

            with st.spinner(
                "KI analysiert das Bild ..."
            ):

                try:

                    label, confidence = classify_image(
                        image
                    )

                    st.session_state.ai_label = label

                    st.session_state.ai_confidence = confidence

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Die KI konnte das Bild nicht analysieren."
                    )

                    st.exception(e)

    # --------------------------------------------------------
    # KI ERGEBNIS
    # --------------------------------------------------------

    if st.session_state.ai_label:

        category = get_category(
            st.session_state.ai_label
        )

        confidence = (
            st.session_state.ai_confidence
            * 100
        )

        st.markdown(
            f"""
            <div class="ai-result">

                <h3>🤖 KI-Erkennung</h3>

                <p>
                    Die KI erkennt:
                    <strong>
                        {st.session_state.ai_label}
                    </strong>
                </p>

                <p>
                    Kategorie:
                    <strong>
                        {category}
                    </strong>
                </p>

                <p>
                    Erkennungsgenauigkeit:
                    <strong>
                        {confidence:.1f} %
                    </strong>
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Weitere Informationen")

        location = st.text_input(
            "📍 Fundort *",
            placeholder="z. B. Sporthalle"
        )

        size = st.text_input(
            "📏 Größe",
            placeholder="z. B. M"
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder="z. B. Schwarz"
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder="Weitere Merkmale ..."
        )

        st.info(
            f"Die Bezeichnung wird automatisch von der KI "
            f"übernommen: **{st.session_state.ai_label}**"
        )

        if st.button(
            "💾 Fundstück speichern",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib den Fundort ein."
                )

            else:

                image_path = save_image(
                    image
                )

                items = load_items()

                new_item = {

                    "id": str(
                        uuid.uuid4()
                    ),

                    "name":
                        st.session_state.ai_label,

                    "category":
                        category,

                    "location":
                        location.strip(),

                    "size":
                        size.strip(),

                    "color":
                        color.strip(),

                    "description":
                        description.strip(),

                    "date":
                        datetime.now().strftime(
                            "%d.%m.%Y"
                        ),

                    "image":
                        image_path
                }

                items.append(
                    new_item
                )

                save_items(
                    items
                )

                st.session_state.ai_label = None
                st.session_state.ai_confidence = None

                st.session_state.page = "Übersicht"

                st.success(
                    "✅ Fundstück wurde gespeichert!"
                )

                st.rerun()


# ============================================================
# SUCHE
# ============================================================

elif st.session_state.page == "Suchen":

    st.subheader("🔎 Fundstücke suchen")

    items = load_items()

    search = st.text_input(
        "Wonach suchst du?",
        placeholder="z. B. Pullover, Schuhe, Sporthalle ..."
    )

    if not items:

        st.markdown("""
        <div class="empty-box">

            <h2>📦 Noch keine Fundstücke</h2>

            <p>
                Sobald ein Fund eingetragen wurde,
                kannst du ihn hier suchen.
            </p>

        </div>
        """, unsafe_allow_html=True)

    elif search:

        search_lower = search.lower()

        results = [

            item

            for item in items

            if (
                search_lower
                in item.get(
                    "name",
                    ""
                ).lower()
            )

            or (
                search_lower
                in item.get(
                    "category",
                    ""
                ).lower()
            )

            or (
                search_lower
                in item.get(
                    "location",
                    ""
                ).lower()
            )

            or (
                search_lower
                in item.get(
                    "color",
                    ""
                ).lower()
            )
        ]

        if results:

            st.write(
                f"**{len(results)} Ergebnis(se)**"
            )

            for item in results:

                st.markdown(
                    f"""
                    <div class="item-card">

                        <div class="badge">
                            {item.get("category", "Sonstiges")}
                        </div>

                        <div class="item-title">
                            {item.get("name", "Unbekannt")}
                        </div>

                        <div class="item-meta">
                            📍 {item.get("location", "-")}<br>
                            📅 {item.get("date", "-")}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Kein passendes Fundstück gefunden."
            )

    else:

        st.info(
            "Gib oben einen Suchbegriff ein."
        )


# ============================================================
# DETAILS
# ============================================================

elif st.session_state.page == "Details":

    items = load_items()

    selected_id = st.session_state.selected

    item = None

    for current_item in items:

        if current_item.get(
            "id"
        ) == selected_id:

            item = current_item
            break

    if item is None:

        st.error(
            "Fundstück nicht gefunden."
        )

        if st.button(
            "← Zurück"
        ):

            st.session_state.page = "Übersicht"
            st.rerun()

    else:

        st.subheader(
            f"📦 {item.get('name', 'Fundstück')}"
        )

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            image_path = item.get(
                "image"
            )

            if (
                image_path
                and Path(image_path).exists()
            ):

                st.image(
                    image_path,
                    use_column_width=True
                )

        with col2:

            st.markdown(
                '<div class="detail-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Kategorie</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("category", "-")}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Fundort</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("location", "-")}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Datum</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("date", "-")}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Größe</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("size", "-") or "-"}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Farbe</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("color", "-") or "-"}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Beschreibung</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="detail-value">{item.get("description", "-") or "-"}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        if st.button(
            "📅 Abholtermin vereinbaren",
            use_container_width=True
        ):

            st.success(
                "Die Funktion zur Terminvereinbarung kann hier ergänzt werden."
            )

        st.write("")

        if st.button(
            "🗑️ Fundstück löschen",
            use_container_width=True
        ):

            items = [
                x
                for x in items
                if x.get("id")
                != item.get("id")
            ]

            save_items(
                items
            )

            st.session_state.selected = None
            st.session_state.page = "Übersicht"

            st.success(
                "Fundstück wurde gelöscht."
            )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Digitales Fundbüro · KI-gestützte Fundstück-Erkennung
</div>
""", unsafe_allow_html=True)
