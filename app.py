
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import uuid


# ============================================================
# SEITENEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATEIEN
# ============================================================

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

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GRUNDLAYOUT
    ------------------------------------------------------- */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* -------------------------------------------------------
       STREAMLIT ELEMENTE
    ------------------------------------------------------- */

    div[data-testid="stVerticalBlock"] {
        gap: 0.7rem;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 600;
        border: 1px solid #dbe3ec;
        background: white;
        color: #263238;
    }

    div.stButton > button:hover {
        border-color: #1976d2;
        color: #1976d2;
        background: #f4f9ff;
    }

    /* -------------------------------------------------------
       HEADER
    ------------------------------------------------------- */

    .header-box {
        background: linear-gradient(
            135deg,
            #1976d2,
            #125bb0
        );
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow:
            0 10px 30px rgba(25, 118, 210, 0.18);
    }

    .header-title {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.7px;
    }

    .header-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 15px;
        margin-top: 6px;
    }

    /* -------------------------------------------------------
       NAVIGATION
    ------------------------------------------------------- */

    .nav-space {
        height: 4px;
    }

    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero-box {
        background: white;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        padding: 28px;
        margin: 20px 0;
        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.05);
    }

    .hero-title {
        color: #17202a;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
    }

    /* -------------------------------------------------------
       STATISTIK
    ------------------------------------------------------- */

    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 17px;
        padding: 20px;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.04);
    }

    .stat-number {
        font-size: 28px;
        font-weight: 800;
        color: #1976d2;
    }

    .stat-label {
        color: #64748b;
        font-size: 13px;
        margin-top: 3px;
    }

    /* -------------------------------------------------------
       FUNDSTÜCK-KARTEN
    ------------------------------------------------------- */

    .item-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.045);
    }

    .item-name {
        font-size: 20px;
        font-weight: 750;
        color: #17202a;
        margin-bottom: 7px;
    }

    .item-category {
        display: inline-block;
        background: #eaf4ff;
        color: #1264b5;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .item-info {
        color: #64748b;
        font-size: 14px;
        line-height: 1.8;
    }

    /* -------------------------------------------------------
       KI
    ------------------------------------------------------- */

    .ai-box {
        background: linear-gradient(
            135deg,
            #eff8ff,
            #f8fbff
        );
        border: 1px solid #b9ddff;
        border-radius: 18px;
        padding: 22px;
        margin: 18px 0;
    }

    .ai-title {
        color: #1264b5;
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .ai-detection {
        color: #17202a;
        font-size: 24px;
        font-weight: 800;
    }

    .ai-confidence {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }

    /* -------------------------------------------------------
       LEERER ZUSTAND
    ------------------------------------------------------- */

    .empty-box {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 20px;
        padding: 55px 25px;
        text-align: center;
        margin-top: 20px;
    }

    .empty-icon {
        font-size: 45px;
        margin-bottom: 10px;
    }

    .empty-title {
        color: #334155;
        font-size: 21px;
        font-weight: 750;
    }

    .empty-text {
        color: #94a3b8;
        margin-top: 5px;
    }

    /* -------------------------------------------------------
       DETAILS
    ------------------------------------------------------- */

    .detail-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.045);
    }

    .detail-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .detail-value {
        color: #263238;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 17px;
    }

    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
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
        with open(
            ITEMS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except Exception:
        return []

    return []


def save_items(items):
    DATA_DIR.mkdir(exist_ok=True)

    with open(
        ITEMS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            items,
            file,
            ensure_ascii=False,
            indent=2
        )


def save_image(image):
    filename = f"{uuid.uuid4()}.jpg"
    path = IMAGES_DIR / filename

    image.convert("RGB").save(
        path,
        "JPEG",
        quality=90
    )

    return str(path)


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():

    labels = {}

    if not LABELS_PATH.exists():
        return labels

    try:
        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(
                    " ",
                    1
                )

                if len(parts) == 2:

                    try:
                        index = int(parts[0])
                        labels[index] = parts[1].strip()

                    except ValueError:
                        pass

    except Exception:
        pass

    return labels


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "Das Modell keras_model.h5 wurde nicht gefunden."
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

    height = input_shape[1]
    width = input_shape[2]

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

    text = label.lower()

    if (
        "pullover" in text
        or "shirt" in text
        or "t-shirt" in text
        or "tshirt" in text
        or "jacke" in text
    ):
        return "Oberteile"

    if (
        "hose" in text
        or "jeans" in text
    ):
        return "Hosen"

    if (
        "schuh" in text
        or "sneaker" in text
    ):
        return "Schuhe"

    return "Sonstiges"


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
    st.session_state.ai_label = None

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header-box">
        <div class="header-title">
            🔎 Digitales Fundbüro
        </div>
        <div class="header-subtitle">
            Fundstücke einfach erfassen, erkennen und wiederfinden
        </div>
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
        "🏠  Übersicht",
        use_container_width=True
    ):
        st.session_state.page = "Übersicht"
        st.rerun()


with nav2:

    if st.button(
        "🔎  Suchen",
        use_container_width=True
    ):
        st.session_state.page = "Suchen"
        st.rerun()


with nav3:

    if st.button(
        "➕  Neuer Fund",
        use_container_width=True
    ):
        st.session_state.page = "Neuer Fund"

        st.session_state.ai_label = None
        st.session_state.ai_confidence = None

        st.rerun()


# ============================================================
# ÜBERSICHT
# ============================================================

if st.session_state.page == "Übersicht":

    items = load_items()

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">
                👋 Willkommen im Fundbüro
            </div>
            <div class="hero-text">
                Du hast etwas gefunden?
                Fotografiere es und lass die KI automatisch
                erkennen, um welchen Gegenstand es sich handelt.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # STATISTIK
    # --------------------------------------------------------

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {len(items)}
                </div>
                <div class="stat-label">
                    Gespeicherte Fundstücke
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stat2:

        categories_count = len(
            set(
                item.get(
                    "category",
                    "Sonstiges"
                )
                for item in items
            )
        )

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {categories_count}
                </div>
                <div class="stat-label">
                    Kategorien
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stat3:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">
                    🤖
                </div>
                <div class="stat-label">
                    KI-Erkennung aktiv
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if st.button(
        "➕  Neues Fundstück hinzufügen",
        use_container_width=True
    ):
        st.session_state.page = "Neuer Fund"
        st.session_state.ai_label = None
        st.session_state.ai_confidence = None
        st.rerun()

    st.write("")

    # --------------------------------------------------------
    # FUNDSTÜCKE
    # --------------------------------------------------------

    st.subheader("Aktuelle Fundstücke")

    categories = [
        "Alle",
        "Oberteile",
        "Hosen",
        "Schuhe",
        "Sonstiges",
    ]

    category_columns = st.columns(
        len(categories)
    )

    for index, category in enumerate(
        categories
    ):

        with category_columns[index]:

            if st.button(
                category,
                key=f"category_{index}",
                use_container_width=True
            ):

                st.session_state.category = category
                st.rerun()

    filtered_items = items.copy()

    if st.session_state.category != "Alle":

        filtered_items = [
            item
            for item in filtered_items
            if item.get("category")
            == st.session_state.category
        ]

    if not filtered_items:

        st.markdown(
            """
            <div class="empty-box">
                <div class="empty-icon">
                    📦
                </div>
                <div class="empty-title">
                    Noch keine Fundstücke
                </div>
                <div class="empty-text">
                    Hier erscheinen deine gespeicherten Fundstücke.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        for index, item in enumerate(
            filtered_items
        ):

            image_column, info_column, action_column = st.columns(
                [1.1, 3.2, 1]
            )

            with image_column:

                image_path = item.get(
                    "image"
                )

                if (
                    image_path
                    and Path(image_path).exists()
                ):

                    st.image(
                        image_path,
                        use_container_width=True
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                            background:#f1f5f9;
                            border-radius:15px;
                            padding:35px 10px;
                            text-align:center;
                            font-size:40px;
                        ">
                            📦
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with info_column:

                st.markdown(
                    f"""
                    <div class="item-card">

                        <div class="item-category">
                            {item.get("category", "Sonstiges")}
                        </div>

                        <div class="item-name">
                            {item.get("name", "Unbekannter Gegenstand")}
                        </div>

                        <div class="item-info">
                            📍 {item.get("location", "Kein Fundort")}<br>
                            📅 {item.get("date", "Kein Datum")}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with action_column:

                st.write("")

                if st.button(
                    "Details",
                    key=f"details_{index}_{item.get('id', '')}",
                    use_container_width=True
                ):

                    st.session_state.selected_id = item.get(
                        "id"
                    )

                    st.session_state.page = "Details"

                    st.rerun()


# ============================================================
# NEUER FUND
# ============================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("➕ Neuen Fund erfassen")

    st.write(
        "Lade ein Foto hoch oder benutze direkt die Kamera."
    )

    upload_column, camera_column = st.columns(2)

    with upload_column:

        uploaded_file = st.file_uploader(
            "📁 Bild auswählen",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )

    with camera_column:

        camera_file = st.camera_input(
            "📷 Kamera verwenden"
        )

    image = None

    if camera_file is not None:

        image = Image.open(
            camera_file
        )

    elif uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

    if image is not None:

        st.write("")

        preview_column, info_column = st.columns(
            [1.2, 1]
        )

        with preview_column:

            st.image(
                image,
                caption="Vorschau",
                use_container_width=True
            )

        with info_column:

            st.info(
                "Die KI versucht automatisch zu erkennen, "
                "welcher Gegenstand auf dem Bild zu sehen ist."
            )

            if st.button(
                "🤖  Gegenstand erkennen",
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

                    except Exception as error:

                        st.error(
                            "Die KI konnte das Bild nicht analysieren."
                        )

                        st.exception(error)

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
            <div class="ai-box">

                <div class="ai-title">
                    🤖 KI-Erkennung
                </div>

                <div class="ai-detection">
                    {st.session_state.ai_label}
                </div>

                <div class="ai-confidence">
                    Kategorie: {category}
                    · Erkennungsgenauigkeit: {confidence:.1f} %
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Weitere Angaben")

        location = st.text_input(
            "📍 Fundort",
            placeholder="z. B. Sporthalle, Schulhof ..."
        )

        size = st.text_input(
            "📏 Größe",
            placeholder="z. B. M, 42 ..."
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder="z. B. Schwarz ..."
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder="Weitere Merkmale oder Hinweise ..."
        )

        if st.button(
            "💾  Fundstück speichern",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib noch den Fundort ein."
                )

            else:

                image_path = save_image(
                    image
                )

                items = load_items()

                new_item = {
                    "id": str(uuid.uuid4()),
                    "name": st.session_state.ai_label,
                    "category": category,
                    "location": location.strip(),
                    "size": size.strip(),
                    "color": color.strip(),
                    "description": description.strip(),
                    "date": datetime.now().strftime(
                        "%d.%m.%Y"
                    ),
                    "image": image_path,
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
                    "Fundstück wurde erfolgreich gespeichert."
                )

                st.rerun()


# ============================================================
# SUCHE
# ============================================================

elif st.session_state.page == "Suchen":

    st.subheader("🔎 Fundstücke suchen")

    items = load_items()

    search = st.text_input(
        "Suche",
        placeholder="z. B. Pullover, Schuhe, Sporthalle ..."
    )

    if not items:

        st.markdown(
            """
            <div class="empty-box">
                <div class="empty-icon">📦</div>
                <div class="empty-title">
                    Noch keine Fundstücke
                </div>
                <div class="empty-text">
                    Es gibt noch nichts zu durchsuchen.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif search.strip():

        search_text = search.lower().strip()

        results = []

        for item in items:

            searchable = " ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("category", "")),
                    str(item.get("location", "")),
                    str(item.get("color", "")),
                    str(item.get("size", "")),
                    str(item.get("description", "")),
                ]
            ).lower()

            if search_text in searchable:
                results.append(item)

        if not results:

            st.info(
                "Kein passendes Fundstück gefunden."
            )

        else:

            st.write(
                f"{len(results)} Fundstück"
                + ("e" if len(results) != 1 else "")
                + " gefunden"
            )

            for index, item in enumerate(
                results
            ):

                with st.container():

                    left, middle, right = st.columns(
                        [1, 3, 1]
                    )

                    with left:

                        image_path = item.get(
                            "image"
                        )

                        if (
                            image_path
                            and Path(image_path).exists()
                        ):

                            st.image(
                                image_path,
                                use_container_width=True
                            )

                        else:

                            st.write("📦")

                    with middle:

                        st.markdown(
                            f"""
                            <div class="item-card">

                                <div class="item-category">
                                    {item.get("category", "Sonstiges")}
                                </div>

                                <div class="item-name">
                                    {item.get("name", "Unbekannt")}
                                </div>

                                <div class="item-info">
                                    📍 {item.get("location", "-")}<br>
                                    📅 {item.get("date", "-")}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with right:

                        if st.button(
                            "Details",
                            key=f"search_details_{index}",
                            use_container_width=True
                        ):

                            st.session_state.selected_id = item.get(
                                "id"
                            )

                            st.session_state.page = "Details"

                            st.rerun()

    else:

        st.info(
            "Gib oben einen Suchbegriff ein."
        )


# ============================================================
# DETAILS
# ============================================================

elif st.session_state.page == "Details":

    items = load_items()

    selected_item = None

    for item in items:

        if item.get("id") == st.session_state.selected_id:

            selected_item = item
            break

    if selected_item is None:

        st.error(
            "Das Fundstück wurde nicht gefunden."
        )

        if st.button(
            "← Zurück zur Übersicht"
        ):

            st.session_state.page = "Übersicht"
            st.rerun()

    else:

        if st.button(
            "← Zurück"
        ):

            st.session_state.page = "Übersicht"
            st.rerun()

        st.write("")

        st.title(
            selected_item.get(
                "name",
                "Fundstück"
            )
        )

        image_column, detail_column = st.columns(
            [1.15, 1]
        )

        with image_column:

            image_path = selected_item.get(
                "image"
            )

            if (
                image_path
                and Path(image_path).exists()
            ):

                st.image(
                    image_path,
                    use_container_width=True
                )

            else:

                st.markdown(
                    """
                    <div class="empty-box">
                        📦
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with detail_column:

            st.markdown(
                '<div class="detail-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Kategorie</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("category", "-")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Fundort</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("location", "-")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Datum</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("date", "-")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Größe</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("size") or "-"}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Farbe</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("color") or "-"}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="detail-label">Beschreibung</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="detail-value">
                    {selected_item.get("description") or "-"}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # ----------------------------------------------------
        # AKTIONEN
        # ----------------------------------------------------

        st.subheader("Aktionen")

        if st.button(
            "📅 Abholung vereinbaren",
            use_container_width=True
        ):

            st.info(
                "Die Abholfunktion kann hier später ergänzt werden."
            )

        if st.button(
            "🗑️ Fundstück löschen",
            use_container_width=True
        ):

            remaining_items = [
                item
                for item in items
                if item.get("id")
                != selected_item.get("id")
            ]

            save_items(
                remaining_items
            )

            st.session_state.selected_id = None
            st.session_state.page = "Übersicht"

            st.success(
                "Fundstück wurde gelöscht."
            )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Digitales Fundbüro · KI-gestützte Fundstück-Erkennung
    </div>
    """,
    unsafe_allow_html=True,
)
