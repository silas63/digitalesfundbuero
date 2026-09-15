import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import base64
import uuid


# ============================================================
# EINSTELLUNGEN
# ============================================================

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

DATA_DIR = Path("fundburo_data")
ITEMS_FILE = DATA_DIR / "fundstuecke.json"
IMAGES_DIR = DATA_DIR / "bilder"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


# ============================================================
# SEITEN-DESIGN
# ============================================================

st.set_page_config(
    page_title="Fundbüro | Kleidung",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .header {
        background-color: #1e73d8;
        padding: 22px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }

    .header h1 {
        margin: 0;
        font-size: 30px;
    }

    .card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e3e7ed;
        margin-bottom: 15px;
    }

    .empty {
        background: white;
        padding: 50px;
        text-align: center;
        border-radius: 15px;
        border: 1px solid #e3e7ed;
    }

    .ai-box {
        background: #eef6ff;
        border: 1px solid #b9dcff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATEN LADEN / SPEICHERN
# ============================================================

def load_items():
    """Lädt alle Fundstücke aus der gemeinsamen JSON-Datei."""

    if not ITEMS_FILE.exists():
        return []

    try:
        with open(ITEMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    """Speichert alle Fundstücke dauerhaft."""

    DATA_DIR.mkdir(exist_ok=True)

    temp_file = DATA_DIR / "fundstuecke_temp.json"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    temp_file.replace(ITEMS_FILE)


def save_image(image, filename):
    """Speichert ein Bild dauerhaft."""

    image_path = IMAGES_DIR / filename
    image.save(image_path, format="JPEG", quality=90)

    return str(image_path)


# ============================================================
# KI
# ============================================================

def load_labels():
    labels = {}

    if not LABELS_PATH.exists():
        return labels

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if len(parts) == 2:
                try:
                    index = int(parts[0])
                    label = parts[1]
                    labels[index] = label
                except ValueError:
                    pass

    return labels


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ keras_model.h5 wurde nicht gefunden.")
        st.info("Lade keras_model.h5 in denselben GitHub-Ordner wie app.py.")
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
    image = image.resize((width, height))

    image_array = np.asarray(image).astype(np.float32)

    # Teachable-Machine-Normalisierung
    image_array = (image_array / 127.5) - 1

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])

    label = labels.get(index, f"Unbekannt ({index})")

    return label, confidence


def get_category(label):
    label_lower = label.lower()

    if (
        "pullover" in label_lower
        or "t-shirt" in label_lower
        or "shirt" in label_lower
    ):
        return "Oberteile"

    if (
        "hose" in label_lower
        or "sporthose" in label_lower
    ):
        return "Hosen"

    if "schuh" in label_lower:
        return "Schuhe"

    if "federtasche" in label_lower:
        return "Sonstiges"

    return "Sonstiges"


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"

if "selected" not in st.session_state:
    st.session_state.selected = None

if "ai_label" not in st.session_state:
    st.session_state.ai_label = None

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">
    <h1>Fundbüro | Kleidung</h1>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button("Übersicht", use_container_width=True):
        st.session_state.page = "Übersicht"
        st.rerun()

with nav2:
    if st.button("Suchen", use_container_width=True):
        st.session_state.page = "Suchen"
        st.rerun()

with nav3:
    if st.button("Neuer Fund", use_container_width=True):
        st.session_state.page = "Neuer Fund"
        st.session_state.ai_label = None
        st.session_state.ai_confidence = None
        st.rerun()

with nav4:
    if st.button("Anmelden", use_container_width=True):
        st.session_state.page = "Anmelden"
        st.rerun()


st.divider()


# ============================================================
# ÜBERSICHT
# ============================================================

if st.session_state.page == "Übersicht":

    st.subheader("Gefundene Gegenstände")

    items = load_items()

    # Suche
    search = st.text_input(
        "🔎 Fundstück suchen",
        placeholder="z. B. Pullover, Schuhe, Sporthose ..."
    )

    if search:
        search_lower = search.lower()

        items = [
            item for item in items
            if search_lower in item.get("name", "").lower()
            or search_lower in item.get("location", "").lower()
            or search_lower in item.get("category", "").lower()
        ]

    # Kategorien
    cat1, cat2, cat3, cat4, cat5 = st.columns(5)

    selected_category = "Alle"

    with cat1:
        if st.button("Alle", use_container_width=True):
            st.session_state.category = "Alle"

    with cat2:
        if st.button("Oberteile", use_container_width=True):
            st.session_state.category = "Oberteile"

    with cat3:
        if st.button("Hosen", use_container_width=True):
            st.session_state.category = "Hosen"

    with cat4:
        if st.button("Schuhe", use_container_width=True):
            st.session_state.category = "Schuhe"

    with cat5:
        if st.button("Sonstiges", use_container_width=True):
            st.session_state.category = "Sonstiges"

    if "category" not in st.session_state:
        st.session_state.category = "Alle"

    selected_category = st.session_state.category

    if selected_category != "Alle":
        items = [
            item for item in items
            if item.get("category") == selected_category
        ]

    st.write("")

    # --------------------------------------------------------
    # KEINE FUNDSTÜCKE
    # --------------------------------------------------------

    if not items:

        st.markdown("""
        <div class="empty">
            <h2>📦 Noch keine Fundstücke</h2>
            <p>
                Es wurden bisher keine Gegenstände eingetragen.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button(
            "➕ Erstes Fundstück eintragen",
            use_container_width=True
        ):
            st.session_state.page = "Neuer Fund"
            st.rerun()

    # --------------------------------------------------------
    # FUNDSTÜCKE
    # --------------------------------------------------------

    else:

        for i, item in enumerate(items):

            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:

                image_file = item.get("image")

                if image_file and Path(image_file).exists():
                    st.image(
                        image_file,
                        caption="Fundstück",
                        use_column_width=True
                    )
                else:
                    st.write("📦")

            with col2:

                st.markdown(
                    f"### {item.get('name', 'Unbekannt')}"
                )

                st.write(
                    f"**Kategorie:** {item.get('category', 'Sonstiges')}"
                )

                st.write(
                    f"**Fundort:** {item.get('location', '-')}"
                )

                st.write(
                    f"**Datum:** {item.get('date', '-')}"
                )

            with col3:

                if st.button(
                    "Details",
                    key=f"details_{item.get('id', i)}",
                    use_container_width=True
                ):
                    st.session_state.selected = item.get(
                        "id",
                        i
                    )
                    st.session_state.page = "Details"
                    st.rerun()

            st.divider()


# ============================================================
# NEUER FUND
# ============================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("Neuen Fund eintragen")

    st.write(
        "Lade ein Foto hoch oder fotografiere den Gegenstand. "
        "Die KI erkennt automatisch, um welchen Gegenstand es sich handelt."
    )

    uploaded_file = st.file_uploader(
        "📷 Bild des Fundstücks",
        type=["jpg", "jpeg", "png"]
    )

    camera_file = st.camera_input(
        "Oder direkt fotografieren"
    )

    image = None

    if camera_file:
        image = Image.open(camera_file)

    elif uploaded_file:
        image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # KI ERKENNUNG
    # --------------------------------------------------------

    if image is not None:

        st.image(
            image,
            caption="Ausgewähltes Fundstück",
            use_column_width=True
        )

        if st.button(
            "🤖 Gegenstand mit KI erkennen",
            use_container_width=True
        ):

            with st.spinner("Die KI analysiert das Bild ..."):

                try:
                    label, confidence = classify_image(image)

                    st.session_state.ai_label = label
                    st.session_state.ai_confidence = confidence

                    st.rerun()

                except Exception as e:

                    st.error(
                        "❌ Die KI konnte das Bild nicht analysieren."
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
            st.session_state.ai_confidence * 100
        )

        st.markdown(
            f"""
            <div class="ai-box">
                <h3>🤖 KI-Erkennung</h3>
                <p>
                    Die KI hat erkannt:
                    <strong>{st.session_state.ai_label}</strong>
                </p>
                <p>
                    Kategorie:
                    <strong>{category}</strong>
                </p>
                <p>
                    Sicherheit:
                    <strong>{confidence:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # WEITERE DATEN
        # ----------------------------------------------------

        st.subheader("Fundstück-Daten")

        location = st.text_input(
            "📍 Fundort *",
            placeholder="z. B. Sporthalle, Raum 204 ..."
        )

        size = st.text_input(
            "📏 Größe",
            placeholder="z. B. S, M, L ..."
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder="z. B. schwarz, blau ..."
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder="Weitere Merkmale des Fundstücks ..."
        )

        st.info(
            f"Die Bezeichnung wird automatisch von der KI übernommen: "
            f"**{st.session_state.ai_label}**"
        )

        if st.button(
            "💾 Fundstück speichern",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib noch den Fundort ein."
                )

            else:

                # Bild dauerhaft speichern
                image_id = str(uuid.uuid4())
                image_filename = f"{image_id}.jpg"

                image_path = save_image(
                    image,
                    image_filename
                )

                # Alle aktuellen Daten laden
                items = load_items()

                # Neues Fundstück
                new_item = {
                    "id": str(uuid.uuid4()),
                    "name": st.session_state.ai_label,
                    "category": category,
                    "location": location.strip(),
                    "size": size.strip(),
                    "color": color.strip(),
                    "description": description.strip(),
                    "date": datetime.now().strftime("%d.%m.%Y"),
                    "image": image_path
                }

                items.append(new_item)

                # DAUERHAFT speichern
                save_items(items)

                # KI-Daten zurücksetzen
                st.session_state.ai_label = None
                st.session_state.ai_confidence = None

                st.success(
                    "✅ Fundstück wurde dauerhaft gespeichert!"
                )

                st.info(
                    "Das Fundstück ist jetzt auch für andere Nutzer sichtbar."
                )

                st.session_state.page = "Übersicht"

                st.rerun()


# ============================================================
# SUCHEN
# ============================================================

elif st.session_state.page == "Suchen":

    st.subheader("🔎 Fundstücke suchen")

    items = load_items()

    search = st.text_input(
        "Suchbegriff",
        placeholder="Was suchst du?"
    )

    if search:

        search_lower = search.lower()

        results = [
            item for item in items
            if search_lower in item.get("name", "").lower()
            or search_lower in item.get("location", "").lower()
            or search_lower in item.get("category", "").lower()
            or search_lower in item.get("color", "").lower()
        ]

        if results:

            st.write(
                f"**{len(results)} Fundstück(e) gefunden**"
            )

            for item in results:

                st.markdown(
                    f"""
                    <div class="card">
                        <h3>{item.get('name', 'Unbekannt')}</h3>
                        <p>
                            Kategorie: {item.get('category', '-')}
                        </p>
                        <p>
                            Fundort: {item.get('location', '-')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning(
                "Kein passendes Fundstück gefunden."
            )

    elif not items:

        st.info(
            "Es sind noch keine Fundstücke gespeichert."
        )


# ============================================================
# DETAILS
# ============================================================

elif st.session_state.page == "Details":

    st.subheader("Fundstück-Details")

    items = load_items()

    selected_id = st.session_state.selected

    item = None

    for current_item in items:

        if current_item.get("id") == selected_id:
            item = current_item
            break

    # Falls alte Nummer statt ID verwendet wird
    if item is None and isinstance(selected_id, int):

        if 0 <= selected_id < len(items):
            item = items[selected_id]

    if item is None:

        st.warning(
            "Dieses Fundstück wurde nicht gefunden."
        )

        if st.button("← Zurück"):
            st.session_state.page = "Übersicht"
            st.rerun()

    else:

        col1, col2 = st.columns(2)

        with col1:

            image_file = item.get("image")

            if image_file and Path(image_file).exists():

                st.image(
                    image_file,
                    caption=item.get("name", "Fundstück"),
                    use_column_width=True
                )

        with col2:

            st.markdown(
                f"# {item.get('name', 'Unbekannt')}"
            )

            st.write(
                f"**Kategorie:** "
                f"{item.get('category', '-')}"
            )

            st.write(
                f"**Datum:** "
                f"{item.get('date', '-')}"
            )

            st.write(
                f"**Fundort:** "
                f"{item.get('location', '-')}"
            )

            st.write(
                f"**Fundnummer:** "
                f"{item.get('id', '-')}"
            )

            st.write(
                f"**Größe:** "
                f"{item.get('size', '-')}"
            )

            st.write(
                f"**Farbe:** "
                f"{item.get('color', '-')}"
            )

            st.write(
                f"**Beschreibung:** "
                f"{item.get('description', '-')}"
            )

        st.divider()

        if st.button(
            "📅 Abholtermin vereinbaren",
            use_container_width=True
        ):
            st.success(
                "Ein Abholtermin kann hier vereinbart werden."
            )

        st.write("")

        if st.button(
            "🗑️ Fundstück löschen",
            use_container_width=True
        ):

            items = [
                x for x in items
                if x.get("id") != item.get("id")
            ]

            save_items(items)

            st.success(
                "Fundstück wurde gelöscht."
            )

            st.session_state.page = "Übersicht"
            st.session_state.selected = None

            st.rerun()


# ============================================================
# ANMELDEN
# ============================================================

elif st.session_state.page == "Anmelden":

    st.subheader("Anmelden")

    st.info(
        "Die Anmeldung kann später mit einem richtigen "
        "Schul-Login verbunden werden."
    )

    username = st.text_input("Benutzername")
    password = st.text_input(
        "Passwort",
        type="password"
    )

    if st.button(
        "Anmelden",
        use_container_width=True
    ):

        st.warning(
            "Die Anmeldefunktion ist noch nicht aktiviert."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Digitales Fundbüro • KI-gestützte Fundstück-Erkennung"
)
