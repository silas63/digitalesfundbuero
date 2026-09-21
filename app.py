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
                radial-gradient(circle at 10% 0%, rgba(99,102,241,.13), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 25%),
                #f7f8fc;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #111827 0%, #312e81 55%, #0369a1 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 45px rgba(15,23,42,.18);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .hero p {
            margin: .55rem 0 0 0;
            color: rgba(255,255,255,.82);
            font-size: 1.08rem;
        }

        .stat-card {
            background: white;
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15,23,42,.06);
            margin-bottom: 1rem;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 800;
            color: #111827;
        }

        .stat-label {
            color: #6b7280;
            font-size: .9rem;
        }

        .item-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,.05);
        }

        .badge {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: .82rem;
            font-weight: 700;
        }

        .muted {
            color: #6b7280;
            font-size: .9rem;
        }

        .success-box {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            margin: 1rem 0;
        }

        .info-box {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e40af;
            margin: 1rem 0;
        }

        div.stButton > button {
            border-radius: 12px;
            font-weight: 700;
        }

        [data-testid="stFileUploader"] {
            border-radius: 16px;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            margin-top: 3rem;
            font-size: .85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATENBANK
# =========================================================

def load_items():
    """Lädt alle Fundstücke aus der JSON-Datei."""
    if not ITEMS_FILE.exists():
        return []

    try:
        with open(ITEMS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []


def save_items(items):
    """Speichert alle Fundstücke sicher."""
    DATA_DIR.mkdir(exist_ok=True)

    temp_file = DATA_DIR / "fundstuecke.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)

        temp_file.replace(ITEMS_FILE)

    except OSError as error:
        st.error(f"Fehler beim Speichern: {error}")


# =========================================================
# BILDER
# =========================================================

def image_to_base64(image):
    """Wandelt ein Bild in komprimiertes JPEG/Base64 um."""
    try:
        image = image.convert("RGB")

        # Große Bilder verkleinern, damit die JSON-Datei
        # nicht unnötig riesig wird.
        max_size = (1200, 1200)
        image.thumbnail(max_size)

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=82,
            optimize=True,
        )

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception:
        return None


def base64_to_image(data):
    """Wandelt Base64 zurück in ein PIL-Bild um."""
    if not data:
        return None

    try:
        raw = base64.b64decode(data)
        image = Image.open(BytesIO(raw))
        return image.convert("RGB")

    except Exception:
        return None


def show_image(image, caption=None):
    """
    Robuste zentrale Bildanzeige.

    Das Bild wird zunächst in JPEG-Bytes umgewandelt.
    Dadurch vermeiden wir Probleme zwischen PIL,
    Streamlit-Versionen und unterschiedlichen Bildformaten.
    """
    if image is None:
        st.info("📷 Kein Bild vorhanden.")
        return

    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()

        if caption:
            st.image(image_bytes, caption=caption)
        else:
            st.image(image_bytes)

    except Exception:
        st.info("📷 Dieses Bild konnte leider nicht angezeigt werden.")


# =========================================================
# KI-MODELL
# =========================================================

def load_labels():
    """Lädt die Labels des Teachable-Machine-Modells."""
    if not LABELS_PATH.exists():
        return []

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            labels = []

            for line in file:
                value = line.strip()

                # Teachable Machine kann Labels manchmal so liefern:
                # 0 Rucksack
                # 1 Trinkflasche
                if value:
                    parts = value.split(maxsplit=1)

                    if len(parts) == 2 and parts[0].isdigit():
                        value = parts[1]

                    labels.append(value)

            return labels

    except OSError:
        return []


@st.cache_resource
def load_model():
    """Lädt das Keras-Modell einmalig."""
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
    """
    Klassifiziert ein Bild mit dem Teachable-Machine-Modell.
    Gibt Label und Wahrscheinlichkeit zurück.
    """
    model = load_model()
    labels = load_labels()

    if model is None:
        return "Modell nicht gefunden", 0.0

    try:
        # Teachable Machine nutzt normalerweise 224x224.
        image = image.convert("RGB")
        resized = image.resize((224, 224))

        array = np.asarray(resized).astype(np.float32)

        # Standard-Normalisierung für Teachable Machine.
        array = (array / 127.5) - 1.0

        array = np.expand_dims(array, axis=0)

        prediction = model.predict(array, verbose=0)

        probabilities = prediction[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        if index < len(labels):
            label = labels[index]
        else:
            label = f"Klasse {index + 1}"

        return label, confidence

    except Exception:
        return "Erkennung fehlgeschlagen", 0.0


# =========================================================
# KATEGORIEN
# =========================================================

def get_category(label):
    """Ordnet KI-Labels einer übersichtlichen Kategorie zu."""
    text = label.lower()

    if any(word in text for word in [
        "rucksack",
        "tasche",
        "beutel",
        "handtasche",
        "schulranzen",
    ]):
        return "🎒 Tasche / Rucksack"

    if any(word in text for word in [
        "flasche",
        "trinkflasche",
        "wasserflasche",
    ]):
        return "🥤 Flasche"

    if any(word in text for word in [
        "schlüssel",
        "key",
    ]):
        return "🔑 Schlüssel"

    if any(word in text for word in [
        "jacke",
        "mantel",
        "hoodie",
        "pullover",
        "shirt",
        "tshirt",
        "kleidung",
    ]):
        return "👕 Kleidung"

    if any(word in text for word in [
        "schuh",
        "turnschuh",
        "sneaker",
    ]):
        return "👟 Schuhe"

    if any(word in text for word in [
        "handy",
        "smartphone",
        "telefon",
    ]):
        return "📱 Handy"

    if any(word in text for word in [
        "kopfhörer",
        "headset",
        "airpods",
    ]):
        return "🎧 Kopfhörer"

    if any(word in text for word in [
        "brille",
        "sonnenbrille",
    ]):
        return "👓 Brille"

    if any(word in text for word in [
        "buch",
        "heft",
        "ordner",
        "block",
    ]):
        return "📚 Schule"

    return "📦 Sonstiges"


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def format_date(value):
    """Formatiert ISO-Datum schön."""
    if not value:
        return "Unbekannt"

    try:
        date = datetime.fromisoformat(value)
        return date.strftime("%d.%m.%Y, %H:%M Uhr")
    except (ValueError, TypeError):
        return str(value)


def item_matches_search(item, search):
    """Durchsucht alle wichtigen Textfelder."""
    search = search.lower().strip()

    if not search:
        return True

    fields = [
        item.get("category", ""),
        item.get("ai_label", ""),
        item.get("location", ""),
        item.get("size", ""),
        item.get("color", ""),
        item.get("description", ""),
    ]

    combined = " ".join(str(field) for field in fields).lower()

    return search in combined


def get_image_for_item(item):
    """Holt das gespeicherte Bild."""
    image_data = item.get("image_data")

    if image_data:
        return base64_to_image(image_data)

    # Rückwärtskompatibilität für ältere Datensätze.
    image_path = item.get("image_path")

    if image_path:
        path = Path(image_path)

        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass

    return None


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


# =========================================================
# DATEN LADEN
# =========================================================

items = load_items()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔎 Digitales Fundbüro</h1>
        <p>
            Fundstücke digital erfassen, automatisch erkennen
            und schnell wiederfinden.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

nav_columns = st.columns(4)

with nav_columns[0]:
    if st.button("🏠 Übersicht", use_container_width=True):
        st.session_state.page = "Übersicht"
        st.rerun()

with nav_columns[1]:
    if st.button("➕ Neuer Fund", use_container_width=True):
        st.session_state.page = "Neuer Fund"
        st.rerun()

with nav_columns[2]:
    if st.button("🔍 Suchen", use_container_width=True):
        st.session_state.page = "Suchen"
        st.rerun()

with nav_columns[3]:
    if st.button("📋 Details", use_container_width=True):
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

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Gespeicherte Fundstücke</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stat2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(categories)}</div>
                <div class="stat-label">Kategorien</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stat3:
        model_available = load_model() is not None

        if model_available:
            model_status = "🟢 Aktiv"
        else:
            model_status = "🔴 Nicht gefunden"

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{model_status}</div>
                <div class="stat-label">KI-Erkennung</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if not items:
        st.markdown(
            """
            <div class="info-box">
                <strong>👋 Noch keine Fundstücke vorhanden.</strong><br>
                Lege über „Neuer Fund“ dein erstes Fundstück an.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.subheader("🆕 Zuletzt hinzugefügt")

        recent_items = list(reversed(items[-8:]))

        for item in recent_items:
            item_id = item.get("id")
            category = item.get("category", "📦 Sonstiges")
            location = item.get("location", "Unbekannt")
            description = item.get("description", "")
            date = format_date(item.get("date"))

            st.markdown('<div class="item-card">', unsafe_allow_html=True)

            left, right = st.columns([1, 2])

            with left:
                image = get_image_for_item(item)
                show_image(image)

            with right:
                st.markdown(
                    f'<span class="badge">{category}</span>',
                    unsafe_allow_html=True,
                )

                st.markdown(f"### {category}")

                if description:
                    st.write(description)
                else:
                    st.write("Keine Beschreibung angegeben.")

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
                    "Details anzeigen",
                    key=f"overview_details_{item_id}",
                ):
                    st.session_state.selected_item_id = item_id
                    st.session_state.page = "Details"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NEUER FUND
# =========================================================

elif st.session_state.page == "Neuer Fund":

    st.subheader("➕ Neues Fundstück")

    st.write(
        "Lade ein Foto hoch oder nutze die Kamera. "
        "Die KI versucht anschließend, das Fundstück zu erkennen."
    )

    upload_tab, camera_tab = st.tabs(
        ["📁 Bild hochladen", "📷 Kamera"]
    )

    uploaded_file = None
    camera_image = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Bild auswählen",
            type=["jpg", "jpeg", "png", "webp"],
            help="Unterstützte Bildformate: JPG, JPEG, PNG und WEBP.",
        )

    with camera_tab:
        camera_image = st.camera_input("Foto aufnehmen")

    image_source = uploaded_file if uploaded_file else camera_image

    selected_image = None

    if image_source is not None:
        try:
            selected_image = Image.open(image_source).convert("RGB")

            st.markdown("### 🖼️ Vorschau")
            show_image(
                selected_image,
                caption="Ausgewähltes Fundstück",
            )

        except Exception:
            st.error(
                "❌ Das Bild konnte nicht gelesen werden. "
                "Bitte versuche ein anderes Bild."
            )

    if selected_image is not None:

        st.markdown("### 🤖 KI-Erkennung")

        if st.button(
            "✨ Fundstück erkennen",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("🔎 KI analysiert das Bild ..."):
                label, confidence = classify_image(selected_image)

            st.session_state.ai_label = label
            st.session_state.ai_confidence = confidence

            st.success("✅ Analyse abgeschlossen!")

        ai_label = st.session_state.ai_label
        ai_confidence = st.session_state.ai_confidence

        if ai_label:
            category = get_category(ai_label)

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"🤖 Erkennung: **{ai_label}**")

            with col2:
                st.info(
                    f"🎯 Sicherheit: **{ai_confidence * 100:.1f}%**"
                )

            st.success(f"📦 Kategorie: **{category}**")

        st.divider()

        st.markdown("### 📝 Angaben zum Fundstück")

        location = st.text_input(
            "📍 Fundort",
            placeholder="z. B. Sporthalle, Raum 204, Schulhof ...",
        )

        size = st.text_input(
            "📏 Größe",
            placeholder="z. B. klein, mittel, groß oder Größe M",
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder="z. B. schwarz, blau, rot ...",
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder=(
                "Weitere Merkmale, Aufdrucke, Besonderheiten ..."
            ),
            height=120,
        )

        save_button = st.button(
            "💾 Fundstück speichern",
            type="primary",
            use_container_width=True,
        )

        if save_button:

            if not location.strip():
                st.warning("⚠️ Bitte gib mindestens den Fundort an.")

            else:
                image_data = image_to_base64(selected_image)

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

                    final_category = get_category(final_label)

                    new_item = {
                        "id": str(uuid.uuid4()),
                        "category": final_category,
                        "ai_label": final_label,
                        "ai_confidence": st.session_state.ai_confidence,
                        "location": location.strip(),
                        "size": size.strip(),
                        "color": color.strip(),
                        "description": description.strip(),
                        "image_data": image_data,
                        "date": datetime.now().isoformat(),
                    }

                    items.append(new_item)
                    save_items(items)

                    st.session_state.selected_item_id = new_item["id"]
                    st.session_state.ai_label = ""
                    st.session_state.ai_confidence = 0.0

                    st.success(
                        "🎉 Fundstück wurde erfolgreich gespeichert!"
                    )

                    st.balloons()

                    st.session_state.page = "Details"
                    st.rerun()


# =========================================================
# SUCHEN
# =========================================================

elif st.session_state.page == "Suchen":

    st.subheader("🔍 Fundstücke suchen")

    search = st.text_input(
        "Suchbegriff",
        placeholder=(
            "z. B. Rucksack, schwarz, Sporthalle, Schlüssel ..."
        ),
    )

    category_filter = st.selectbox(
        "📦 Kategorie",
        [
            "Alle Kategorien",
            *sorted(
                set(
                    item.get("category", "📦 Sonstiges")
                    for item in items
                )
            ),
        ],
    )

    results = []

    for item in items:

        if not item_matches_search(item, search):
            continue

        if (
            category_filter != "Alle Kategorien"
            and item.get("category") != category_filter
        ):
            continue

        results.append(item)

    st.write(f"**{len(results)} Fundstück(e) gefunden.**")

    if not results:

        st.info(
            "🔎 Keine passenden Fundstücke gefunden."
        )

    else:

        for item in reversed(results):

            item_id = item.get("id")
            category = item.get("category", "📦 Sonstiges")
            location = item.get("location", "Unbekannt")
            color = item.get("color", "")
            size = item.get("size", "")
            description = item.get("description", "")

            st.markdown(
                '<div class="item-card">',
                unsafe_allow_html=True,
            )

            left, right = st.columns([1, 2])

            with left:
                image = get_image_for_item(item)
                show_image(image)

            with right:

                st.markdown(
                    f'<span class="badge">{category}</span>',
                    unsafe_allow_html=True,
                )

                st.markdown(f"### {category}")

                if description:
                    st.write(description)

                details = []

                if location:
                    details.append(f"📍 {location}")

                if color:
                    details.append(f"🎨 {color}")

                if size:
                    details.append(f"📏 {size}")

                if details:
                    st.write(" • ".join(details))

                if st.button(
                    "📋 Details",
                    key=f"search_details_{item_id}",
                ):
                    st.session_state.selected_item_id = item_id
                    st.session_state.page = "Details"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DETAILS
# =========================================================

elif st.session_state.page == "Details":

    st.subheader("📋 Fundstück-Details")

    selected_item = None

    if st.session_state.selected_item_id:

        for item in items:
            if item.get("id") == st.session_state.selected_item_id:
                selected_item = item
                break

    if selected_item is None:

        st.info(
            "ℹ️ Wähle zuerst ein Fundstück aus."
        )

        if items:
            options = {}

            for item in items:
                item_id = item.get("id")
                category = item.get(
                    "category",
                    "📦 Sonstiges",
                )
                location = item.get(
                    "location",
                    "Unbekannt",
                )

                label = f"{category} — {location}"
                options[label] = item_id

            selected_label = st.selectbox(
                "Fundstück auswählen",
                list(options.keys()),
            )

            if st.button(
                "Fundstück öffnen",
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

        confidence = selected_item.get(
            "ai_confidence",
            0.0,
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
            selected_item.get("date")
        )

        left, right = st.columns([1, 1])

        with left:

            image = get_image_for_item(selected_item)

            show_image(
                image,
                caption="Fundstück",
            )

        with right:

            st.markdown(
                f'<span class="badge">{category}</span>',
                unsafe_allow_html=True,
            )

            st.markdown(f"## {category}")

            st.markdown(
                f"""
                <div class="item-card">
                    <strong>🤖 KI-Erkennung</strong><br>
                    {ai_label}<br><br>

                    <strong>🎯 Sicherheit</strong><br>
                    {confidence * 100:.1f}%<br><br>

                    <strong>📍 Fundort</strong><br>
                    {location}<br><br>

                    <strong>📏 Größe</strong><br>
                    {size if size else "Keine Angabe"}<br><br>

                    <strong>🎨 Farbe</strong><br>
                    {color if color else "Keine Angabe"}<br><br>

                    <strong>🕒 Eingetragen</strong><br>
                    {date}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 📝 Beschreibung")

        if description:
            st.write(description)
        else:
            st.write("Keine Beschreibung vorhanden.")

        st.divider()

        delete_col, back_col = st.columns(2)

        with back_col:

            if st.button(
                "⬅️ Zurück zur Übersicht",
                use_container_width=True,
            ):
                st.session_state.page = "Übersicht"
                st.rerun()

        with delete_col:

            delete_button = st.button(
                "🗑️ Fundstück löschen",
                use_container_width=True,
            )

            if delete_button:
                st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete", False):

            st.warning(
                "⚠️ Möchtest du dieses Fundstück wirklich löschen?"
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:

                if st.button(
                    "Ja, endgültig löschen",
                    type="primary",
                    use_container_width=True,
                ):

                    selected_id = selected_item.get("id")

                    items = [
                        item
                        for item in items
                        if item.get("id") != selected_id
                    ]

                    save_items(items)

                    st.session_state.selected_item_id = None
                    st.session_state.confirm_delete = False
                    st.session_state.page = "Übersicht"

                    st.success(
                        "🗑️ Fundstück wurde gelöscht."
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
        🔎 Digitales Fundbüro • KI-gestützte Fundstück-Erkennung
    </div>
    """,
    unsafe_allow_html=True,
)
