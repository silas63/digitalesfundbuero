import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime


# =========================================================
# KONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f8fc;
}

.main-header {
    background: linear-gradient(90deg, #145da0, #2878c8);
    padding: 18px 30px;
    border-radius: 0 0 12px 12px;
    color: white;
    margin-bottom: 25px;
}

.main-header h1 {
    margin: 0;
    font-size: 32px;
    color: white;
}

.main-header p {
    margin: 5px 0 0 0;
    color: white;
    opacity: 0.9;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dce5ef;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

.badge {
    background-color: #d9f7df;
    color: #14732c;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: bold;
    display: inline-block;
}

.ai-box {
    background-color: #eef7ff;
    border: 1px solid #b9ddff;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}

.empty-box {
    background-color: white;
    padding: 40px;
    border-radius: 12px;
    border: 1px solid #dce5ef;
    text-align: center;
    margin-top: 20px;
}

h1, h2, h3 {
    color: #14263d;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LABELS LADEN
# =========================================================

def load_labels():

    if not LABELS_PATH.exists():

        st.error("❌ labels.txt wurde nicht gefunden.")

        st.info(
            "Lade labels.txt in denselben GitHub-Ordner "
            "wie app.py und keras_model.h5."
        )

        st.stop()

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return labels


# =========================================================
# KI-MODELL LADEN
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error("❌ keras_model.h5 wurde nicht gefunden.")

        st.info(
            "Lade keras_model.h5 in denselben GitHub-Ordner "
            "wie app.py."
        )

        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# =========================================================
# BILD KLASSIFIZIEREN
# =========================================================

def classify_image(image, model, labels):

    input_shape = model.input_shape

    height = int(input_shape[1])
    width = int(input_shape[2])

    image = image.convert("RGB")
    image = image.resize((width, height))

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
    )[0]

    best_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[best_index]
    )

    if best_index < len(labels):
        label = labels[best_index]
    else:
        label = f"Klasse {best_index}"

    return label, confidence, prediction


# =========================================================
# KATEGORIE AUS KI-ERGEBNIS
# =========================================================

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

    if "federtasche" in label:
        return "Sonstiges"

    return "Sonstiges"


# =========================================================
# LEERE LISTE
# =========================================================
#
# WICHTIG:
# Es gibt KEINE Demo-Fundstücke.
# Das Fundbüro startet komplett leer.
#

def create_demo_items():
    return []


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Übersicht"


if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = []


if "selected" not in st.session_state:
    st.session_state.selected = 0


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<h1>🔎 Fundbüro | Kleidung</h1>

<p>
Digitales Fundbüro · Fundstücke einfach finden und verwalten
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("### 🔎 Suche")

    search = st.text_input(
        "Suche",
        placeholder="Suche nach Gegenstand ..."
    )

    st.divider()

    if st.button(
        "🏠 Übersicht",
        use_container_width=True
    ):

        st.session_state.page = "Übersicht"

    if st.button(
        "🔎 Suchen",
        use_container_width=True
    ):

        st.session_state.page = "Suchen"

    if st.button(
        "➕ Neuer Fund",
        use_container_width=True
    ):

        st.session_state.page = "Neuer Fund"

    if st.button(
        "👤 Anmelden",
        use_container_width=True
    ):

        st.session_state.page = "Anmelden"

    st.divider()

    st.caption(
        "Schulprojekt – Digitales Fundbüro"
    )

    st.caption(
        "KI-Erkennung mit Teachable Machine"
    )


# =========================================================
# MODEL UND LABELS
# =========================================================

model = load_model()
labels = load_labels()


# =========================================================
# ÜBERSICHT
# =========================================================

if st.session_state.page == "Übersicht":

    st.header("Alle Fundstücke")

    categories = [
        "Alle",
        "Oberteile",
        "Hosen",
        "Schuhe",
        "Sonstiges"
    ]

    selected_category = st.radio(
        "Kategorie",
        categories,
        horizontal=True
    )

    # Fundstücke aus Session State holen
    items = list(
        st.session_state.fundstuecke
    )

    # Suche
    if search:

        search_lower = search.lower()

        items = [
            item
            for item in items
            if (
                search_lower
                in item["name"].lower()
            )
            or (
                search_lower
                in item["location"].lower()
            )
            or (
                search_lower
                in item["description"].lower()
            )
            or (
                search_lower
                in item["category"].lower()
            )
            or (
                search_lower
                in item["color"].lower()
            )
        ]

    # Kategorie
    if selected_category != "Alle":

        items = [
            item
            for item in items
            if item["category"]
            == selected_category
        ]

    # =====================================================
    # KEINE FUNDSTÜCKE
    # =====================================================

    if not items:

        st.markdown("""
        <div class="empty-box">

        <h2>📦 Noch keine Fundstücke</h2>

        <p>
        Hier wurden noch keine Fundstücke eingetragen.
        </p>

        <p>
        Gehe auf <b>„Neuer Fund“</b>, um das erste
        Fundstück einzutragen.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button(
            "➕ Erstes Fundstück eintragen",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "Neuer Fund"
            st.rerun()

    # =====================================================
    # FUNDSTÜCKE ANZEIGEN
    # =====================================================

    else:

        columns = st.columns(3)

        for index, item in enumerate(items):

            with columns[index % 3]:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    item["name"]
                )

                st.markdown(
                    '<span class="badge">'
                    'Gefunden'
                    '</span>',
                    unsafe_allow_html=True
                )

                st.write("")

                st.write(
                    f"📅 {item['date']}"
                )

                st.write(
                    f"📍 {item['location']}"
                )

                st.write(
                    f"🔢 {item['number']}"
                )

                if st.button(
                    "Details →",
                    key=f"detail_{index}"
                ):

                    original_index = (
                        st.session_state
                        .fundstuecke
                        .index(item)
                    )

                    st.session_state.selected = (
                        original_index
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

    st.header("➕ Neuen Fund eintragen")

    st.write(
        "Hier kann eine Person ein Fundstück "
        "eintragen und optional ein Foto "
        "von der KI erkennen lassen."
    )

    st.divider()

    # =====================================================
    # BILD
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        uploaded_file = st.file_uploader(
            "📁 Foto hochladen",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    with col2:

        camera_file = st.camera_input(
            "📸 Kamera benutzen"
        )

    image_file = (
        camera_file
        if camera_file is not None
        else uploaded_file
    )

    if image_file is not None:

        image = Image.open(
            image_file
        ).convert("RGB")

        # Für ältere Streamlit-Versionen:
        st.image(
            image,
            caption="Fundstück",
            use_column_width=True
        )

        if st.button(
            "🤖 KI-Erkennung starten",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Die KI analysiert das Bild ..."
            ):

                try:

                    label, confidence, prediction = (
                        classify_image(
                            image,
                            model,
                            labels
                        )
                    )

                    st.session_state.ai_label = (
                        label
                    )

                    st.session_state.ai_confidence = (
                        confidence
                    )

                    st.session_state.ai_prediction = (
                        prediction
                    )

                except Exception as error:

                    st.error(
                        "❌ Die KI-Erkennung konnte "
                        "nicht durchgeführt werden."
                    )

                    st.code(
                        str(error)
                    )

    # =====================================================
    # KI RESULTAT
    # =====================================================

    if "ai_label" in st.session_state:

        st.markdown(
            '<div class="ai-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🤖 KI-Erkennung"
        )

        st.success(
            "Erkannt: "
            + st.session_state.ai_label
        )

        st.metric(
            "Sicherheit",
            f"{st.session_state.ai_confidence * 100:.1f} %"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # =====================================================
    # FUNDSTÜCK-DATEN
    # =====================================================

    st.subheader(
        "📋 Informationen zum Fundstück"
    )

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input(
            "Bezeichnung",
            value=st.session_state.get(
                "ai_label",
                ""
            ),
            placeholder="z. B. Pullover"
        )

        location = st.text_input(
            "📍 Fundort",
            placeholder=(
                "z. B. Schule, Sporthalle, "
                "Bahnhof ..."
            )
        )

        size = st.selectbox(
            "📏 Größe",
            [
                "Keine Angabe",
                "XS",
                "S",
                "M",
                "L",
                "XL"
            ]
        )

    with col2:

        category_default = "Sonstiges"

        if "ai_label" in st.session_state:

            category_default = get_category(
                st.session_state.ai_label
            )

        category_options = [
            "Oberteile",
            "Hosen",
            "Schuhe",
            "Sonstiges"
        ]

        category = st.selectbox(
            "👕 Kategorie",
            category_options,
            index=category_options.index(
                category_default
            )
        )

        color = st.text_input(
            "🎨 Farbe",
            placeholder=(
                "z. B. rot, blau, schwarz ..."
            )
        )

        description = st.text_area(
            "📝 Beschreibung",
            placeholder=(
                "Besondere Merkmale des "
                "Fundstücks ..."
            )
        )

    st.divider()

    # =====================================================
    # SPEICHERN
    # =====================================================

    if st.button(
        "💾 Fundstück speichern",
        type="primary",
        use_container_width=True
    ):

        if not name.strip():

            st.warning(
                "⚠️ Bitte eine Bezeichnung eingeben."
            )

        elif not location.strip():

            st.warning(
                "⚠️ Bitte einen Fundort eingeben."
            )

        else:

            # Eindeutige Fundnummer
            number = datetime.now().strftime(
                "%Y-%m%d-%H%M%S"
            )

            new_item = {

                "name":
                    name.strip(),

                "category":
                    category,

                "date":
                    datetime.now().strftime(
                        "%d.%m.%Y"
                    ),

                "location":
                    location.strip(),

                "number":
                    number,

                "size":
                    size,

                "color":
                    color.strip()
                    if color.strip()
                    else "Keine Angabe",

                "description":
                    description.strip()
                    if description.strip()
                    else "Keine Beschreibung."

            }

            # Fundstück hinzufügen
            st.session_state.fundstuecke.insert(
                0,
                new_item
            )

            st.success(
                "✅ Fundstück wurde erfolgreich "
                "eingetragen!"
            )

            st.info(
                f"Fundnummer: {number}"
            )

            # KI-Daten zurücksetzen
            st.session_state.pop(
                "ai_label",
                None
            )

            st.session_state.pop(
                "ai_confidence",
                None
            )

            st.session_state.pop(
                "ai_prediction",
                None
            )

            st.session_state.page = (
                "Übersicht"
            )

            st.rerun()


# =========================================================
# SUCHE
# =========================================================

elif st.session_state.page == "Suchen":

    st.header("🔎 Fundstücke suchen")

    query = st.text_input(
        "Suchbegriff",
        placeholder=(
            "z. B. Pullover, Schuhe, "
            "Bahnhof ..."
        )
    )

    if query:

        query_lower = query.lower()

        results = [
            item
            for item
            in st.session_state.fundstuecke
            if query_lower
            in str(item).lower()
        ]

    else:

        results = (
            st.session_state.fundstuecke
        )

    st.write(
        f"**{len(results)} Fundstück(e) gefunden**"
    )

    if not results:

        st.info(
            "🔍 Keine passenden Fundstücke gefunden."
        )

    else:

        for index, item in enumerate(
            results
        ):

            with st.container(
                border=True
            ):

                st.subheader(
                    item["name"]
                )

                st.write(
                    f"📍 {item['location']}  |  "
                    f"📅 {item['date']}  |  "
                    f"🔢 {item['number']}"
                )

                st.write(
                    f"👕 Kategorie: "
                    f"{item['category']}"
                )

                st.write(
                    f"🎨 Farbe: "
                    f"{item['color']}"
                )

                st.write(
                    item["description"]
                )

                if st.button(
                    "Details →",
                    key=f"search_detail_{index}"
                ):

                    original_index = (
                        st.session_state
                        .fundstuecke
                        .index(item)
                    )

                    st.session_state.selected = (
                        original_index
                    )

                    st.session_state.page = (
                        "Details"
                    )

                    st.rerun()


# =========================================================
# DETAILANSICHT
# =========================================================

elif st.session_state.page == "Details":

    # Sicherheitsprüfung
    if not st.session_state.fundstuecke:

        st.info(
            "Es gibt noch keine Fundstücke."
        )

        if st.button(
            "➕ Neuer Fund",
            type="primary"
        ):

            st.session_state.page = (
                "Neuer Fund"
            )

            st.rerun()

    else:

        if (
            st.session_state.selected < 0
            or
            st.session_state.selected
            >= len(
                st.session_state.fundstuecke
            )
        ):

            st.session_state.selected = 0

        item = (
            st.session_state
            .fundstuecke[
                st.session_state.selected
            ]
        )

        st.header(
            "Detailansicht"
        )

        st.title(
            item["name"]
        )

        st.markdown(
            '<span class="badge">'
            'Gefunden'
            '</span>',
            unsafe_allow_html=True
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"📅 **Funddatum:** "
                f"{item['date']}"
            )

            st.write(
                f"📍 **Fundort:** "
                f"{item['location']}"
            )

            st.write(
                f"🔢 **Fundnummer:** "
                f"{item['number']}"
            )

            st.write(
                f"👕 **Kategorie:** "
                f"{item['category']}"
            )

        with col2:

            st.write(
                f"📏 **Größe:** "
                f"{item['size']}"
            )

            st.write(
                f"🎨 **Farbe:** "
                f"{item['color']}"
            )

            st.write(
                f"📝 **Beschreibung:** "
                f"{item['description']}"
            )

        st.divider()

        if st.button(
            "👤 Abholtermin vereinbaren",
            type="primary",
            use_container_width=True
        ):

            st.info(
                "📅 Demo-Funktion: "
                "Hier könnte später ein "
                "Abholtermin vereinbart werden."
            )

        if st.button(
            "← Zurück zur Übersicht",
            use_container_width=True
        ):

            st.session_state.page = (
                "Übersicht"
            )

            st.rerun()


# =========================================================
# ANMELDEN
# =========================================================

elif st.session_state.page == "Anmelden":

    st.header(
        "👤 Anmelden"
    )

    st.write(
        "Melde dich an, um Fundstücke "
        "zu verwalten."
    )

    with st.form(
        "login_form"
    ):

        email = st.text_input(
            "E-Mail"
        )

        password = st.text_input(
            "Passwort",
            type="password"
        )

        submit = st.form_submit_button(
            "Anmelden"
        )

    if submit:

        st.info(
            "Die Anmeldung ist in dieser "
            "Schulprojekt-Version als Demo "
            "vorbereitet."
        )
