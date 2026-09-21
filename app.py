import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import base64
import io
import html
from datetime import datetime


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_DIR = "fundburo_data"
DATA_PATH = os.path.join(DATA_DIR, "fundstuecke.json")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# CLEAN DESIGN
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background: #f6f7f9;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 28px;
        padding-bottom: 60px;
    }

    /* ---------- HEADER ---------- */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 55px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: #111827;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
        box-shadow: 0 5px 18px rgba(17, 24, 39, 0.15);
    }

    .brand-name {
        font-size: 17px;
        font-weight: 800;
        color: #111827;
        line-height: 1;
    }

    .brand-small {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 4px;
        font-weight: 600;
        letter-spacing: .03em;
    }

    /* ---------- HERO ---------- */

    .hero {
        max-width: 720px;
        margin-bottom: 42px;
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 700;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: .12em;
        margin-bottom: 13px;
    }

    .hero-title {
        font-size: clamp(44px, 6vw, 76px);
        line-height: .98;
        letter-spacing: -0.055em;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }

    .hero-description {
        max-width: 500px;
        color: #6b7280;
        font-size: 16px;
        line-height: 1.65;
        margin-top: 22px;
    }

    /* ---------- STATS ---------- */

    .stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 58px;
    }

    .stat {
        background: white;
        border: 1px solid #e8eaf0;
        border-radius: 20px;
        padding: 22px 24px;
    }

    .stat-number {
        font-size: 31px;
        line-height: 1;
        font-weight: 800;
        color: #111827;
    }

    .stat-label {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 8px;
        font-weight: 600;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: #111827;
        margin-bottom: 18px;
    }

    .section-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-top: -10px;
        margin-bottom: 22px;
    }

    /* ---------- FUND CARDS ---------- */

    .fund-card {
        background: white;
        border: 1px solid #e8eaf0;
        border-radius: 22px;
        overflow: hidden;
        margin-bottom: 20px;
        transition:
            transform .18s ease,
            box-shadow .18s ease,
            border-color .18s ease;
    }

    .fund-card:hover {
        transform: translateY(-5px);
        border-color: #d7d9e2;
        box-shadow: 0 18px 45px rgba(17, 24, 39, .10);
    }

    .fund-image-wrap {
        width: 100%;
        height: 220px;
        overflow: hidden;
        background: #eef0f4;
    }

    .fund-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .fund-info {
        padding: 18px 19px 19px 19px;
    }

    .fund-name {
        font-size: 16px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 11px;
    }

    .pills {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        background: #f3f4f6;
        color: #6b7280;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 600;
    }

    .pill-main {
        background: #111827;
        color: white;
    }

    /* ---------- EMPTY ---------- */

    .empty {
        background: white;
        border: 1px dashed #d8dbe3;
        border-radius: 22px;
        padding: 60px 30px;
        text-align: center;
    }

    .empty-icon {
        font-size: 38px;
        margin-bottom: 13px;
    }

    .empty-title {
        font-weight: 750;
        color: #111827;
        font-size: 18px;
    }

    .empty-text {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 7px;
    }

    /* ---------- FORMS ---------- */

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e8eaf0;
        border-radius: 22px;
        padding: 25px;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #e1e4ea;
        font-weight: 650;
        min-height: 42px;
        transition: all .15s ease;
    }

    .stButton > button:hover {
        border-color: #111827;
        transform: translateY(-1px);
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border: none;
        border-top: 1px solid #e7e9ee;
        margin: 40px 0;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 18px;
            padding-right: 18px;
        }

        .stats {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 48px;
        }

        .fund-image-wrap {
            height: 190px;
        }

        .topbar {
            margin-bottom: 40px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA
# =========================================================

def load_items():
    if not os.path.exists(DATA_PATH):
        return []

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


items = load_items()


# =========================================================
# IMAGE HELPERS
# =========================================================

def image_to_base64(image):
    buffer = io.BytesIO()

    image = image.convert("RGB")
    image.thumbnail((1400, 1400))

    image.save(
        buffer,
        format="JPEG",
        quality=82,
        optimize=True,
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(data):
    try:
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def image_bytes(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=85,
    )
    return buffer.getvalue()


def get_item_image(item):
    if item.get("image_data"):
        return base64_to_image(item["image_data"])

    old_path = item.get("image_path")

    if old_path and os.path.exists(old_path):
        try:
            return Image.open(old_path).convert("RGB")
        except Exception:
            pass

    return None


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        return []

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                labels.append(parts[1].strip())
            else:
                labels.append(parts[0].strip())

    return labels


model = load_model()
labels = load_labels()


def classify_image(image):

    if model is None or not labels:
        return "Unbekannt", 0.0

    image = image.convert("RGB").resize((224, 224))

    arr = np.asarray(image).astype(np.float32)

    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index])

    if index >= len(labels):
        return "Unbekannt", confidence

    return labels[index], confidence


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None


def go(page):
    st.session_state.page = page
    st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">⌕</div>
            <div>
                <div class="brand-name">Fundbüro</div>
                <div class="brand-small">DIGITAL</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Digitales Fundbüro</div>

            <div class="hero-title">
                Gefunden.<br>
                Gespeichert.
            </div>

            <div class="hero-description">
                Fundstücke fotografieren, automatisch erkennen
                und später wiederfinden.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = set()

    for item in items:
        category = item.get("category", "").strip()

        if category:
            categories.add(category)

    locations = set()

    for item in items:
        location = item.get("location", "").strip()

        if location:
            locations.add(location)

    st.markdown(
        f"""
        <div class="stats">

            <div class="stat">
                <div class="stat-number">{len(items)}</div>
                <div class="stat-label">Fundstücke</div>
            </div>

            <div class="stat">
                <div class="stat-number">{len(categories)}</div>
                <div class="stat-label">Kategorien</div>
            </div>

            <div class="stat">
                <div class="stat-number">{len(locations)}</div>
                <div class="stat-label">Fundorte</div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        if st.button(
            "＋  Fundstück hinzufügen",
            use_container_width=True,
        ):
            go("new")

    with right:
        if st.button(
            "⌕  Fundstücke durchsuchen",
            use_container_width=True,
        ):
            go("search")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Zuletzt</div>',
        unsafe_allow_html=True,
    )

    if not items:

        st.markdown(
            """
            <div class="empty">
                <div class="empty-icon">⌕</div>
                <div class="empty-title">Noch keine Fundstücke</div>
                <div class="empty-text">
                    Füge dein erstes Fundstück hinzu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        recent = list(reversed(items[-6:]))

        cols = st.columns(3)

        for index, item in enumerate(recent):

            with cols[index % 3]:

                image = get_item_image(item)

                image_html = ""

                if image:
                    encoded = base64.b64encode(
                        image_bytes(image)
                    ).decode("utf-8")

                    image_html = (
                        '<div class="fund-image-wrap">'
                        f'<img class="fund-image" '
                        f'src="data:image/jpeg;base64,{encoded}">'
                        "</div>"
                    )
                else:
                    image_html = (
                        '<div class="fund-image-wrap"></div>'
                    )

                name = html.escape(
                    item.get("name", "Fundstück")
                )

                category = html.escape(
                    item.get("category", "Sonstiges")
                )

                location = html.escape(
                    item.get("location", "Unbekannt")
                )

                st.markdown(
                    f"""
                    <div class="fund-card">

                        {image_html}

                        <div class="fund-info">

                            <div class="fund-name">
                                {name}
                            </div>

                            <div class="pills">

                                <span class="pill pill-main">
                                    {category}
                                </span>

                                <span class="pill">
                                    📍 {location}
                                </span>

                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                item_id = item.get("id")

                if st.button(
                    "Details",
                    key=f"home_details_{item_id}_{index}",
                    use_container_width=True,
                ):
                    st.session_state.selected_item = item_id
                    go("details")


# =========================================================
# NEW ITEM
# =========================================================

elif st.session_state.page == "new":

    if st.button("← Zurück"):
        go("home")

    st.markdown(
        """
        <div class="hero" style="margin-top:30px;">
            <div class="hero-kicker">Neuer Eintrag</div>
            <div class="hero-title" style="font-size:52px;">
                Fundstück<br>hinzufügen.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Foto",
        type=["jpg", "jpeg", "png", "webp"],
    )

    camera = st.camera_input("Oder Kamera verwenden")

    selected_image = uploaded or camera

    image = None

    if selected_image:
        try:
            image = Image.open(selected_image).convert("RGB")

            st.image(
                image,
                width=500,
            )

        except Exception:
            st.error("Das Bild konnte nicht gelesen werden.")

    if image:

        if st.button(
            "Bild automatisch erkennen",
            use_container_width=True,
        ):

            category, confidence = classify_image(image)

            st.session_state.detected_category = category
            st.session_state.detected_confidence = confidence

            st.rerun()

    detected = st.session_state.get(
        "detected_category",
        "",
    )

    if detected:

        confidence = st.session_state.get(
            "detected_confidence",
            0.0,
        )

        st.success(
            f"Erkannt: {detected} "
            f"({confidence * 100:.0f} %)"
        )

    with st.form("new_item_form"):

        name = st.text_input(
            "Name",
            placeholder="z. B. Schwarzer Rucksack",
        )

        category = st.text_input(
            "Kategorie",
            value=detected,
            placeholder="z. B. Rucksack",
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle",
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Kurze Beschreibung …",
            height=100,
        )

        submitted = st.form_submit_button(
            "Fundstück speichern",
            use_container_width=True,
        )

    if submitted:

        if not image:
            st.warning("Bitte zuerst ein Foto hinzufügen.")

        elif not name.strip():
            st.warning("Bitte einen Namen eingeben.")

        else:

            new_item = {
                "id": datetime.now().strftime(
                    "%Y%m%d%H%M%S%f"
                ),
                "name": name.strip(),
                "category": category.strip() or "Sonstiges",
                "location": location.strip() or "Unbekannt",
                "description": description.strip(),
                "date": datetime.now().strftime("%d.%m.%Y"),
                "image_data": image_to_base64(image),
            }

            items.append(new_item)
            save_items(items)

            st.session_state.pop(
                "detected_category",
                None,
            )

            st.session_state.pop(
                "detected_confidence",
                None,
            )

            st.success("Fundstück gespeichert.")
            go("home")


# =========================================================
# SEARCH
# =========================================================

elif st.session_state.page == "search":

    if st.button("← Zurück"):
        go("home")

    st.markdown(
        """
        <div class="hero" style="margin-top:30px;">
            <div class="hero-kicker">Suche</div>
            <div class="hero-title" style="font-size:52px;">
                Wiederfinden.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Suche",
        placeholder="Name, Kategorie oder Fundort …",
        label_visibility="collapsed",
    )

    query = query.lower().strip()

    if query:

        results = []

        for item in items:

            searchable = " ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("category", "")),
                    str(item.get("location", "")),
                    str(item.get("description", "")),
                ]
            ).lower()

            if query in searchable:
                results.append(item)

    else:
        results = list(reversed(items))

    st.markdown(
        f'<div class="section-title">{len(results)} Fundstücke</div>',
        unsafe_allow_html=True,
    )

    if not results:

        st.markdown(
            """
            <div class="empty">
                <div class="empty-icon">⌕</div>
                <div class="empty-title">Nichts gefunden</div>
                <div class="empty-text">
                    Versuch einen anderen Suchbegriff.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        cols = st.columns(3)

        for index, item in enumerate(results):

            with cols[index % 3]:

                image = get_item_image(item)

                if image:
                    st.image(
                        image,
                        width=500,
                    )

                st.markdown(
                    f"**{item.get('name', 'Fundstück')}**"
                )

                st.caption(
                    f"{item.get('category', 'Sonstiges')}  ·  "
                    f"📍 {item.get('location', 'Unbekannt')}"
                )

                if st.button(
                    "Details",
                    key=f"search_details_{item.get('id')}_{index}",
                    use_container_width=True,
                ):
                    st.session_state.selected_item = item.get("id")
                    go("details")


# =========================================================
# DETAILS
# =========================================================

elif st.session_state.page == "details":

    if st.button("← Zurück"):
        go("home")

    selected_id = st.session_state.selected_item

    item = None

    for current in items:

        if current.get("id") == selected_id:
            item = current
            break

    if not item:

        st.warning("Fundstück nicht gefunden.")

        if st.button("Zur Übersicht"):
            go("home")

    else:

        st.markdown(
            """
            <div class="hero" style="margin-top:30px;">
                <div class="hero-kicker">Fundstück</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.15, 1])

        with left:

            image = get_item_image(item)

            if image:
                st.image(
                    image,
                    width=700,
                )

        with right:

            name = item.get(
                "name",
                "Fundstück",
            )

            category = item.get(
                "category",
                "Sonstiges",
            )

            location = item.get(
                "location",
                "Unbekannt",
            )

            description = item.get(
                "description",
                "",
            )

            date = item.get(
                "date",
                "",
            )

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #e8eaf0;
                    border-radius:22px;
                    padding:26px;
                ">

                    <div style="
                        font-size:30px;
                        font-weight:800;
                        letter-spacing:-.04em;
                        color:#111827;
                        margin-bottom:20px;
                    ">
                        {html.escape(str(name))}
                    </div>

                    <div class="pills" style="margin-bottom:24px;">

                        <span class="pill pill-main">
                            {html.escape(str(category))}
                        </span>

                        <span class="pill">
                            📍 {html.escape(str(location))}
                        </span>

                    </div>

                    <div style="
                        color:#9ca3af;
                        font-size:12px;
                        margin-bottom:7px;
                    ">
                        Gefunden am
                    </div>

                    <div style="
                        color:#374151;
                        font-size:14px;
                        margin-bottom:24px;
                    ">
                        {html.escape(str(date))}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if description:

                st.markdown(
                    "### Beschreibung"
                )

                # Wichtig:
                # Benutzertext wird NICHT als HTML interpretiert.
                st.write(description)

            st.markdown("")

            if st.button(
                "Fundstück löschen",
                use_container_width=True,
            ):

                items = [
                    x for x in items
                    if x.get("id") != selected_id
                ]

                save_items(items)

                st.session_state.selected_item = None

                st.success("Fundstück gelöscht.")

                go("home")
