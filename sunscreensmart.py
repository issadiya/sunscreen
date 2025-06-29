import streamlit as st, requests, pandas as pd, numpy as np, joblib, os, cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pydeck as pdk

# Train model
def train_model():
    df = pd.DataFrame({'skin_type': [1,2,3,4,5,6]*10, 'uv_index': [1,3,5,7,9,11]*10, 'spf': [15,15,30,30,50,50]*10})
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(df[['skin_type', 'uv_index']], df['spf'])
    joblib.dump(model, 'spf_model.pkl')

if not os.path.exists('spf_model.pkl'): train_model()
model = joblib.load('spf_model.pkl')

# IP-based location
def get_location_from_ip():
    try:
        d = requests.get('https://ipinfo.io/json').json()
        lat, lon = map(float, d['loc'].split(','))
        return lat, lon, d.get("city", ""), d.get("country", "")
    except: return None, None, "Unknown", "Unknown"

# Get UV Index
def get_uv(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=uv_index_max&timezone=auto"
        return round(requests.get(url).json()['daily']['uv_index_max'][0], 2)
    except: return None

# Image to skin type
def detect_skin_tone(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(cv2.resize(img, (200, 200)), cv2.COLOR_BGR2RGB)
    bright = np.mean(KMeans(n_clusters=1, n_init=10).fit(img_rgb.reshape(-1, 3)).cluster_centers_[0])
    return 1 if bright > 200 else 2 if bright > 170 else 3 if bright > 140 else 4 if bright > 110 else 5 if bright > 80 else 6

# Quiz to skin type
def calc_skin_type(q1, q2, q3, q4, q5): return min((q1+q2+q3+q4+q5 - 1)//6 + 1, 6)

# SPF recommendations
def recommend(spf, texture):
    db = {
        15: {"Dry": "Vanicream SPF 15", "Normal": "Lakme Sun Expert 15", "Oily": "Neutrogena Clear Face 15"},
        30: {"Dry": "La Roche-Posay SPF 30", "Normal": "Lotus SPF 30", "Oily": "Minimalist SPF 30"},
        50: {"Dry": "CeraVe AM SPF 50", "Normal": "Lakme Ultra Matte SPF 50", "Oily": "Neutrogena Dry Touch 50"}
    }
    return db.get(spf, {}).get(texture, "Generic SPF 50")

# UI
st.set_page_config("Smart Sunscreen Advisor")
st.title("🌞 Smart Sunscreen Advisor")
method = st.radio("Skin Type Detection", ["Manual", "Quiz", "Upload Photo"])
final_skin_type = None

if method == "Manual":
    final_skin_type = st.selectbox("Select Type", [1,2,3,4,5,6])
elif method == "Quiz":
    v = {"Very fair": 1, "Fair": 2, "Medium": 3, "Olive": 4, "Brown": 5, "Black": 6,
         "Always": 1, "Usually": 2, "Sometimes": 3, "Rarely": 4, "Never": 5,
         "Never": 1, "Slightly": 2, "Gradually": 3, "Quickly": 4, "Always": 5,
         "Red": 1, "Blonde": 2, "Brown": 3, "Dark Brown": 4, "Black": 5,
         "Blue": 1, "Green": 2, "Hazel": 3, "Brown": 4, "Black": 5}
    r = lambda q, o: v[st.radio(q, o)]
    final_skin_type = calc_skin_type(
        r("1. Skin Tone:", ["Very fair","Fair","Medium","Olive","Brown","Black"]),
        r("2. Sunburn Tendency:", ["Always","Usually","Sometimes","Rarely","Never"]),
        r("3. Tanning Ability:", ["Never","Slightly","Gradually","Quickly","Always"]),
        r("4. Hair Color:", ["Red","Blonde","Brown","Dark Brown","Black"]),
        r("5. Eye Color:", ["Blue","Green","Hazel","Brown","Black"]))
elif method == "Upload Photo":
    up = st.file_uploader("Upload skin photo", type=["jpg","jpeg","png"])
    if up: final_skin_type = detect_skin_tone(up)

texture = st.selectbox("Skin Texture", ["Dry", "Oily", "Normal"])
mode = st.radio("📍 Location Mode", ["Auto Detect", "Manual"])

if mode == "Auto Detect":
    lat, lon, city, country = get_location_from_ip()
    st.success(f"📍 {city}, {country}")
else:
    countries = {"India": (20.5,78.9), "Australia": (-25.2,133.7), "Brazil": (-14.2,-51.9), "Canada": (56.1,-106.3)}
    country = st.selectbox("Choose Country", list(countries.keys()))
    lat, lon = countries[country]

uv = get_uv(lat, lon)

if uv and final_skin_type:
    spf = model.predict([[final_skin_type, uv]])[0]
    product = recommend(spf, texture)
    st.success(f"SPF {spf} Recommended")
    st.info(f"UV: {uv}, Skin Type: {final_skin_type}, Texture: {texture}")
    st.markdown(f"**Top Product:** {product}")
    st.markdown("🔔 Reapply every 2 hours.")
elif uv and not final_skin_type:
    st.warning("Detect your skin type first.")
elif not uv:
    st.warning("UV data unavailable.")

# UV Map
st.subheader("🌍 UV Index Map")
df = pd.DataFrame({
    'Country': ['India', 'Australia', 'Brazil', 'Canada'],
    'lat': [20.5, -25.2, -14.2, 56.1],
    'lon': [78.9, 133.7, -51.9, -106.3],
    'uv_index': [uv if country == "India" else 9, 11, 10, 4]
})
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(latitude=20, longitude=80, zoom=1.5),
    layers=[
        pdk.Layer("TileLayer", data="http://c.tile.openstreetmap.org/{z}/{x}/{y}.png"),
        pdk.Layer("ScatterplotLayer", data=df,
                  get_position='[lon, lat]', get_color='[255, 100 - uv_index*10, 100]',
                  get_radius='uv_index * 10000', pickable=True)
    ],
    tooltip={"text": "UV Index in {Country}: {uv_index}"}
))

# Best time
st.subheader("⏰ Best Time To Go Out")
if uv >= 8:
    st.warning("Avoid 10 AM–4 PM. Best: Before 9:30 AM or After 4:30 PM")
elif uv >= 5:
    st.success("Best hours: Before 10 AM or After 4 PM")
else:
    st.success("Safe to go out anytime!")

st.markdown("<center><small>Built with ❤️</small></center>", unsafe_allow_html=True)
