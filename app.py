import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="Singapore Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Singapore Waste & Recycling Classifier")
st.caption("Upload a photo of any item to find out how to dispose of it correctly in Singapore.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a MobileNetV2 model trained on ~15,000 waste images "
        "to identify waste categories and provide NEA-aligned disposal instructions."
    )
    st.header("Waste Categories")
    categories = {
        "📦 Cardboard": "Blue recycling bin",
        "🍶 Glass": "Blue recycling bin",
        "🥫 Metal": "Blue recycling bin",
        "📄 Paper": "Blue recycling bin",
        "🧴 Plastic": "Blue recycling bin",
        "🗑️ Trash": "General waste bin"
    }
    for cat, bin_type in categories.items():
        st.write(f"**{cat}** → {bin_type}")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photo of a single item with good lighting"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing image..."):
        try:
            from predict import predict, get_model_and_classes
            model, classes = get_model_and_classes()
            result = predict(image, model, classes)

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            disposal = result["disposal"]
            top3 = result["top3"]

            st.divider()

            emoji = disposal.get("emoji", "♻️")
            st.markdown(f"## {emoji} {predicted_class.upper()}")

            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.progress(confidence / 100)

            with st.expander("See all predictions"):
                for cls, conf in top3:
                    st.write(f"**{cls}**: {conf:.1f}%")
                    st.progress(conf / 100)

            st.divider()

            st.markdown(f"### 🗂️ How to Dispose")
            st.info(f"**Bin:** {disposal.get('bin', 'General Waste Bin')}")
            st.write(disposal.get("instructions", "Please dispose responsibly."))

            st.markdown("### 💡 NEA Tip")
            st.success(disposal.get("nea_tip", ""))

            if confidence < 60:
                st.warning(
                    "⚠️ Confidence is below 60%. Try uploading a clearer photo "
                    "with better lighting and a single item in frame."
                )

        except Exception as e:
            st.error(f"Error during classification: {e}")

st.divider()
st.caption("Disposal guidelines based on NEA Singapore recycling standards. Learn more at nea.gov.sg")