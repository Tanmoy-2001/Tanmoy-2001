import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Vision Transformer Classifier", page_icon="🖼️")
st.title("Image Classification with Vision Transformer")
st.caption("Pretrained Hugging Face ViT model. First launch downloads model weights.")

@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
top_k = st.slider("Number of predictions", 1, 10, 5)
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Classify image", type="primary"):
        try:
            with st.spinner("Loading model and running inference..."):
                classifier = load_classifier()
                predictions = classifier(image, top_k=top_k)
            st.subheader("Predictions")
            for item in predictions:
                st.write(f"**{item['label']}** · {item['score']:.1%}")
                st.progress(float(item["score"]))
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
else:
    st.info("Upload an image to see model predictions.")
