import streamlit as st
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title="ShopEase Sentiment Analyser",
    page_icon="🛍️",
    layout="centered"
)

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="AishatOlisa/shopease-sentiment-distilbert",
        return_all_scores=True
    )

# Label mapping
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Colour mapping
colour_map = {
    "Negative": "#e74c3c",
    "Neutral": "#f39c12",
    "Positive": "#2980b9"
}

# App header
st.title("🛍️ ShopEase Europe")
st.subheader("Customer Review Sentiment Analyser")
st.markdown("Paste a customer review below to classify its sentiment using a fine-tuned DistilBERT model trained on genuine Amazon customer reviews.")
st.markdown("---")

# Input
review_text = st.text_area(
    "Customer Review",
    placeholder="e.g. The delivery was late and the driver left the package in the rain...",
    height=150
)

# Analyse button
if st.button("Analyse Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review before analysing.")
    else:
        with st.spinner("Analysing..."):
            model = load_model()
            results = model(review_text[:512])

        scores = results[0] if isinstance(results[0], list) else results
        top = max(scores, key=lambda x: x["score"])
        sentiment = label_map.get(top["label"], top["label"])
        confidence = top["score"] * 100
        colour = colour_map[sentiment]

        st.markdown("---")
        st.markdown(f"### Predicted Sentiment")
        st.markdown(
            f"<h2 style='color:{colour}'>{sentiment}</h2>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("#### Score Breakdown")
        for s in sorted(scores, key=lambda x: x["score"], reverse=True):
            label = label_map.get(s["label"], s["label"])
            pct = s["score"] * 100
            st.progress(s["score"], text=f"{label}: {pct:.1f}%")

st.markdown("""
### Welcome to the ShopEase Europe Sentiment Classifier

Enter any customer review below to predict whether its sentiment is **Positive**, **Negative**, or **Neutral** using a **DistilBERT** model fine-tuned on **21,055 Amazon customer reviews**.

The model returns both the predicted sentiment and its confidence score.

#### Try these example reviews

**🔴 Negative**
> *"The driver marked my order as delivered, but nothing arrived. I've called customer service four times and still haven't received a resolution."*

**🟢 Positive**
> *"Fast delivery, excellent packaging, and outstanding customer support. I've been shopping here for years and have always had a great experience."*

**🟡 Neutral**
> *"The product works as described. Nothing particularly impressive or disappointing about it."*
""")

st.markdown("---")