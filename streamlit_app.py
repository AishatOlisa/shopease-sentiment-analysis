import streamlit as st
from transformers import pipeline

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
st.markdown("""
This tool classifies customer reviews as **Negative**, **Positive**, or **Neutral** 
using a DistilBERT model fine-tuned on 21,055 genuine Amazon customer reviews.
The model returns a predicted sentiment label and confidence score across all three classes.
""")

st.markdown("#### Example reviews to try")
st.markdown("""
**🔴 Negative**
> *"The driver marked my order as delivered but nothing arrived. I've called customer service four times and still have no resolution."*

**🟢 Positive**
> *"Fast delivery, excellent packaging, and outstanding customer support. Been shopping here for years and never disappointed."*

**🟡 Neutral**
> *"The product works as described. Nothing particularly impressive or disappointing about it."*
""")

st.markdown("---")

# Input
review_text = st.text_area(
    "Paste a customer review below",
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
        st.markdown("### Predicted Sentiment")
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

st.markdown("---")
st.caption("Model: AishatOlisa/shopease-sentiment-distilbert | GitHub: github.com/AishatOlisa/shopease-sentiment-analysis")