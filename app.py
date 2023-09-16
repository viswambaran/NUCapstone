import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


#Removing Stopwords from the headlines 
def process_headline(headline):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenize
    tokens = word_tokenize(headline)

    # Remove stopwords and lemmatize
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]

    return " ".join(clean_tokens)

# Initialize zero-shot classification pipeline with the specified model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")





financial_news_taxonomy = {
    "Market Movements": ["Stock Trends", "Commodity Prices", "Currency Rates", "Bond Yields"],
    "Company News": ["Earnings", "Product Launches", "M&A", "Legal & Regulatory", "Leadership"],
    "Economic Data": ["Employment", "Inflation", "GDP", "Consumer Sentiment"],
    "Policy & Regulation": ["Central Bank Actions", "Trade & Tax Policies", "Environmental Rules"],
    "Global Events": ["Geopolitical Issues", "Elections", "Natural Disasters", "Health Crises"],
    "Sector Highlights": ["Tech Developments", "Banking News", "Energy Updates", "Healthcare Innovations", "Real Estate Trends"],
    "Investment Insights": ["Fund Activities", "Asset Trends", "Investment Strategies"]
}



print(financial_news_taxonomy)

financial_news_terms = [item for sublist in financial_news_taxonomy.values() for item in sublist]



def classify_text(text):
    result = classifier(text, financial_news_terms)
    return result['labels'][0], result['scores'][0]

# Streamlit app

st.set_page_config(
    page_title="Zero-shot Classifier App",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Zero-shot Classifier for Financial News.")
st.write("Classify financial news headlines using zero-shot classification.")

# Text area
user_input = st.text_area("Enter your headline here:", "")

# File upload
uploaded_file = st.file_uploader("Or upload a CSV file", type=["csv"])

column_name = None
df = None

if uploaded_file:
    # Load the CSV to check columns
    df = pd.read_csv(uploaded_file)
    
    # Give user choice of which column to classify
    column_name = st.selectbox("Choose a column to classify:", df.columns)


# Button to run classifier
if st.button("Classify"):
    progress_bar = st.progress(0)

    if user_input:
        label, score = classify_text(user_input)
        st.write(f"Predicted Label: {label}")
        st.write(f"Confidence Score: {score:.4f}")
        progress_bar.progress(1.0)  # Complete the progress bar when done

    elif uploaded_file and column_name:
        if column_name in df.columns:
            total_len = len(df)
            for i, row in enumerate(df[column_name]):
                df.at[i, 'Predicted Label'], df.at[i, 'Confidence Score'] = classify_text(row)
                # Update the progress bar (as a fraction between 0.0 and 1.0)
                progress_bar.progress((i + 1) / total_len)
            st.write(df)
            
            # EDA Charts
            
            # Bar chart for label distribution
            st.subheader('Predicted Label Distribution')
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df['Predicted Label'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
            plt.xticks(rotation=45)
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution of Predicted Labels')
            st.pyplot(fig1)
            
            # Histogram for confidence scores
            st.subheader('Confidence Score Distribution')
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df['Confidence Score'].hist(bins=30, ax=ax2, color='salmon')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Confidence Scores')
            st.pyplot(fig2)
            
            # Word cloud for headlines
            st.subheader('Word Cloud from Headlines')
            text = ' '.join(df[column_name].apply(process_headline))  # Process headlines
            wordcloud = WordCloud(
                background_color='white',
                colormap='viridis',
                width=800,
                height=400,
                max_words=200,
            ).generate(text)
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)
            
        else:
            st.write(f"Column '{column_name}' not found in the uploaded CSV.")

    progress_bar.empty()  # Reset the progress bar after completion