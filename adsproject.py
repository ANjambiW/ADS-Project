import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TITLE ---
st.title("Question and Answer Dashboard")
#st.markdown("This dashboard loads insights from a pre-cleaned dataset — no upload needed.")

# --- Load Data ---
try:
    ishamba_plant_lstock_share = pd.read_excel("ishamba_plant_lstock_share.xlsx")
    st.success("Dataset is loaded.")
except FileNotFoundError:
    st.error("File 'ishamba_plant_lstock_share.xlsx' not found in the app folder.")
    st.stop()

# --- Data ---
st.subheader("View Data")
st.dataframe(ishamba_plant_lstock_share.head())

# --- Dataset Summary ---
st.subheader("Quick Statistics")
st.write("Total Queries:", len(ishamba_plant_lstock_share))
if 'Customer_id' in ishamba_plant_lstock_share.columns:
    st.write("Unique Farmers:", ishamba_plant_lstock_share['Customer_id'].nunique())
if 'County' in ishamba_plant_lstock_share.columns:
    st.write("Number of Counties Represented:", ishamba_plant_lstock_share['County'].nunique())

# --- Distribution by County ---
if 'County' in ishamba_plant_lstock_share.columns:
    st.subheader("Number of Questions by County")
    st.bar_chart(ishamba_plant_lstock_share['County'].value_counts())

# --- Sample Responses ---
if 'Response' in ishamba_plant_lstock_share.columns:
    st.subheader("Sample Responses")
    st.write(ishamba_plant_lstock_share['Response'].dropna().sample(5, random_state=1))

# --- Filter by County ---
if 'County' in ishamba_plant_lstock_share.columns:
    st.subheader("Filter Questions by County")
    county = st.selectbox("Select County", ishamba_plant_lstock_share['County'].dropna().unique())
    county_df = ishamba_plant_lstock_share[ishamba_plant_lstock_share['County'] == county]
    st.dataframe(county_df)

# --- Filter by County & About ---
if 'County' in ishamba_plant_lstock_share.columns and 'About' in ishamba_plant_lstock_share.columns and 'Category' in ishamba_plant_lstock_share.columns:
    st.subheader("Filter Questions by About and County")

    about_filter = st.multiselect("Filter by About", ishamba_plant_lstock_share['About'].dropna().unique())
    filtered_data = ishamba_plant_lstock_share.copy()

    if about_filter:
        filtered_data = filtered_data[filtered_data['About'].isin(about_filter)]

    county_filter = st.multiselect("Select County to Display", filtered_data['County'].dropna().unique())
    if county_filter:
        filtered_data = filtered_data[filtered_data['County'].isin(county_filter)]

    if not filtered_data.empty:
        grouped = filtered_data.groupby(['County', 'Category']).size().reset_index(name='Query Count')
        pivot_table = grouped.pivot(index='County', columns='Category', values='Query Count').fillna(0)
        st.write("Number of Questions per Category in Selected County/Counties")
        st.dataframe(pivot_table)
        st.bar_chart(pivot_table)
    else:
        st.info("No data available for the selected filters.")

# --- NLP Question Responder ---
if 'Description_Clean' in ishamba_plant_lstock_share.columns and 'Responses_Clean' in ishamba_plant_lstock_share.columns:
    st.title("Question Responder")

    st.markdown("Type your question below to get the most similar response from past data.")

    user_query = st.text_input("Ask your question:")

    if user_query:
        corpus = ishamba_plant_lstock_share['Description_Clean'].dropna().tolist()
        corpus.append(user_query)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        most_similar_index = cosine_similarities.argmax()

        matched_question = ishamba_plant_lstock_share.iloc[most_similar_index]['Description_Clean']
        matched_response = ishamba_plant_lstock_share.iloc[most_similar_index]['Responses_Clean']

        st.subheader("Most similar question from the data:")
        st.write(matched_question)

        st.subheader("Suggested response:")
        st.write(matched_response)
