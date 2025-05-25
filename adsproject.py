import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TITLE ---
st.title("NLP Dashboard for Farmer Queries")
st.markdown("This dashboard loads insights from a pre-cleaned dataset — no upload needed.")

# --- Load Data ---
try:
    qadatav2 = pd.read_excel("qadatav2.xlsx")
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("File 'qadatav2.xlsx' not found in the app folder.")
    st.stop()

# --- Preview of Data ---
st.subheader("Preview of the Dataset")
st.dataframe(qadatav2.head())

# --- Dataset Summary ---
st.subheader("Summary Statistics")
st.write("Total Queries:", len(qadatav2))
if 'Customer_id' in qadatav2.columns:
    st.write("Unique Farmers:", qadatav2['Customer_id'].nunique())
if 'County' in qadatav2.columns:
    st.write("Number of Counties Represented:", qadatav2['County'].nunique())

# --- Distribution by County ---
if 'County' in qadatav2.columns:
    st.subheader("Number of Questions by County")
    st.bar_chart(qadatav2['County'].value_counts())

# --- Sample Responses ---
if 'Response' in qadatav2.columns:
    st.subheader("Sample Responses")
    st.write(qadatav2['Response'].dropna().sample(5, random_state=1))

# --- Filter by County ---
if 'County' in qadatav2.columns:
    st.subheader("Filter Questions by County")
    county = st.selectbox("Select County", qadatav2['County'].dropna().unique())
    county_df = qadatav2[qadatav2['County'] == county]
    st.dataframe(county_df)

# --- Filter by County & About ---
if 'County' in qadatav2.columns and 'About' in qadatav2.columns and 'Category' in qadatav2.columns:
    st.subheader("Filter Queries by County and Category")

    about_filter = st.multiselect("Filter by About", qadatav2['About'].dropna().unique())
    filtered_data = qadatav2.copy()

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
if 'Description_Clean' in qadatav2.columns and 'Responses_Clean' in qadatav2.columns:
    st.title("Ask a Question")

    st.markdown("Type your question below to get the most relevant advisory response from our past data.")

    user_query = st.text_input("Enter your question:")

    if user_query:
        corpus = qadatav2['Description_Clean'].dropna().tolist()
        corpus.append(user_query)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        most_similar_index = cosine_similarities.argmax()

        matched_question = qadatav2.iloc[most_similar_index]['Description_Clean']
        matched_response = qadatav2.iloc[most_similar_index]['Responses_Clean']

        st.subheader("Most similar question from the data:")
        st.write(matched_question)

        st.subheader("Suggested response:")
        st.write(matched_response)
