import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Text Data")

# File uploader
uploaded_file = st.file_uploader("Upload cleaned dataset (.csv or .xlsx)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write(df.head())

# ----------------------------------

# Title
st.title("Descriptive Statistics")

# Description
st.write("""
#Far.
""")

# Check if dataframe exists
if 'df' in locals() or 'df' in globals():

    # Quick statistics
    st.subheader("Summary Statistics")
    st.write("Total Queries:", len(df))
    st.write("Unique Farmers:", df['Customer_id'].nunique())
    st.write("Number of Counties Represented:", df['County'].nunique())

    # Visualize Queries per County
    st.subheader("Number of Questions by County")
    st.bar_chart(df['County'].value_counts())

    # Sample Farmer Questions
    # st.subheader("Sample Farmer Questions")
    # st.write(df['Description_Clean'].sample(10))

    # Filter by County
    st.subheader("Filter Questions by County")
    county = st.selectbox("Select County", df['County'].unique())
    filtered_df = df[df['County'] == county]
    st.dataframe(filtered_df)

else:
    st.error("Data not found. Please upload a cleaned dataset to continue.")

# #############

st.subheader("Filter Queries by County and Category")

# Filter by About
about_filter = st.multiselect("Filter by About", df['About'].dropna().unique())

filtered_data = df.copy()

if about_filter:
    filtered_data = filtered_data[filtered_data['About'].isin(about_filter)]

# Filter by County (multi-select so user can choose which counties to view → clean and neat)
county_filter = st.multiselect("Select County to Display", filtered_data['County'].dropna().unique())

if county_filter:
    filtered_data = filtered_data[filtered_data['County'].isin(county_filter)]

if not filtered_data.empty:
    # Group data by County and Category
    grouped = filtered_data.groupby(['County', 'Category']).size().reset_index(name='Query Count')

    # Pivot for easier plotting
    pivot_table = grouped.pivot(index='County', columns='Category', values='Query Count').fillna(0)

    st.write("Number of Questions per Category in Selected County/Counties")
    st.dataframe(pivot_table)

    # Combine into one plot
    st.bar_chart(pivot_table)

else:
    st.info("No data available for the selected filters.")

# ######################

# Load cleaned dataset
df = pd.read_excel("qadatav2.xlsx")

st.title("Responder")

st.write("""
Type your question below and get the most relevant advisory response from our past data.
""")

# Input user query
user_query = st.text_input("Enter your question:")

if user_query:

    # Vectorize the questions (Description_Clean + user query)
    corpus = df['Description_Clean'].dropna().tolist()
    corpus.append(user_query)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    # Compute similarity between user query and all existing questions
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    # Get the index of the most similar question
    most_similar_index = cosine_similarities.argmax()

    # Get the matching question and response
    matched_question = df.iloc[most_similar_index]['Description_Clean']
    matched_response = df.iloc[most_similar_index]['Responses_Clean']

    st.subheader("Most similar question from the data:")
    st.write(matched_question)

    st.subheader("Response:")
    st.write(matched_response)

