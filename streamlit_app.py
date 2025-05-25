%%writefile app.py
import streamlit as st
import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.title("Cold Email Generator")

url = st.text_input("Enter Job Posting URL")
submit_button = st.button("Generate Email")

if submit_button and url:
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key="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")
    loader = WebBaseLoader(url)
    data = loader.load().pop().page_content
    #st.success("Job posting loaded!")

    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
    )
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke({"data": data})
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    #st.success("Job details extracted!")
    df=pd.read_csv("/content/my_portfolio.csv")
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")
    if not collection.count():
      for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])
    job=json_res
    links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])


    prompt_email = PromptTemplate.from_template(
             """
         ### JOB DESCRIPTION:
    {job_description}
  Your job is to write a cold email to the client regarding the job mentioned above describing your capability
  to fulfill their needs.
  Also add the most relevant ones from the following links to showcase portfolio: {link_list}
  Remember you are Sidra, ML Engineer at XYZ company.
  Avoid generic introductions—focus on **value, relevance, and engagement**.
  ### EMAIL (NO PREAMBLE):
    """
    )
    chain_email = prompt_email | llm
    res = chain_email.invoke({"job_description": str(job), "link_list": links})
    st.text_area("Generated Email", res.content, height=300)

    # Sidebar with instructions
st.sidebar.markdown("## Guide")
st.sidebar.info(
    "It allows users to input the URL of a company's careers page. "
    "The tool then extracts job listings from that page and generates personalized cold emails. "
    "These emails include relevant portfolio links sourced from a vector database, based on the specific job descriptions."
)
