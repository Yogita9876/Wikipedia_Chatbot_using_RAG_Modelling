import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import requests
from langchain_core.documents import Document

load_dotenv()






## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Wikipedia based Q&A chatbot using RAG modelling")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the Wikipedia context.
Please augment your response with the most accurate and latest information you gather 
from wikipedia in response to the question and supplement that with your own database
<context>
{context}
</context>
Questions:{input}

Answer:
"""
)

# Wikipedia API endpoint
WIKIPEDIA_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def search_wikipedia(query, limit=5):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": limit
    }
    response = requests.get(WIKIPEDIA_API_ENDPOINT, params=params)
    return response.json()["query"]["search"]

def get_wikipedia_content(page_id):
    params = {
        "action": "query",
        "format": "json",
        "pageids": page_id,
        "prop": "extracts",
        "explaintext": True
    }
    response = requests.get(WIKIPEDIA_API_ENDPOINT, params=params)
    return response.json()["query"]["pages"][str(page_id)]["extract"]

def create_wikipedia_knowledge_base(query):
    try:
        search_results = search_wikipedia(query)
        
        if not search_results:
            raise ValueError("No search results found for the given query.")
        
        documents = []
        for result in search_results:
            content = get_wikipedia_content(result['pageid'])
            doc = Document(page_content=content, metadata={"title": result['title'], "pageid": result['pageid']})
            documents.append(doc)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(split_documents, embeddings)
        
        return vector_store
    
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

st.sidebar.title("Wikipedia Q&A Chatbot")
query = st.sidebar.text_input("Enter your question:")


if query:
    with st.spinner("Fetching relevant Wikipedia articles..."):
        vector_store = create_wikipedia_knowledge_base(query)
        
    if vector_store is None:
        st.error("Failed to create knowledge base. No relevant Wikipedia articles found. Please try a different query.")
    else:
        st.sidebar.success("Knowledge base created from Wikipedia articles")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating answer..."):
            try:
                response = retrieval_chain.invoke({'input': query})
                st.write("Answer:", response['answer'])

                with st.expander("Relevant Wikipedia Excerpts"):
                    if "context" in response and response["context"]:
                        for i, doc in enumerate(response["context"]):
                            st.write(f"Excerpt {i+1}:")
                            st.write(doc.page_content)
                            st.write(f"Source: {doc.metadata['title']}")
                            st.write("---")
                    else:
                        st.write("No relevant excerpts found.")
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.write("This chatbot uses Wikipedia as its knowledge source, which might not be up to date.")






