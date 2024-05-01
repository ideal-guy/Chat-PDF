import streamlit as st
from dotenv import load_dotenv
import pickle
import joblib
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACE_API_KEY")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - LLAMA2 7B model (OpenSource) from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    - FAISS for vector storage
    - OpenAI for embeddings
    - PyPDF2 for PDF reading
                

    ''')
    add_vertical_space(5)
    st.write('Made by [Adarsh Pratap Singh](https://youtube.com/@engineerprompt)')



def main():
    st.header("Chat with pdf")

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    #st.write(pdf)
    if pdf is not None:
        #pdf reader
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     # st.write('Embeddings Loaded from the Disk')s
        # else:
        #     OPENAI_API_KEYS = os.getenv("OPENAI_API_KEYS")
        #     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEYS)
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)
        OPENAI_API_KEYS = os.getenv("OPENAI_API_KEYS")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEYS)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            # docs = VectorStore.similarity_search(query=query, k=3)
            
            llm = llm=HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", 
                                        model_kwargs={"temperature":0.5, 
                                                      "max_length":64})
            

            prompt_template = """Given the following context and a question, generate an answer based on this context only.

                CONTEXT: {context}

                QUESTION: {question}
                
                ANSWER:
                """

            PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                    )
            
            chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type = "stuff",
                    retriever=VectorStore.as_retriever(search_kwargs={"k":3}),
                    input_key="query",
                    chain_type_kwargs={"prompt": PROMPT}
            )

            with get_openai_callback() as cb:
                response = chain({"query":query})["result"]
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()
