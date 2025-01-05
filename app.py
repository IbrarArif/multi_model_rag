import os
import uuid
import base64
import streamlit as st
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from PIL import Image

# Initialize Streamlit app
st.set_page_config(page_title='Dog Health Analyzer', layout='wide')
st.markdown('# Welcome to the Dog Health Analyzer!')

# Load PDF file
file_uploader = st.sidebar.file_uploader("Upload your PDF here...", type='pdf')
if file_uploader:
    uploaded_pdf = file_uploader
    
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=uploaded_pdf.name,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir="./images",
    )

    # Get text summaries and table summaries
    text_elements = []
    table_elements = []

    text_summaries = []
    table_summaries = []

    summary_prompt = """
    Summarize the following {element_type}:
    {element}
    """
    summary_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024),
        prompt=PromptTemplate.from_template(summary_prompt)
    )

    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = summary_chain.run({'element_type': 'text', 'element': e})
            text_summaries.append(summary)

        elif 'Table' in repr(e):
            table_elements.append(e.text)
            summary = summary_chain.run({'element_type': 'table', 'element': e})
            table_summaries.append(summary)

    # Get image summaries
    image_elements = []
    image_summaries = []

    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def summarize_image(encoded_image):
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images related to Dog's health."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the contents of this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        response = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024).invoke(prompt)
        return response.content

    for i in os.listdir("./images"):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join("./images", i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)

    # Create Documents and Vectorstore
    documents = []
    retriever = None

    for e, s in zip(text_elements, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'text',
                'original_content': e
            }
        )
        documents.append(doc)

    for e, s in zip(table_elements, table_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'table',
                'original_content': e
            }
        )
        documents.append(doc)

    for e, s in zip(image_elements, image_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': e
            }
        )
        documents.append(doc)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()

    # Answer questions
    while True:
        question = st.text_input("Type your question here...")
        if question:
            results = retriever.get_relevant_documents(question)
            if len(results) > 0:
                st.write(f"**Your Question:** {question}")
                st.write(f"**Our Answer:** \n\n{''.join([x.page_content for x in results])}")
            else:
                st.write(f"**Your Question:** {question}\nNo matching documents were found.")
else:
    st.markdown('Please upload a PDF first.')