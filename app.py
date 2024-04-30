import os
from PIL import Image, ImageDraw,ImageEnhance,PngImagePlugin
import numpy as np
import cv2
import streamlit as st
from datetime import datetime
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import google.generativeai as genai

# setting up all the relevant keys
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
genai.configure(api_key='AIzaSyDMAqL5ga6BQzk_UJwmahsFuSNz4Awm-5c')
model_gem_pro = genai.GenerativeModel('gemini-pro')
model_gem_pro_vis = genai.GenerativeModel('gemini-pro-vision')
lang = 'en'
fitz_doc = []
img_path = "C://Users/mitta/OneDrive/Desktop/telegram_bot/hackathon_chatbot/img.png"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature":0.55, "max_length":10}
)

# defining the useful functons
def caption_image(array_np):
    model_gem_pro = genai.GenerativeModel('gemini-pro')
    model_gem_pro_vis = genai.GenerativeModel('gemini-pro-vision')
    response_caption = model_gem_pro_vis.generate_content(["Describe the photo/image in the file, also describe the text in the file",Image.fromarray(array_np)])
    try:
        response_caption.resolve()
        return response_caption.text
    except:
        pass
    
def create_text_image(page):
    mat = fitz.Matrix(2, 2)
    page_image = page.get_pixmap(matrix=mat)
    th = int(page_image.height * 0.05)
    chth = page_image.height * 0.01
    cwth = page_image.width * 0.01
    cath = chth * cwth * 100
    text_image = Image.new('RGB', (page_image.width, page_image.height), (0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    blocks = page.get_text("blocks")
    for block in blocks:
        x, y, w, h, text, _, _ = block
        draw.rectangle([2*x, 2*y, 2*w, 2*h], fill="white")
    maskArray = np.array(text_image)
    image = cv2.cvtColor(maskArray, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))
    diagArray = np.maximum(maskArray, np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.h, page_image.w, page_image.n))
    diagGray = cv2.cvtColor(diagArray, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diagGray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < cwth or h < chth or w*h < cath:
            continue
        y -= th//2
        h += th
        temp = []
        for block in rectangles:
            x1 = int(block[0]) - th//2
            y1 = int(block[1]) - th//2
            w1 = int(block[2]) + th
            h1 = int(block[3]) + th
            if ((y1 < y and y1+h1 > y) or (y < y1 and y+h > y1)) and (((x1 < x and x1+w1 > x) or (x < x1 and x+w > x1))):
                temp.append((x1, y1, w1, h1))
        temp.append((x, y, w, h))
        selected.append(temp)
    maskArray[0:page_image.height, 0:page_image.width] = 255
    for sel in selected:
        for i, temp in enumerate(sel):
            x, y, w, h = temp
            maskArray[y: y + h, x: x + w] = 0
    diagArray = np.maximum(maskArray, np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.h, page_image.w, page_image.n))
    diagram = Image.fromarray(diagArray)
    ddraw = ImageDraw.Draw(diagram)
    images = []
    for i, sel in enumerate(selected):
        if len(sel) == 0:
            continue
        imx, imy, imw, imh = sel[0]
        for temp in sel:
            x, y, w, h = temp
            imx = min(imx, x)
            imy = min(imy, y)
        for i, temp in enumerate(sel):
            x, y, w, h = temp
            imw = max(imw + imx, w + x) - imx
            imh = max(imh + imy, h + y) - imy
        new = diagArray[imy:imy + imh, imx:imx + imw]
        images.append(new)
    return images

def image_extractor(page):
    images = create_text_image(page)
    return images


def relevant_images(relevant_page_no,query):
    relevant_images = []
    image_docs = []
    for i in relevant_page_no:
        page = fitz_doc[i]
        image_r = image_extractor(page)
        relevant_images.extend(image_r)
    for i in range(len(relevant_images)):
        image_doc = Document(page_content = caption_image(relevant_images[i]),metadata={ 'index' : i})
        image_docs.append(image_doc)

    if(len(relevant_images)==0):
        return []

    faiss_index_images = FAISS.from_documents(image_docs,instructor_embeddings)
    qa_chain_images = ConversationalRetrievalChain.from_llm(
        llm,
        faiss_index_images.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,)
    result_images = qa_chain_images({'question': query,'chat_history':""})

    to_return = []
    for i in range(len(result_images['source_documents'])):
        to_return.append(Image.fromarray(relevant_images[result_images['source_documents'][i].metadata['index']]))

    return to_return

def translate_to_en(raw_input):
    model_gem_pro = genai.GenerativeModel('gemini-pro')
    model_gem_pro_vis = genai.GenerativeModel('gemini-pro-vision')
    try:
        prompt = "Convert this to english - "+raw_input
        response = model_gem_pro.generate_content(prompt)
        return response.text
    except:
        pass

def translate_to_ja(raw_input):
    model_gem_pro = genai.GenerativeModel('gemini-pro')
    model_gem_pro_vis = genai.GenerativeModel('gemini-pro-vision')
    try:
        prompt = "Convert this to japanese - "+raw_input
        response = model_gem_pro.generate_content(prompt)
        return response.text
    except:
        pass

# Load PDF and process text
def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    docs = []
    split_text = ""
    for i in range(len(pdf_reader.pages)):
        if lang == 'en':
            split_text = text_splitter.split_text(pdf_reader.pages[i].extract_text())
        else:
            split_text = text_splitter.split_text(translate_to_en(pdf_reader.pages[i].extract_text()))
        for j in split_text:
            doc = Document(page_content=j, metadata={"index": i + 1})
            docs.append(doc)

    faiss_index = FAISS.from_documents(docs, instructor_embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        faiss_index.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
    )

    return qa_chain, pdf_reader

# Function to generate a response based on user input
def generate_response(input_text, qa_chain, pdf_reader):
    chat_history = st.session_state.chat_history
    result = qa_chain({'question': input_text, 'chat_history': chat_history})
    page_no = []
    for i in result['source_documents']:
        page_no.append(i.metadata['index'])

    response = '''Most Relevant pages:\n'''
    for i in page_no[:3]:
        response += f'{i} '
        block_texts = fitz_doc[i-1].get_text("blocks")
        for j in block_texts[::-1]:
            j_4_replaced = j[4].replace('-', '')
            if j[4].count('-') == 1 and (j_4_replaced).strip().isnumeric():
                response += f"{j[4]}\n"
                break

    response_print = response
    response = ""

    response += result['answer'][result['answer'].rfind('Helpful Answer:'):]
    response_to_feed = result['answer'][result['answer'].rfind('Helpful Answer:'):]
    chat_history.append((input_text, response_to_feed))
    a = relevant_images(page_no[:2],input_text)
    if len(a):
        a[0].save('img'+str(0)+'.png')
    new_res = model_gem_pro.generate_content(response+"- make this text look a little conversational,remember only a little")
    try:
        new_res.resolve()
        try:
            opinion = model_gem_pro_vis.generate_content([str("Only answer as either yes or no, Is the image provided is relevant to the query appropiately - "+input_text),a[0]])
            opinion.resolve()
            if opinion.text == 'no':
                img_page = fitz_doc[page_no[0]-1].get_pixmap()
                img_page = np.frombuffer(img_page.samples, dtype=np.uint8).reshape(img_page.h, img_page.w,img_page.n)
                img_page.save('img0.png')
        except:
            
            img_page = fitz_doc[page_no[0]-1].get_pixmap()
            img_page = np.frombuffer(img_page.samples, dtype=np.uint8).reshape(img_page.h, img_page.w,img_page.n)
            Image.fromarray(img_page).save('img0.png')
            
        return response_print + new_res.text
    except:
        return "No relevant information found"

    


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_html' not in st.session_state:
    st.session_state.chat_html = ""

if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

if 'pdf_reader' not in st.session_state:
    st.session_state.pdf_reader = None

if 'prev_resp' not in st.session_state:
    st.session_state.prev_html = ""

if 'prev_resp' not in st.session_state:
    st.session_state.prev_history = []

if 'most_rel_image' not in st.session_state:
    st.session_state.most_rel_image = None

if 'existing_docs' not in st.session_state:
    st.session_state.existing_docs = []



# Streamlit UI
def main():
    global i
    global lang
    global fitz_doc
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>YAMAHA X IIT MANDI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin:5px'>DOCBOT</h3>", unsafe_allow_html=True)
    with st.container():
        def store_image(loc):
            st.image(loc,width=600)

    # Define a container to hold the entire chat interface
        with st.container():
            # Upload file section
            st.subheader("Upload an Image or PDF:")
            uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
            fitz_doc = fitz.open(uploaded_file)
            if(uploaded_file is not None and st.session_state.file_processed==False):
                st.session_state.qa_chain, st.session_state.pdf_reader = process_pdf(uploaded_file)
                st.session_state.file_processed = True
            st.divider()
            language_options = ['English', 'Japanese']
            
            selected_language = st.selectbox('Select Language:', language_options)
            if selected_language == 'English':
                st.write('English language is selected.')
                lang = 'ENGLISH'
            elif selected_language == 'Japanese':
                st.write('Japanese language is selected.')
                lang = 'Japanese'
            

            with st.container():
                chat_container = st.empty()
                chat_container.markdown(st.session_state.chat_html, unsafe_allow_html=True)

                input_text = ""
                if "my_text" not in st.session_state:
                    st.session_state.my_text = ""
                def send():
                    st.session_state.my_text = st.session_state.input
                    st.session_state.input = ""
                st.text_input(label="input", placeholder="Enter your Query here:" ,disabled=False, label_visibility="collapsed",key="input", on_change=send)
                input_text = st.session_state.my_text
                col1, col2 = st.columns([1,15])
                if col2.button("New Prompt"):
                    st.session_state.prev_html = st.session_state.chat_html
                    st.session_state.prev_history = st.session_state.chat_history
                    st.session_state.chat_html = ""
                    st.session_state.chat_history = []
                    chat_container.empty()
                if col1.button("Send"):
                    if input_text:
                        user_message = f'<div class="fadeout" style="background-color:#475569; color:white;padding:10px; border-radius:25px; margin:10px 0 10px auto; text-align:left; width:55%;">' \
                                        f'<strong style="color:#FF0000;">You:</strong><p>{input_text}</p>' \
                                        f'<span style="color:gray; font-size:12px;">Sent at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>' \
                                        f'</div>'
                        st.session_state.chat_html += "</div>"+ user_message
                        chat_container.markdown(st.session_state.chat_html, unsafe_allow_html=True)
                        response = str(generate_response(input_text, st.session_state.qa_chain, st.session_state.pdf_reader))
                        
                        bot_response = f'<div class="fadeout" style="background-color:#475569;color:white; padding:10px; border-radius:25px; margin:10px 0 10px 0; text-align:left; width:55%;">' \
                                        f'<strong style="color:#cffafe;">ChatBot:</strong><p>{response}</p>' \
                                        f'<span style="color:gray; font-size:12px;">Sent at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>' \
                                        f'</div>'
                        st.session_state.chat_html += bot_response
                        final = "<p>Chatsbots response</p>"
                        chat_container.markdown(st.session_state.chat_html+final, unsafe_allow_html=True)
                        store_image("img0.png")

if __name__ == "__main__":

    main()
