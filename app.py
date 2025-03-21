from langchain_community.document_loaders import PyPDFLoader# type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings# type: ignore
from langchain.chains import ConversationalRetrievalChain# type: ignore
from langchain.memory import ConversationBufferMemory# type: ignore
from langchain_community.vectorstores import FAISS# type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter# type: ignore
from langchain_core.prompts import PromptTemplate# type: ignore
from langchain_community.chat_message_histories import StreamlitChatMessageHistory # type: ignore
import streamlit as st
import os
from tempfile import NamedTemporaryFile
import yaml
 
# Load API Key
OPENAI_API_KEY = yaml.safe_load(open('./credentials.yml'))['openai']
 
# 1.0 INITIALIZE SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
 
# 2.0 PROCESS PDF FUNCTION
def process_pdf(file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
   
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
       
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
       
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(splits, embeddings)
       
        return vectorstore.as_retriever()
    finally:
        os.remove(file_path)
 
# 3.0 CHAIN WITH MEMORY
def get_conversation_chain(retriever):
    template = """Você é um assistente especialista em análise de documentos. Use o contexto para responder.
    Contexto:
    {context}
 
    Histórico da Conversa:
    {chat_history}
 
    Pergunta: {question}. Sempre responda apenas com base no contexto, caso não ache nenhuma resposta contida dentro do contexto, apenas responda "Nenhuma resposta baseada no documento encontrada!."
    Resposta útil:"""
   
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
 
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
 
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
 
# 4.0 STREAMLIT INTERFACE
st.title("Chat com Documentos - H&P")
st.subheader("Envie um PDF para conversar com o documento")
import streamlit as st
 
st.markdown(
    '### APLICATIVO EM FASE DE TESTES - <span style="color:red">USAR APENAS DADOS PÚBLICOS</span>',
    unsafe_allow_html=True
)
 
with st.expander("Boas práticas de prompt"):
    st.markdown("### Para escrever um bom prompt alguns elementos são necessários, são eles:")
    st.markdown("**Contexto:** Forneça informações de fundo ou contexto para orientar a conversa.")
    st.markdown("**Instrução Clara:** Uma tarefa ou pergunta específica que você quer que o modelo execute.")
    st.markdown("**Exemplos ou Casos de Uso:** Inclua exemplos ou cenários para ilustrar o que você espera.")
    st.markdown("**Restrições ou Limitações:** Defina limites para a resposta, como tamanho, tom ou escopo.")
    st.markdown("**Uso de Palavras-Chave:** Inclua palavras-chave relevantes para orientar o modelo. Exemplo: Discuta os impactos da mineração de ouro na **biodiversidade**, com foco em **desmatamento** e **contaminação da água**.")
    st.markdown("**Formato de Resposta:** Especifique como você quer que a resposta seja estruturada.")
   
# Upload PDF
uploaded_file = st.file_uploader(
    label="Escolha um arquivo PDF",
    type="pdf"
)
 
# Processar PDF
if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processando PDF..."):
        try:
            st.session_state.retriever = process_pdf(uploaded_file)
            st.session_state.qa_chain = get_conversation_chain(st.session_state.retriever)
            st.session_state.pdf_processed = True
            st.success("PDF carregado! Faça suas perguntas.")
        except Exception as e:
            st.error(f"Erro: {str(e)}")
 
# Exibir histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# Input de chat
if prompt := st.chat_input("Faça sua pergunta sobre o documento"):
    if not st.session_state.pdf_processed:
        st.warning("Envie um PDF primeiro.")
        st.stop()
   
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
   
    # Gerar resposta
    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            try:
                response = st.session_state.qa_chain({"question": prompt})
                answer = response["answer"]
               
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
           
            except Exception as e:
                st.error(f"Erro: {str(e)}")
 
 
 