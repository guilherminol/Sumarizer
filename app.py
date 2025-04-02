from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os
from tempfile import NamedTemporaryFile
import msal
import requests

# Carregar configurações do st.secrets
OPENAI_API_KEY = st.secrets.openai.api_key
CLIENT_ID = st.secrets.microsoft.client_id
TENANT_ID = st.secrets.microsoft.tenant_id
CLIENT_SECRET = st.secrets.microsoft.client_secret
REDIRECT_URI = st.secrets.microsoft.redirect_uri
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]

# 1.0 INICIALIZAR ESTADO DA SESSÃO
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
if "auth" not in st.session_state:
    st.session_state.auth = {
        "authenticated": False,
        "user": None,
        "email": None,
        "token": None
    }

# 2.0 FUNÇÕES DE AUTENTICAÇÃO
def get_aad_token():
    cache = msal.SerializableTokenCache()
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
        token_cache=cache
    )
    
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPE, account=accounts[0])
        return result
    return None

def validate_company_email(email):
    return email.endswith("@hep.solutions")

# 3.0 INTERFACE DE LOGIN
if not st.session_state.auth["authenticated"]:
    st.title("Acesso Corporativo")
    st.markdown("Por favor, faça login com sua conta corporativa")

    if st.button("Entrar com Microsoft"):
        app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET
        )
        auth_url = app.get_authorization_request_url(
            SCOPE,
            redirect_uri=REDIRECT_URI
        )
        # Redirecionamento automático via JavaScript
        js = f"window.location.href = '{auth_url}'"
        st.write(f'<script>{js}</script>', unsafe_allow_html=True)

    query_params = st.query_params
    if "code" in query_params:
        with st.spinner("Autenticando..."):
            try:
                app = msal.ConfidentialClientApplication(
                    CLIENT_ID,
                    authority=AUTHORITY,
                    client_credential=CLIENT_SECRET
                )
                
                result = app.acquire_token_by_authorization_code(
                    query_params["code"],
                    scopes=SCOPE,
                    redirect_uri=REDIRECT_URI
                )

                if "access_token" in result:
                    user_info = requests.get(
                        "https://graph.microsoft.com/v1.0/me",
                        headers={"Authorization": f"Bearer {result['access_token']}"}
                    ).json()

                    if validate_company_email(user_info.get("mail", "")):
                        st.session_state.auth = {
                            "authenticated": True,
                            "user": user_info["displayName"],
                            "email": user_info["mail"],
                            "token": result['access_token']
                        }
                        st.rerun()
                    else:
                        st.error("Acesso permitido apenas para colaboradores com email corporativo.")
                else:
                    st.error("Falha na autenticação: " + result.get("error_description", ""))
            except Exception as e:
                st.error(f"Erro na autenticação: {str(e)}")
    
    st.stop()

# 4.0 INTERFACE PRINCIPAL
st.title(f"Bem-vindo, {st.session_state.auth['user']}!")
st.subheader("Chat com Documentos - H&P")

if st.button("Sair"):
    st.session_state.auth = {
        "authenticated": False,
        "user": None,
        "email": None,
        "token": None
    }
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()

# 5.0 PROCESSAMENTO DE PDF
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

# 6.0 CADEIA DE CONVERSA
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

# 7.0 INTERFACE DO CHAT
with st.expander("Boas práticas de prompt"):
    st.markdown("### Para escrever um bom prompt alguns elementos são necessários, são eles:")
    st.markdown("**Contexto:** Forneça informações de fundo ou contexto para orientar a conversa.")
    st.markdown("**Instrução Clara:** Uma tarefa ou pergunta específica que você quer que o modelo execute.")
    st.markdown("**Exemplos ou Casos de Uso:** Inclua exemplos ou cenários para ilustrar o que você espera.")
    st.markdown("**Restrições ou Limitações:** Defina limites para a resposta, como tamanho, tom ou escopo.")
    st.markdown("**Uso de Palavras-Chave:** Inclua palavras-chave relevantes para orientar o modelo. Exemplo: Discuta os impactos da mineração de ouro na **biodiversidade**, com foco em **desmatamento** e **contaminação da água**.")
    st.markdown("**Formato de Resposta:** Especifique como você quer que a resposta seja estruturada.")

uploaded_file = st.file_uploader(
    label="Escolha um arquivo PDF",
    type="pdf"
)

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processando PDF..."):
        try:
            st.session_state.retriever = process_pdf(uploaded_file)
            st.session_state.qa_chain = get_conversation_chain(st.session_state.retriever)
            st.session_state.pdf_processed = True
            st.success("PDF carregado! Faça suas perguntas.")
        except Exception as e:
            st.error(f"Erro: {str(e)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre o documento"):
    if not st.session_state.pdf_processed:
        st.warning("Envie um PDF primeiro.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            try:
                response = st.session_state.qa_chain({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Erro: {str(e)}")