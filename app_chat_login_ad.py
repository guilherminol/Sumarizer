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
import yaml
import msal
import requests
from urllib.parse import urlencode

# Carregar configurações
credentials = yaml.safe_load(open('./credentials.yml'))
OPENAI_API_KEY = credentials['openai']
AAD_CONFIG = credentials['microsoft']

# Configurações do Azure AD
CLIENT_ID = AAD_CONFIG['client_id']
TENANT_ID = AAD_CONFIG['tenant_id']
CLIENT_SECRET = AAD_CONFIG['client_secret']
REDIRECT_URI = AAD_CONFIG['redirect_uri']
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]

# 1.0 INICIALIZAR ESTADO DA SESSÃO
if "auth" not in st.session_state:
    st.session_state.auth = {
        "authenticated": False,
        "user": None,
        "email": None,
        "token": None
    }

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# 2.0 FUNÇÕES DE AUTENTICAÇÃO
def validate_company_email(email):
    return email.endswith("@hep.solutions")

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

# 3.0 FUNÇÕES DE GERENCIAMENTO DE CHATS
def create_new_chat():
    chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "name": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        "pdf_processed": False,
        "retriever": None,
        "qa_chain": None
    }
    st.session_state.active_chat = chat_id

def render_sidebar():
    with st.sidebar:
        st.header("Histórico de Chats")
        
        # Botão para novo chat
        if st.button("➕ Novo Chat"):
            create_new_chat()
        
        # Listagem dos chats existentes
        for chat_id, chat in st.session_state.chats.items():
            col1, col2 = st.columns([6,1])
            with col1:
                last_message = chat['messages'][-1]['content'][:20] + "..." if chat['messages'] else "Nova conversa"
                if st.button(
                    f"💬 {chat['name']}",
                    key=f"btn_{chat_id}",
                    help=last_message,
                    use_container_width=True
                ):
                    st.session_state.active_chat = chat_id
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.active_chat == chat_id:
                        create_new_chat()
                    st.rerun()

# 4.0 PROCESSAMENTO DE PDF
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

# 5.0 CADEIA DE CONVERSA
def get_conversation_chain(retriever, memory):
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
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# 6.0 INTERFACE DE LOGIN
if not st.session_state.auth["authenticated"]:
    st.title("Acesso Corporativo")
    st.markdown("Por favor, faça login com sua conta corporativa")

    if st.button("Entrar com Microsoft"):
        query_params = {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": " ".join(SCOPE),
            "response_mode": "query"
        }
        auth_url = f"{AUTHORITY}/oauth2/v2.0/authorize?{urlencode(query_params)}"
        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
        st.stop()
        
    # Processar resposta do Azure AD
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

# 7.0 INTERFACE PRINCIPAL
if st.session_state.auth["authenticated"]:
    if not st.session_state.chats:
        create_new_chat()
    
    render_sidebar()
    
    active_chat = st.session_state.chats[st.session_state.active_chat]
    
    st.title(f"Bem-vindo, {st.session_state.auth['user']}!")
    st.subheader(f"{active_chat['name']} - Chat com Documentos H&P")

    if st.sidebar.button("Sair"):
        st.session_state.auth = {
            "authenticated": False,
            "user": None,
            "email": None,
            "token": None
        }
        st.session_state.chats = {}
        st.session_state.active_chat = None
        st.rerun()

    # 8.0 UPLOAD E PROCESSAMENTO DE PDF
    uploaded_file = st.file_uploader(
        label="Escolha um arquivo PDF",
        type="pdf",
        key=f"uploader_{active_chat['id']}"
    )

    if uploaded_file and not active_chat['pdf_processed']:
        with st.spinner("Processando PDF..."):
            try:
                active_chat['retriever'] = process_pdf(uploaded_file)
                active_chat['qa_chain'] = get_conversation_chain(
                    active_chat['retriever'],
                    active_chat['memory']
                )
                active_chat['pdf_processed'] = True
                st.success("PDF carregado! Faça suas perguntas.")
            except Exception as e:
                st.error(f"Erro: {str(e)}")

    # 9.0 EXIBIÇÃO DE MENSAGENS
    for message in active_chat['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 10.0 TRATAMENTO DE INPUT DO USUÁRIO
    if prompt := st.chat_input("Faça sua pergunta sobre o documento"):
        if not active_chat['pdf_processed']:
            st.warning("Envie um PDF primeiro.")
            st.stop()
        
        active_chat['messages'].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    response = active_chat['qa_chain']({"question": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    active_chat['messages'].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erro: {str(e)}")