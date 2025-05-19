import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GRPQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
WikiWrapper =  WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=WikiWrapper)

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
ArxivWrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=ArxivWrapper)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.who.int/")
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama3")


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.vectorstores.qdrant import Qdrant as QdrantVectorStore

collection_name = "health_docs"
qdrant_client = QdrantClient(location=":memory:")

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=4096,  # Match the output size of the Ollama embedding
            distance=Distance.COSINE
        )
    )

vectordb = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings
    
)
vectordb.add_documents(documents)

# Create retriever
retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "Health_search",
    "Search for information about Health. For any questions relating to health, you must use this tool!"
)

tools = [wiki, arxiv, retriever_tool]

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

# Get the prompt to use
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

### Agents
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

## Agent Executer
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test it
agent_executor.invoke({"input": "Tell me about cancer"})
