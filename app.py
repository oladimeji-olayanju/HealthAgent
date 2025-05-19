import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GRPQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
WikiWrapper =  WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=WikiWrapper)
wiki.name

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
ArxivWrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=ArxivWrapper)
arxiv.name

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.who.int/")
docs = loader.load()
docs

from langchain_text_splitters import RecursiveCharacterTextSplitter
documents = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=100).split_documents(docs)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama3")
from langchain_community.vectorstores import FAISS
vectordb = FAISS.from_documents(documents,embedding=embeddings)
retriever=vectordb.as_retriever()
retriever

from langchain.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever,"Health_search",
                      "Search for information about Health. For any questions relating to health, you must use this tool!")
retriever_tool.name
tools = [wiki,arxiv,retriever_tool]
tools
from langchain_groq import ChatGroq
llm=ChatGroq(model="llama3-8b-8192")

from langchain import hub
# Get the prompt to use 
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

### Agents
from langchain.agents import create_openai_tools_agent
agent=create_openai_tools_agent(llm,tools,prompt)

## Agent Executer
from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
agent_executor

agent_executor.invoke({"input":"Tell me about cancer"})