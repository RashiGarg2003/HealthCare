import langchain
langchain.verbose = False
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

import os
google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",   
    temperature=0.5,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
 
# llm = genai.GenerativeModel(
#     model_name="gemini-flash-latest",
#     generation_config={
#         "temperature": 0.5
#     }
# )

   
# Create a Buffer memory 
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.load_local(
    "faiss_index",
    hf_embeddings,
    allow_dangerous_deserialization=True
)
 
memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True , output_key="answer")

retriever = faiss_db.as_retriever(search_type="mmr", search_kwargs={"k":5})
 
# Create Chat Prompt Template
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template = """
    You are an expert Healthcare AI Assistant.
    Answer ONLY using the provided medical context.
    If the answer is not present in the context,say:"I could not find sufficient medical evidence in the knowledge base."
    Guidelines:
    - Be medically accurate
    - Do not hallucinate
    - Do not invent treatments
    - Keep answers clear and professional
    - Mention uncertainty when necessary
    Context:
    {context}
    Chat History:
    {chat_history}
    Question:
    {question}
    Medical Answer:
    """
 
)
# Create QA Chain
from langchain.chains import ConversationalRetrievalChain
QA_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    # verbose=False,
    combine_docs_chain_kwargs={
        "prompt": prompt
    },
    output_key = 'answer'
)

def ask_question(question: str) -> dict:
    result = QA_chain.invoke({
        "question": question
    })
    print("=" * 70)
    print(f"Question: {question}")
    print("\nAnswer:\n")
    print(result["answer"])
    print("\n" + "=" * 70)
    print("Retrieved Sources:\n")
    for i, doc in enumerate(result["source_documents"], start=1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        print(f"[{i}] Source: {source}")
        print(doc.page_content[:300])
        print("\n" + "-" * 50)
        print("=" * 70)
    return result
 
from typing import TypedDict, Annotated, List, Dict,Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

   
# Share State
class HealthcareState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str          # original user question
    retrieved_context: str   # what retriever agent found
    consultation_response: str # what consultation agent generated
    diagnosis_result: str    # what diagnosis agent suggested
    final_response: str      # combined final answer
    finished: bool

 
@tool
def retrieve_medical_articles(query: str) -> str:
   """
   Retriever Agent Tool.
   Retrieves relevant medical articles,
   patient notes and guidelines
   from the healthcare knowledge base.
   """
   result = ask_question(query)
   return result["answer"]

 
@tool
def generate_consultation_response(query: str) -> str:
   """
   Consultation Agent Tool.
   Generates detailed medical consultation
   response grounded in retrieved context.
   """
   consultation_query = f"""
   As a medical consultant, provide a detailed
   response for: {query}
   Base your response on medical evidence
   and clinical guidelines.
   """
   result = ask_question(consultation_query)
   return result["answer"]

 
@tool
def suggest_diagnosis(symptoms: str) -> str:
   """
   Diagnosis Support Agent Tool.
   Suggests possible medical conditions
   based on symptoms from retrieved evidence.
   """
   diagnosis_query = f"""
   Based on these symptoms: {symptoms}
   What are the possible medical conditions?
   What diagnostic tests are recommended?
   What are the red flag symptoms to watch for?
   """
   result = ask_question(diagnosis_query)
   return result["answer"]

 
RETRIEVER_AGENT_PROMPT = """
You are a Medical Retriever Agent.
Your ONLY job is to retrieve relevant medical
information from the knowledge base.
When user asks a medical question:
1. Use retrieve_medical_articles tool to fetch
  relevant medical articles and guidelines
2. Return the retrieved information clearly
3. Do NOT diagnose or consult - only retrieve
Always use the retrieve_medical_articles tool
for every user query.
"""


 
CONSULTATION_AGENT_PROMPT = """
You are a Medical Consultation Agent.
Your job is to generate detailed medical responses
grounded in the retrieved context.
When given retrieved medical information:
1. Use generate_consultation_response tool
2. Provide clear, detailed medical consultation
3. Always base response on retrieved evidence
4. Include relevant medical guidelines
Always use generate_consultation_response tool.
"""

 
DIAGNOSIS_AGENT_PROMPT = """
You are a Medical Diagnosis Support Agent.
Your job is to suggest possible conditions
based on symptoms from retrieved evidence.
When given symptoms or medical context:
1. Use suggest_diagnosis tool
2. List possible conditions clearly
3. Suggest relevant diagnostic tests
4. Mention red flag symptoms
5. Always recommend consulting a real doctor
Always use suggest_diagnosis tool.
"""

 # Each agent gets its own tools
retriever_tools = [retrieve_medical_articles]
consultation_tools = [generate_consultation_response]
diagnosis_tools = [suggest_diagnosis]

# retriever_llm = llm.bind_tools(retriever_tools)
# consultation_llm = llm.bind_tools(consultation_tools)
# diagnosis_llm = llm.bind_tools(diagnosis_tools)
# print("All 3 agents bound to their tools!")

retriever_llm = llm
consultation_llm = llm
diagnosis_llm = llm
 

# Graph Nodes
  #  print("\n🔍 Retriever Agent working...")
#    messages = [
#        SystemMessage(content=RETRIEVER_AGENT_PROMPT),
#        HumanMessage(content=state["user_query"])
#    ]
#    response = retriever_llm.invoke(messages)
#    print(response)
#    retrieved_text = str(response.content)
#    return {
#        **state,
#        "messages": [response],
#        "retrieved_context": retrieved_text
#     }
def retriever_agent_node(state: HealthcareState):
   """
   Agent 1: Retrieves medical articles,
   patient notes and guidelines from FAISS.
   """
   print("\n🔍 Retriever Agent working...")
   response = retriever_llm.invoke(
      f"Answer this medical question briefly: {state['user_query']}"
    )
   print(response)
   return {
      **state,
      "retrieved_context": str(response.content)
    }
 
 
#    print("\n👨‍⚕️ Consultation Agent working...")
#    consultation_input = f"""
#    User Question: {state['user_query']}
#    Retrieved Medical Context:
#    {state['retrieved_context']}
#    Please provide a detailed medical consultation.
#    """
#    messages = [
#        SystemMessage(content=CONSULTATION_AGENT_PROMPT),
#        HumanMessage(content=consultation_input)
#    ]
#    response = consultation_llm.invoke(messages)
#    print(response)
#    consultation_text = str(response.content)
#    return {
#        **state,
#        "messages": [response],
#        "retrieved_context": consultation_text
#     }
def consultation_agent_node(state: HealthcareState):
   """
   Agent 2: Generates consultation response
   grounded in retrieved context.
   """
   print("\n👨‍⚕️ Consultation Agent working...")
   response = consultation_llm.invoke(
      f"Provide medical consultation advice for: {state['user_query']}"
    )
   print(response)
   return {
       **state,
       "consultation_response": str(response.content)
    }
 

def diagnosis_agent_node(state: HealthcareState):
   """
   Agent 3: Suggests possible conditions
   from retrieved evidence.
   """
   print("\n🩺 Diagnosis Agent working...")
   response = diagnosis_llm.invoke(
       f"Suggest possible diagnosis for: {state['user_query']}"
    )
   print(response)
   return {
       **state,
       "diagnosis_result": str(response.content)
    }
 
#    print("\n🩺 Diagnosis Support Agent working...")
#    diagnosis_input = f"""
#    User Query: {state['user_query']}
#    Retrieved Context: {state['retrieved_context']}
#    Consultation Response:
#    {state['consultation_response']}
#    Based on this evidence, suggest possible
#    conditions and diagnostic approach.
#    """
#    messages = [
#        SystemMessage(content=DIAGNOSIS_AGENT_PROMPT),
#        HumanMessage(content=diagnosis_input)
#    ]
#    response = diagnosis_llm.invoke(messages)
#    print(response)
#    diagnosis_text = str(response.content)
#    return {
#        **state,
#        "messages": [response],
#        "retrieved_context": diagnosis_text
#     }

 
def final_response_node(state: HealthcareState):

    final_text = f"""

Question: {state.get('user_query', '')}

Retrieved Info:

{state.get('retrieved_context', '')}

Consultation:

{state.get('consultation_response', '')}

Diagnosis:

{state.get('diagnosis_result', '')}

"""

    return {

        **state,

        "final_response": final_text

    }
 

def human_input_node(state: HealthcareState):
    """Gets next question from user."""
    user_text = input("\nYou (or type 'quit' to exit): ").strip()
    if user_text.lower() in {"quit", "exit", "bye", "goodbye"}:
        print("\n🏥 Stay healthy! Goodbye!")
        return {**state, "finished": True}
    return {
        **state,
        "user_query": user_text,
        "retrieved_context": "",
        "consultation_response": "",
        "diagnosis_result": "",
        "finished": False
    }
 
# Tool executor nodes for each agent
retriever_tool_node = ToolNode(retriever_tools)
consultation_tool_node = ToolNode(consultation_tools)
diagnosis_tool_node = ToolNode(diagnosis_tools)

# Routing Functions
def route_retriever(state: HealthcareState):
   """Route retriever to its tools or next agent."""
   last = state["messages"][-1]
   if hasattr(last, "tool_calls") and last.tool_calls:
       return "retriever_tools"
   return "consultation_agent"

 
def route_consultation(state: HealthcareState):
   """Route consultation to its tools or next agent."""
   last = state["messages"][-1]
   if hasattr(last, "tool_calls") and last.tool_calls:
       return "consultation_tools"
   return "diagnosis_agent"

 
def route_diagnosis(state: HealthcareState):
   """Route diagnosis to its tools or final response."""
   last = state["messages"][-1]
   if hasattr(last, "tool_calls") and last.tool_calls:
       return "diagnosis_tools"
   return "final_report"

 
def route_human(state: HealthcareState):
   """End or continue conversation."""
   if state.get("finished"):
       return END
   return "retriever_agent"

   
# Build a graph
graph = StateGraph(HealthcareState)

# Add all nodes
graph.add_node("retriever_agent", retriever_agent_node)
#graph.add_node("retriever_tools", retriever_tool_node)
graph.add_node("consultation_agent", consultation_agent_node)
#graph.add_node("consultation_tools", consultation_tool_node)
graph.add_node("diagnosis_agent", diagnosis_agent_node)
#graph.add_node("diagnosis_tools", diagnosis_tool_node)
graph.add_node("final_report", final_response_node)
#graph.add_node("human_input", human_input_node)

graph.add_edge(START, "retriever_agent")
graph.add_edge("retriever_agent", "consultation_agent")
graph.add_edge("consultation_agent", "diagnosis_agent")
graph.add_edge("diagnosis_agent", "final_report")
graph.add_edge("final_report", END)
 
# Compile
healthcare_graph = graph.compile()
print("Healthcare Multi-Agent Graph compiled!")