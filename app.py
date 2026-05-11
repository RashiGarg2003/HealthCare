from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag_pipeline import  healthcare_graph

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

class AgentResponse(BaseModel):
    question: str
    retrieved_context: str
    consultation_response: str
    diagnosis_result: str
    final_response: str
 

app = FastAPI(
    title="Healthcare Knowledge Assistant API",
    description="RAG Pipeline with Multi-Agent LangGraph",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Healthcare Knowledge Assistant API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

# @app.post("/query", response_model=QueryResponse)
# def query(request: QueryRequest):
#     """Basic RAG query endpoint."""
#     try:
#         result = ask_question(qa_chain, request.question)
#         return QueryResponse(
#             question=request.question,
#             answer=result["answer"],
#             sources=result["sources"]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent-query", response_model=AgentResponse)
def agent_query(request: QueryRequest):
    """Multi-agent LangGraph query endpoint."""
    try:
        result = healthcare_graph.invoke({
            "messages": [],
            "user_query": request.question,
            "retrieved_context": "",
            "consultation_response": "",
            "diagnosis_result": "",
            "final_response": "",
            "finished": False
        })
        return AgentResponse(
            question=request.question,
            retrieved_context=result.get("retrieved_context", "No context available"),
            consultation_response=result.get("consultation_response", "No consultation response"),
            diagnosis_result=result.get("diagnosis_result", "No diagnosis result"),
            final_response=result.get("final_response", "No final response")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 