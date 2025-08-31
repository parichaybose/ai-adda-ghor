# Plan Advisor + Bill Breakdown (RAG + OpenAI Tool Calling) — Heavily Commented

## 1) What this does — in simple terms

We build a tiny assistant that can:
- Read a mini **plan catalog** (local `plans.json`).
- Pick the best plan for your **usage** (data GB, voice minutes, budget).
- Compute your **first-month bill** (base fee − discount + overage).
- Answer **policy** questions grounded in a local FAQ (`policy/faq.md`) using **RAG over Pinecone**.
- Let the LLM **decide** which functions to call via **OpenAI tool calling**.

Two flavors are provided:
- **A) Without LangChain** — raw OpenAI SDK with the official `tools=[...]` argument and a simple two-turn loop.
- **B) With LangChain** — same concept using `@tool` + `bind_tools` for cleaner schemas and message handling.

**Execution flow (both flavors):**
1. **User asks**: “I use ~60GB and ~1000 minutes, budget ₹800. Suggest a plan, compute bill, and confirm if Silver supports family pooling.”  
2. The **model decides** which **tools** to call (`match_plan`, `compute_first_month_bill`, `search_policy`).  
3. Your **Python functions** run, returning structured JSON results.  
4. Those results are **sent back** to the model.  
5. The model returns a **final, natural-language answer** (with plan + bill + policy note).

---

## 2) Folder prep (once)

```
project/
  plans.json
  policy/
    faq.md
```

**plans.json (example):**
```json
[
  {"code":"SILVER","name":"Silver 40GB","data_gb":40,"minutes":500,"monthly_fee":499,"overage_per_gb":20,"overage_per_min":0.5,"discount_pct":0},
  {"code":"GOLD","name":"Gold 75GB","data_gb":75,"minutes":1000,"monthly_fee":799,"overage_per_gb":18,"overage_per_min":0.4,"discount_pct":5},
  {"code":"PLAT","name":"Platinum 120GB","data_gb":120,"minutes":2000,"monthly_fee":1099,"overage_per_gb":15,"overage_per_min":0.3,"discount_pct":8}
]
```

**policy/faq.md (example):**
```
Fair Usage Policy (FUP): After exceeding plan data, overage charges apply per GB.
Unused minutes do not roll over. Family pooling is not supported on Silver plan.
...
```

---

## 3) A) WITHOUT LANGCHAIN — Raw OpenAI Tool Calling + Pinecone RAG (Very Commented)

> Install once:
> ```bash
> pip install openai pinecone-client tiktoken
> ```

```python
# -----------------------------------------------
# IMPORTS — explicit so it's easy to see sources
# -----------------------------------------------
import os                  # stdlib: read environment variables, etc.
import json                # stdlib: parse JSON files (plans.json) and format tool results
import pathlib             # stdlib: file path utilities
import re                  # stdlib: simple text cleanup
from typing import List, Dict, Any, Optional  # stdlib: type hints for readability

import tiktoken                              # third-party: tokenization (pip install tiktoken)
from openai import OpenAI                    # third-party: OpenAI SDK (pip install openai)
from pinecone import Pinecone, ServerlessSpec # third-party: Pinecone SDK (pip install pinecone-client)

# -----------------------------------------------
# CONFIGURATION — models + clients + index name
# -----------------------------------------------
OPENAI_MODEL_LLM = "gpt-4o-mini"             # chat model for reasoning + responses
OPENAI_MODEL_EMB = "text-embedding-3-small"  # embedding model (dimension=1536)

# Read API keys from environment variables (export/set beforehand)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Pick a Pinecone index name (create if missing). In production, pass via env var.
INDEX_NAME = os.environ.get("PINECONE_INDEX", "policy-faq")

# -----------------------------------------------
# PINECONE INDEX SETUP — create if not present
# - dimension MUST match embedding model (1536 for text-embedding-3-small)
# - metric 'cosine' is standard for text similarity
# -----------------------------------------------
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # pick a region close to your runtime
    )
index = pc.Index(INDEX_NAME)  # handle to our Pinecone index

# -----------------------------------------------
# CHUNKING — break long text into overlapping windows by tokens
# Why overlap? To preserve context that straddles chunk boundaries.
# -----------------------------------------------
def split_tokens(text: str, max_tokens: int = 1000, overlap: int = 150, enc_name: str = "cl100k_base") -> List[str]:
    enc = tiktoken.get_encoding(enc_name)             # choose a tokenizer
    clean = re.sub(r"\s+", " ", text).strip()         # normalize whitespace
    toks = enc.encode(clean)                          # tokenize string -> token IDs
    step = max_tokens - overlap                       # slide amount each window
    chunks: List[str] = []
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]                 # slice tokens for this chunk
        chunks.append(enc.decode(window))             # convert back to text for embedding
        i += step                                     # move forward, keeping 'overlap' tokens
    return chunks

# -----------------------------------------------
# EMBEDDINGS — convert text chunks into vectors (list[float])
# We batch inputs for fewer HTTP round trips (OpenAI handles lists).
# -----------------------------------------------
def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_MODEL_EMB, input=texts)
    return [d.embedding for d in resp.data]           # extract the vectors

# -----------------------------------------------
# BUILD POLICY INDEX — read policy/faq.md, chunk, embed, upsert to Pinecone
# NOTE: in a real app, do this once on startup or after docs change.
# -----------------------------------------------
def build_policy_index(policy_dir: str = "policy", namespace: Optional[str] = None) -> None:
    vectors: List[Dict[str, Any]] = []
    for p in pathlib.Path(policy_dir).glob("**/*.md"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        chunks = split_tokens(raw, 1000, 150)        # 1000 tokens per chunk, 150 overlap
        vecs = embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            vectors.append({
                "id": f"{p.stem}-{i}",              # stable-ish ID per chunk (stem + index)
                "values": vec,                       # the embedding vector
                "metadata": {                        # metadata you can filter/log/cite
                    "text": chunk,
                    "source": str(p),
                    "chunk_id": i
                }
            })
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)  # upsert all at once (ok for small corpora)

# -----------------------------------------------
# RAG RETRIEVAL — query by embedding similarity (top-k)
# Returns the Pinecone matches (each has .score, .metadata, etc.)
# -----------------------------------------------
def search_policy(query: str, k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    qv = embed([query])[0]                            # embed question
    res = index.query(                                # Pinecone vector search
        vector=qv, top_k=k, include_metadata=True, namespace=namespace
    )
    return res["matches"]

# -----------------------------------------------
# PLAN CATALOG — load local JSON file with plan options
# -----------------------------------------------
def load_plans(path: str = "plans.json") -> List[Dict[str, Any]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

# -----------------------------------------------
# PLAN MATCHER — pick best plans given usage
# Scoring is simple and *explainable*: penalize shortages more than surpluses.
# -----------------------------------------------
def match_plan(usage: Dict[str, Any], plans: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
    # Desired usage from user
    want_data = float(usage.get("data_gb", 0.0))
    want_min  = float(usage.get("minutes", 0.0))
    budget    = float(usage.get("budget", 1e9))  # if budget missing, treat as very high

    ranked: List[Dict[str, Any]] = []
    for p in plans:
        # Compute shortages (if any). Surplus is OK (no penalty).
        gap_data = max(0.0, want_data - float(p["data_gb"]))
        gap_min  = max(0.0, want_min  - float(p["minutes"]))

        # Weight data more than minutes (domain choice, not sacred)
        distance = gap_data * 2.0 + gap_min * 0.01

        # Soft penalty if plan price exceeds budget
        price_penalty = 0.0 if p["monthly_fee"] <= budget else (p["monthly_fee"] - budget) * 0.001

        score = distance + price_penalty
        ranked.append({**p, "_score": score})

    # Sort ascending: smaller score == better match
    ranked.sort(key=lambda x: x["_score"])
    return ranked[:top_n]

# -----------------------------------------------
# BILL CALCULATION — month 1 total = base − discount + overage
# Overage uses plan rates: per GB and per minute.
# -----------------------------------------------
def compute_bill(plan: Dict[str, Any], usage: Dict[str, Any]) -> Dict[str, Any]:
    base = float(plan["monthly_fee"])
    discount = base * (float(plan.get("discount_pct", 0.0)) / 100.0)

    # Compute usage beyond plan allowances
    over_gb  = max(0.0, float(usage.get("data_gb", 0.0)) - float(plan["data_gb"]))
    over_min = max(0.0, float(usage.get("minutes", 0.0)) - float(plan["minutes"]))

    overage = over_gb * float(plan["overage_per_gb"]) + over_min * float(plan["overage_per_min"])
    total = round(base - discount + overage, 2)

    return {
        "plan_code": plan["code"],
        "plan_name": plan["name"],
        "base_fee": base,
        "discount": round(discount, 2),
        "overage": round(overage, 2),
        "total": total
    }

# ===================================================================
# OPENAI TOOL CALLING — define tool schemas and Python dispatchers
# ===================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "match_plan",
            "description": "Return best-matching plans for the given usage profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usage": {
                        "type": "object",
                        "properties": {
                            "data_gb": {"type": "number"},
                            "minutes": {"type": "number"},
                            "budget":  {"type": "number"}
                        },
                        "required": ["data_gb", "minutes"]
                    },
                    "top_n": {"type": "integer", "default": 2}
                },
                "required": ["usage"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_first_month_bill",
            "description": "Compute base - discount + overage for the selected plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_code": {"type": "string"},
                    "usage": {
                        "type": "object",
                        "properties": {
                            "data_gb": {"type": "number"},
                            "minutes": {"type": "number"}
                        },
                        "required": ["data_gb", "minutes"]
                    }
                },
                "required": ["plan_code", "usage"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Semantic search over policy FAQ; returns top chunks as context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

# In-memory cache so we don't keep reloading plans
_PLANS_CACHE: Optional[List[Dict[str, Any]]] = None

def tool_match_plan(usage: Dict[str, Any], top_n: int = 2) -> str:
    """Implementation for 'match_plan' tool. Reads plans (cached), ranks, returns top-N. Returns JSON string."""
    global _PLANS_CACHE
    if _PLANS_CACHE is None:
        _PLANS_CACHE = load_plans()
    top = match_plan(usage, _PLANS_CACHE, top_n)
    return json.dumps({"plans": top}, ensure_ascii=False)

def tool_compute_first_month_bill(plan_code: str, usage: Dict[str, Any]) -> str:
    """Implementation for 'compute_first_month_bill' tool. Computes bill for the chosen plan code."""
    global _PLANS_CACHE
    if _PLANS_CACHE is None:
        _PLANS_CACHE = load_plans()
    selected = next((p for p in _PLANS_CACHE if p["code"] == plan_code), None)
    if not selected:
        return json.dumps({"error": f"unknown plan_code {plan_code}"})
    bill = compute_bill(selected, usage)
    return json.dumps({"bill": bill, "plan": selected}, ensure_ascii=False)

def tool_search_policy(query: str, k: int = 5) -> str:
    """Implementation for 'search_policy' tool. Runs vector search and returns top text chunks."""
    matches = search_policy(query, k=k)
    items = [{"text": m["metadata"]["text"], "source": m["metadata"]["source"], "score": m["score"]} for m in matches]
    return json.dumps({"matches": items}, ensure_ascii=False)

PY_TOOLS = {
    "match_plan": tool_match_plan,
    "compute_first_month_bill": tool_compute_first_month_bill,
    "search_policy": tool_search_policy
}

def run_tool_call(name: str, args_json: str) -> str:
    """Generic runner: parse JSON args and call the Python function. Returns JSON string."""
    args = json.loads(args_json or "{}")
    if name not in PY_TOOLS:
        return json.dumps({"error": f"unknown tool {name}"})
    return PY_TOOLS[name](**args)

def ask_with_tools(user_question: str) -> str:
    """Two-turn tool-calling loop using raw OpenAI SDK."""
    first = client.chat.completions.create(
        model=OPENAI_MODEL_LLM,
        messages=[{"role": "user", "content": user_question}],
        tools=TOOLS_SPEC,
        tool_choice="auto",
        temperature=0.2
    )

    msg = first.choices[0].message
    tool_calls = msg.tool_calls or []

    if not tool_calls:
        return msg.content or ""

    tool_messages = []
    for call in tool_calls:
        result = run_tool_call(call.function.name, call.function.arguments)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": result
        })

    final = client.chat.completions.create(
        model=OPENAI_MODEL_LLM,
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in tool_calls
            ]},
            *tool_messages
        ],
        temperature=0.2
    )
    return final.choices[0].message.content or ""
```

```python
# ---------------------------------------------------
# LANGCHAIN VERSION — compact, with heavy comments
# ---------------------------------------------------
# Install:
#   pip install langchain langchain-openai langchain-pinecone pinecone-client tiktoken

import os, json, pathlib
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
emb = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ.get("PINECONE_INDEX", "policy-faq")
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))

# VectorStore wrapper gives us a retriever for policy/faq.md (assumes upsert done)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def load_plans(path: str = "plans.json") -> List[Dict[str, Any]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

_PLANS = load_plans()

@tool("match_plan")
def match_plan_tool(usage: Dict[str, Any], top_n: int = 2) -> str:
    """Return best-matching plans for the given usage. Returns JSON string."""
    want_data = float(usage.get("data_gb", 0.0))
    want_min  = float(usage.get("minutes", 0.0))
    budget    = float(usage.get("budget", 1e9))
    ranked = []
    for p in _PLANS:
        gap_data = max(0.0, want_data - float(p["data_gb"]))
        gap_min  = max(0.0, want_min - float(p["minutes"]))
        distance = gap_data * 2.0 + gap_min * 0.01
        price_penalty = 0.0 if p["monthly_fee"] <= budget else (p["monthly_fee"] - budget) * 0.001
        ranked.append({**p, "_score": distance + price_penalty})
    ranked.sort(key=lambda x: x["_score"])
    return json.dumps({"plans": ranked[:top_n]}, ensure_ascii=False)

@tool("compute_first_month_bill")
def compute_first_month_bill_tool(plan_code: str, usage: Dict[str, Any]) -> str:
    """Compute first-month bill for selected plan code. Returns JSON string."""
    plan = next((p for p in _PLANS if p["code"] == plan_code), None)
    if not plan:
        return json.dumps({"error": f"unknown plan_code {plan_code}"})
    base = float(plan["monthly_fee"])
    discount = base * (float(plan.get("discount_pct", 0.0)) / 100.0)
    over_gb  = max(0.0, float(usage.get("data_gb", 0.0)) - float(plan["data_gb"]))
    over_min = max(0.0, float(usage.get("minutes", 0.0)) - float(plan["minutes"]))
    overage  = over_gb * float(plan["overage_per_gb"]) + over_min * float(plan["overage_per_min"])
    total = round(base - discount + overage, 2)
    return json.dumps({"bill":{"base_fee":base,"discount":round(discount,2),"overage":round(overage,2),"total":total},"plan":plan}, ensure_ascii=False)

@tool("search_policy")
def search_policy_tool(query: str, k: int = 5) -> str:
    """Semantic search in policy FAQ using retriever. Returns JSON string with top chunk texts."""
    docs = retriever.invoke(query)
    items = [{"text": d.page_content, "source": d.metadata.get("source","")} for d in docs[:k]]
    return json.dumps({"matches": items}, ensure_ascii=False)

# Bind tools: model can emit tool_calls for these functions
tools = [match_plan_tool, compute_first_month_bill_tool, search_policy_tool]
llm_w_tools = llm.bind_tools(tools)

def ask_with_langchain_tools(user_question: str) -> str:
    """Two-turn loop using LangChain messages + bind_tools."""
    # Turn 1: model may request tools
    ai: AIMessage = llm_w_tools.invoke([HumanMessage(content=user_question)])

    if not getattr(ai, "tool_calls", None):
        return ai.content

    # Execute each requested tool and prepare ToolMessage objects
    tool_msgs: List[ToolMessage] = []
    for call in ai.tool_calls:
        name = call["name"]
        args = call["args"]
        if name == "match_plan":
            out = match_plan_tool.invoke(args)
        elif name == "compute_first_month_bill":
            out = compute_first_month_bill_tool.invoke(args)
        elif name == "search_policy":
            out = search_policy_tool.invoke(args)
        else:
            out = json.dumps({"error": f"unknown tool {name}"})
        tool_msgs.append(ToolMessage(tool_call_id=call["id"], name=name, content=out))

    # Turn 2: feed tool results back for final answer
    final: AIMessage = llm_w_tools.invoke([HumanMessage(content=user_question), ai, *tool_msgs])
    return final.content

# Example (uncomment to try):
# q = "I use 60GB and 1000 minutes, budget 800. Recommend plan, compute bill, and check if Silver supports family pooling."
# print(ask_with_langchain_tools(q))
```

---

## 6) Quick test harness

```python
# ONE-TIME (raw SDK version): build Pinecone index from policy/faq.md
# build_policy_index()

question = "I use around 60GB and 1000 minutes, budget 800. Suggest a plan, compute first month bill, and confirm if Silver supports family pooling."
# print(ask_with_tools(question))             # Without LangChain
# print(ask_with_langchain_tools(question))   # With LangChain
```

---

**That’s it!** You now have a fully annotated, end-to-end demo of **tool calling** (agent-like) and **RAG with Pinecone**, in both raw SDK and LangChain styles.
