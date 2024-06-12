# Treatment Checker

## Overview
This is a toy version of a system which helps diagnose patients using Retrieval Augmented Generation and Langchain.

It demonstrates serveral fun and interesting techniques such as:
* RAG with reciprocal rank fusion (RAG-Fusion).
* Semantic search with vector stores.
* Semantic query classification and routing.
* Few-shot prompt engineering.
* Output schema enforcement with output parsers.
* Langchain component chaining.

## Processes
The system is divided into three distinct processes, providing 3 similar elements of functionality. All functionality is based on a user query, which is classified then routed to the relevent process.

```mermaid
flowchart LR

A[Question] -->|Text| B{Semantic\nQuery Router}
B -->|Full Service| C1[RAG] -->|Hist Docs| C2[Rag-Fusion] --> C3(Answer)
B -->|Diagnosis Advice| D1[Rag-Fusion] --> D2(Answer)
B -->|Patient Hist| E1[Patient History] --> E2(Answer)
```

#### Patient History Process: 
This process is designed to answer questions about a specific patient’s medical history.

```mermaid a
flowchart LR

A[Question] -->|Text| B[Retriever] -->|Simularity\nSearch| C[Patient\nHistory DB] -->|Doc| D[Chat Prompt] --> E[Output\nParser] --> F(Answer)
```

#### Diagnosis Advice Process:
This process provides suggestions for a drug to diagnose based on symptoms from the app’s drug database.

```mermaid a
flowchart LR

A[Question] -->|Text| B[Query\nTransformation]-->|Query 1| L[Query\nTransformation]
B[Query\nTransformation]-->|Query 2| L[Query\nTransformation]
B[Query\nTransformation]-->|Query 3| L[Query\nTransformation]
L[Retriever] -->|Simularity\nSearch| C[Patient\nHistory DB]
L[Retriever] -->|Simularity\nSearch| C[Patient\nHistory DB]
L[Retriever] -->|Simularity\nSearch| C[Patient\nHistory DB]
C[Patient\nHistory DB] -->|Doc| H{Reciprocal\nRank Fusion}
C[Patient\nHistory DB] -->|Doc| H{Reciprocal\nRank Fusion}
C[Patient\nHistory DB] -->|Doc| H{Reciprocal\nRank Fusion}
H -->|Best Docs| D[Chat Prompt] --> E[Output\nParser] --> F(Answer)
```

#### Full Service Process:
This process makes drug recommendations based on the patient’s medical history and the available drugs in the drug database.

*Simplified*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Patient\nIdentification] -->|Patient| B1[Patient\nHist DB]
A[Question] -->|Text| C[Multi\nQuery] -->|Many Questions| D[Reciprocal\nRank\nFusion]
D -->|Drug DB Context| E[Prompt]
A -->|Orig Question| E[Prompt]
B1 -->|Patient Hist DB Context| E[Prompt]
E --> E2[Chat Model] --> E3[Parser] --> E4[Answer]
```

*Patient Identification Sub-Process*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Retriever] --> C[Patient ID\nPrompt] --> D[Output\nParser]
```

*Multi Query Sub-Process*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Query\nTransformation]-->|Query 1| L[Query\nTransformation]
B[Query\nTransformation]-->|Query 2| L[Query\nTransformation]
B[Query\nTransformation]-->|Query 3| L(OUT)
```
*Reciprocal Rank Fusion Sub-Process*
```mermaid  a
flowchart LR

a1[IN] -->|Query 1| L[Retriever] -->|Simularity\nSearch| C[DB]
a2[IN] -->|Query 2| L[Retriever] -->|Simularity\nSearch| C[DB]
a3[IN] -->|Query 3| L[Retriever] -->|Simularity\nSearch| C[DB]
```

