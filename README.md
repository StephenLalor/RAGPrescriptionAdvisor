# Retrieval Augmented Generation Prescription Advisor

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

A[Question] -->|Text| B{Semantic<br>Query Router}
B -->|Full Service| C1[RAG] -->|Hist Docs| C2[Rag-Fusion] --> C3(Answer)
B -->|Diagnosis Advice| D1[Rag-Fusion] --> D2(Answer)
B -->|Patient Hist| E1[Patient History] --> E2(Answer)
```

#### Patient History Process: 
This process is designed to answer questions about a specific patient’s medical history.

```mermaid a
flowchart LR

A[Question] -->|Text| B[Retriever] -->|Simularity<br>Search| C[Patient<br>History DB] -->|Doc| D[Chat Prompt] --> E[Output<br>Parser] --> F(Answer)
```

#### Diagnosis Advice Process:
This process provides suggestions for a drug to diagnose based on symptoms from the app’s drug database.

```mermaid a
flowchart LR

A[Question] -->|Text| B[Query <br> Transformation]-->|Query 1| L[Query<br>Transformation]
B[Query<br>Transformation]-->|Query 2| L[Query<br>Transformation]
B[Query<br>Transformation]-->|Query 3| L[Query<br>Transformation]
L[Retriever] -->|Simularity<br>Search| C[Patient<br>History DB]
L[Retriever] -->|Simularity<br>Search| C[Patient<br>History DB]
L[Retriever] -->|Simularity<br>Search| C[Patient<br>History DB]
C[Patient<br>History DB] -->|Doc| H{Reciprocal<br>Rank Fusion}
C[Patient<br>History DB] -->|Doc| H{Reciprocal<br>Rank Fusion}
C[Patient<br>History DB] -->|Doc| H{Reciprocal<br>Rank Fusion}
H -->|Best Docs| D[Chat Prompt] --> E[Output<br>Parser] --> F(Answer)
```

#### Full Service Process:
This process makes drug recommendations based on the patient’s medical history and the available drugs in the drug database.

*Simplified*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Patient<br>Identification] -->|Patient| B1[Patient<br>Hist DB]
A[Question] -->|Text| C[Multi<br>Query] -->|Many Questions| D[Reciprocal<br>Rank<br>Fusion]
D -->|Drug DB Context| E[Prompt]
A -->|Orig Question| E[Prompt]
B1 -->|Patient Hist DB Context| E[Prompt]
E --> E2[Chat Model] --> E3[Parser] --> E4[Answer]
```

*Patient Identification Sub-Process*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Retriever] --> C[Patient ID<br>Prompt] --> D[Output<br>Parser]
```

*Multi Query Sub-Process*
```mermaid  a
flowchart LR

A[Question] -->|Text| B[Query<br>Transformation]-->|Query 1| L[Query<br>Transformation]
B[Query<br>Transformation]-->|Query 2| L[Query<br>Transformation]
B[Query<br>Transformation]-->|Query 3| L(OUT)
```
*Reciprocal Rank Fusion Sub-Process*
```mermaid  a
flowchart LR

a1[IN] -->|Query 1| L[Retriever] -->|Simularity<br>Search| C[DB]
a2[IN] -->|Query 2| L[Retriever] -->|Simularity<br>Search| C[DB]
a3[IN] -->|Query 3| L[Retriever] -->|Simularity<br>Search| C[DB]
```

