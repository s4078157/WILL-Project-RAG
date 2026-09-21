# DATA CONTRACT





1. ### Chunk Format

###### Every research chunk must contain:



* `chunk\_id`
* `paper\_id`
* `title`
* `authors`
* `year`
* `page`
* `doi`
* `text`

###### 

###### example:

```

chunk\\\_id: P01\_C001

paper\\\_id:  P01

title: Sleep Deprivation and Attention

authors: Jhon Doe

year: 2020

page:  6

doi: 10.xxxx/xxxxx

text: (Research chunk content)

```



### 2\. DPR Passage Input



###### Each chunk text will be encoded using:

`facebook/dpr-ctx\_encoder-multiset-base`

###### and the output  embedding will be stored in the FAISS index



###### Flow:

Chunk Text

`DPR Context Encoder → Passage Embedding → FAISS`



### 3\. Retrieval Output

###### Retriever should return Top-5 chunks in this template format:

```

\[

&#x09;{

&#x09;"chunk\_id": "P01\_C001",

&#x09;"paper\_id": "P01",

&#x20;       "title": "Sleep Deprivation and Attention",

&#x20;       "page": 6,

&#x20;       "doi": "10.xxxx/xxxxx",

&#x20;       "text": "...",

&#x20;       "score": 0.88

&#x09;}

] 

```

###### Question retrieval flow:

`→ DPR Question Encoder → Question Embedding → FAISS Search → Top-5 Chunks`





### 4\. RAG Output

###### final RAG system **should** return:

```

{

&#x20;   "question": "...",

&#x20;   "answer": "...",

&#x20;   "sources": \[

&#x20;       {

&#x20;           "chunk\_id": "P01\_C001",

&#x20;           "title": "Sleep Deprivation and Attention",

&#x20;           "page": 6,

&#x20;           "doi": "10.xxxx/xxxxx"

&#x20;       }

&#x20;   ],

&#x20;   "retrieved\_chunks": \[...]

}

```



### 5\. Source Traceability

###### Every answer **must** **be** traceable back to:

`Chunk → Research Paper → Page → DOI / Source`

###### and The LLM **should not create sources that were not retrieved** (Hallucinate)



### 6\. Insufficient Evidence

###### If the Top-5 retrieved chunks do not contain enough evidence to answer the question, the generator should return an insufficient-evidence response instead of inventing an answer.





