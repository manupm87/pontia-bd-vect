# Referencias · Sesión 3

## Datos y modelo

1. Amazon Science. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data). Dataset y licencia Apache-2.0.
2. Wang et al. [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533). Trabajo de E5.
3. Hugging Face. [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small). Model card, prefijos `query:`/`passage:` y dimensión.

## De índice a base de datos

4. Johnson, Douze y Jégou. [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734). FAISS como biblioteca de índices vectoriales.
5. Pinecone. [Create an index](https://docs.pinecone.io/reference/api/2026-04/control-plane/create_index). Dimensión, métrica, serverless, regiones y planes.
6. Pinecone. [Local development with Pinecone Local](https://docs.pinecone.io/guides/operations/local-development). Emulador en memoria y límite de 100.000 registros por índice.
7. Chroma. [Collections](https://docs.trychroma.com/docs/collections/manage-collections). Creación, acceso y modificación de colecciones.
8. Chroma. [Metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering). Gramática `where`.
9. Chroma. [Client-server mode](https://docs.trychroma.com/guides/deploy/client-server-mode). Despliegue y `HttpClient`.
10. Weaviate. [Python client](https://docs.weaviate.io/weaviate/client-libraries/python). Cliente v4, cierre de conexiones, batching y filtros.
11. Weaviate. [Bring your own vectors](https://docs.weaviate.io/weaviate/starter-guides/custom-vectors). Vectores aportados por el cliente.
12. Weaviate. [Filtering](https://docs.weaviate.io/weaviate/concepts/filtering). Prefiltrado, ACORN y estrategia plana para subconjuntos restrictivos.
13. Milvus. [`create_collection()` con PyMilvus 2.6.x](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/create_collection.md). Esquema, dimensión, métrica y shards.
14. Milvus. [Milvus Lite](https://milvus.io/docs/milvus_lite.md). Modo local embebido y diferencias de uso.
15. Milvus. [Consistency](https://milvus.io/docs/consistency.md). Niveles y *guarantee timestamp*.
16. Qdrant. [Collections, points and payload](https://qdrant.tech/documentation/overview/). Modelo de datos, UUIDs, HNSW, segmentos y shards.
17. Qdrant. [Quickstart: query and filter](https://qdrant.tech/documentation/quick-start/). `query_points`, `Filter` y recomendación de payload indexes.
18. Qdrant. [Payload](https://qdrant.tech/documentation/concepts/payload/). Tipos y rutas JSON.
19. Qdrant. [Low-latency search](https://qdrant.tech/documentation/guides/low-latency-search/). Índices de payload, filtros y recursos.

## LangChain

20. LangChain. [Embedding model integrations](https://docs.langchain.com/oss/python/integrations/embeddings/index). `HuggingFaceEmbeddings`, prompts y normalización.
21. LangChain. [Chroma vector store integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma). Búsqueda, filtros, retriever y MMR.
22. LangChain. [Qdrant vector store integration](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant). Conexión, búsqueda y modos de recuperación.
23. LangChain. [`PineconeVectorStore` reference](https://reference.langchain.com/python/langchain-pinecone/vectorstores/PineconeVectorStore). Referencia de la integración separada. En la resolución bloqueada en julio de 2026, `langchain-pinecone 0.2.13` exige `pinecone < 8`; por ello no se instala junto a Pinecone 9.x.
24. LangChain. [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document). Contrato de contenido, ID y metadata.
25. LangChain. [Retrievers](https://docs.langchain.com/oss/python/integrations/retrievers). Interfaz común e invocación.

