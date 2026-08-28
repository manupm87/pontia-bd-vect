# Referencias primarias y documentación oficial

## Dataset del caso práctico

- [Shopping Queries Dataset](https://github.com/amazon-science/esci-data): repositorio oficial de Amazon Science, esquema, descarga y licencia Apache-2.0.
- [ESCI: Improving Product Search with Customer Behavior](https://arxiv.org/abs/2206.06588): publicación original del dataset y de las etiquetas Exact, Substitute, Complement e Irrelevant.

## APIs de embeddings

### OpenAI

- [Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings): uso del endpoint, dimensión acortada y ejemplos de recuperación.
- [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small): página oficial del modelo, contexto y precios vigentes.
- [text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large): página oficial del modelo, contexto y precios vigentes.
- [Embeddings FAQ](https://help.openai.com/en/articles/6824809-embeddings-frequently-asked-questions): normalización L2 y consecuencias para coseno y producto escalar.

### Cohere

- [Cohere Embed](https://docs.cohere.com/docs/cohere-embed): modalidades, idiomas, contexto y dimensiones de Embed 4.
- [Embed API v2](https://docs.cohere.com/v2/reference/embed): contrato de ClientV2, input_type, embedding_types y output_dimension.
- [Embed 4 multimodal](https://docs.cohere.com/changelog/embed-multimodal-v4): anuncio oficial de capacidades multimodales.

### Google Gemini

- [Gemini API embeddings](https://ai.google.dev/gemini-api/docs/embeddings): gemini-embedding-2, modalidades, dimensiones, normalización, prefijos y semántica de Content.
- [Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations): retirada de gemini-embedding-001 el 14 de julio de 2026.
- [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog): paso de gemini-embedding-2 a disponibilidad estable.

## Modelos con pesos accesibles

- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): model card oficial, 384 dimensiones, 50 idiomas y Apache-2.0.
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small): model card oficial, prefijos query/passage, 384 dimensiones y licencia MIT.
- [Multilingual E5 Text Embeddings](https://arxiv.org/abs/2402.05672): publicación de la familia E5 multilingüe.
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3): implementación y salidas dense, sparse y ColBERT.
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216): publicación original sobre multi-functionality, multi-linguality y multi-granularity.
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): model card oficial, instrucciones, contexto, MRL y Apache-2.0.
- [Qwen3 Embedding](https://arxiv.org/abs/2506.05176): publicación de la familia de embeddings y rerankers.
- [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card): tamaño, idiomas, dimensiones Matryoshka y términos de Gemma.
- [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe): model card oficial, arquitectura MoE, dimensiones, idiomas y Apache-2.0.
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B): model card oficial de la variante multimodal 2B.
- [Qwen3-VL Embedding and Reranker](https://arxiv.org/abs/2601.04720): publicación de embeddings y reranking visual-lingüístico.
- [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4): model card, modos single-vector y multi-vector, y Qwen Research License.
- [Jina Embeddings v4](https://arxiv.org/abs/2506.18902): publicación del modelo universal multimodal.
- [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small): model card, contexto largo, MRL y licencia CC-BY-NC-4.0.
- [jina-embeddings-v5-omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small): model card de texto, imagen, vídeo, audio y PDF; licencia CC-BY-NC-4.0.
- [pplx-embed-v1-0.6B](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b): model card, MRL, cuantización y licencia MIT.
- [pplx-embed](https://arxiv.org/abs/2602.11151): publicación técnica de la familia.
- [ColPali v1.3](https://huggingface.co/vidore/colpali-v1.3): model card del checkpoint.
- [ColPali reference implementation](https://github.com/illuin-tech/colpali): repositorio oficial de ColPali, ColQwen y sucesores comunitarios.

## Representaciones dispersas y densas clásicas

- [Term-weighting approaches in automatic text retrieval](https://doi.org/10.1016/0306-4573(88)90021-0): análisis clásico de ponderación de términos y TF-IDF.
- [Introduction to Information Retrieval: The term vocabulary and postings lists](https://nlp.stanford.edu/IR-book/html/htmledition/the-term-vocabulary-and-postings-lists-1.html): referencia abierta sobre vocabularios, postings e índices invertidos.
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781): Word2Vec, CBOW y Skip-gram.
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/): vectores basados en estadísticas globales de coocurrencia.
- [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/): FastText y n-gramas de caracteres.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/): representaciones contextuales con Transformers.
- [Sentence-BERT](https://aclanthology.org/D19-1410/): encoders siameses para embeddings de oración y búsqueda eficiente.

## Dimensión, sparse aprendido, late interaction e híbridos

- [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html): entrenamiento de representaciones anidadas y dimensión adaptativa.
- [SPLADE v2](https://arxiv.org/abs/2109.10086): representación dispersa aprendida y expansión léxica.
- [ColBERT](https://arxiv.org/abs/2004.12832): interacción tardía eficiente sobre vectores contextualizados por token.
- [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/): entrenamiento denoised y compresión residual del índice.
- [ColPali](https://proceedings.iclr.cc/paper_files/paper/2025/hash/99e9e141aafc314f76b0ca3dd66898b3-Abstract-Conference.html): late interaction sobre páginas como imágenes.
- [Reciprocal Rank Fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/): fusión de rankings basada en posiciones.
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085): cross-encoder para reranking de candidatos.

## Evaluación

- [BEIR](https://arxiv.org/abs/2104.08663): benchmark heterogéneo de recuperación zero-shot.
- [MTEB](https://arxiv.org/abs/2210.07316): benchmark original de 8 familias de tareas, 58 datasets y 112 idiomas.
- [MTEB repository](https://github.com/embeddings-benchmark/mteb): código, datasets y evolución del benchmark.
- [MTEB Benchmark API](https://docs.mteb.org/api/benchmark/): definición versionada de benchmarks y tareas.
- [MTEB leaderboard](https://leaderboard.mteb.org/): tabla viva; consultar con fecha y filtros, no como verdad atemporal.
- [MMTEB](https://arxiv.org/abs/2502.13595): ampliación a más de 500 tareas, 250+ idiomas y 10 categorías.
- [ColPali y ViDoRe](https://arxiv.org/abs/2407.01449): presentación del benchmark de recuperación visual documental.
- [ViDoRe Benchmark v2](https://arxiv.org/abs/2505.17166): segunda versión del benchmark visual.


