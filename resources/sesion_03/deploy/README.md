# Despliegues locales de la sesión 3

Una base de datos local no significa necesariamente una base de datos embebida en el proceso de Python. En el recorrido principal, el notebook es un cliente y cada motor corre como servicio Docker. Esa separación reproduce mejor el contrato que encontraremos en producción: red, readiness, persistencia, logs y ciclo de vida independiente.

Los compose deben ejecutarse **uno cada vez**. Milvus utiliza además etcd y MinIO, y puede necesitar 6–8 GiB de memoria disponible en Docker Desktop. Todos los nombres y volúmenes incluyen `bbdd-vectoriales-s03` para que la frontera de limpieza sea visible.

## Comandos comunes

```bash
docker compose -f deploy/qdrant/compose.yaml pull
docker compose -f deploy/qdrant/compose.yaml up -d
docker compose -f deploy/qdrant/compose.yaml ps
docker compose -f deploy/qdrant/compose.yaml logs --tail=100
```

Detener conservando datos:

```bash
docker compose -f deploy/qdrant/compose.yaml down
```

Borrar contenedores **y el volumen persistente de esa base**:

```bash
docker compose -f deploy/qdrant/compose.yaml down --volumes
```

El último comando es destructivo. Antes de ejecutarlo, confirma que el nombre del proyecto y todos los volúmenes contienen `bbdd-vectoriales-s03`. Nunca reutilices estos comandos con una colección compartida.

## Chroma 1.5.9

```bash
docker compose -f deploy/chroma/compose.yaml up -d
curl -fsS http://localhost:8000/api/v2/heartbeat
```

El servicio persiste en `bbdd-vectoriales-s03-chroma-data`. Para experimentos pequeños, `chromadb.PersistentClient(path=...)` evita Docker y conserva el estado en una carpeta. `EphemeralClient` sirve para tests, no para demostrar un proceso independiente.

## Weaviate 1.38.2

```bash
docker compose -f deploy/weaviate/compose.yaml up -d
curl -fsS http://localhost:8080/v1/.well-known/ready
```

Se exponen REST en 8080 y gRPC en 50051. El despliegue desactiva módulos de vectorización porque los embeddings ya existen: el ejercicio es BYOV (*bring your own vectors*). El acceso anónimo solo es aceptable en este entorno local aislado; un despliegue real requiere autenticación y autorización.

## Milvus Standalone 2.6.18

```bash
docker compose -f deploy/milvus/compose.yaml up -d
curl -fsS http://localhost:9091/healthz
```

Aunque se denomine *Standalone*, intervienen tres servicios: Milvus, etcd para metadatos y MinIO para objetos. Ese hecho es parte de la comparación operativa. Para desarrollo puede utilizarse Milvus Lite mediante `MilvusClient("archivo.db")`; comparte buena parte de la API, pero no reproduce la topología ni el comportamiento distribuido.

## Qdrant 1.18.2

```bash
docker compose -f deploy/qdrant/compose.yaml up -d
curl -fsS http://localhost:6333/healthz
```

HTTP se expone en 6333 y gRPC en 6334. El notebook crea un índice de payload para `metadata.brand`; un filtro correcto no basta para garantizar un filtrado eficiente cuando el campo carece del índice adecuado. Para tests, `QdrantClient(":memory:")` conserva todo en el proceso.

## Pinecone Local

```bash
docker compose -f deploy/pinecone-local/compose.yaml up -d
```

Pinecone Local es un emulador en memoria para desarrollo. No persiste datos al reiniciar, limita cada índice a 100.000 registros y no implementa el contrato operativo completo del servicio cloud. La imagen oficial se publica mediante una etiqueta móvil `latest`; por eso este compose documenta la excepción aunque los demás motores estén fijados por versión. En Apple Silicon se ejecuta como `linux/amd64`.

El notebook principal de Pinecone utiliza Cloud. Para experimentar con Local, configura el SDK según la guía oficial y crea el índice contra el control plane en `localhost:5080`. No presentes su latencia como predicción de Pinecone Serverless.

