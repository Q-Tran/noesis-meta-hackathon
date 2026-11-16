"""Milvus connection and indexing helpers."""
from pymilvus import MilvusClient, DataType
from config import settings

class VectorStore:
    def _shape_fields(self) -> list[str]:
        """The vector collections the installed profile's shapes require.

        Each `ExtractionShape` with a `fuzzy_text` merges by embedding similarity
        in the collection named after its field, so the collection set is a
        property of the declared shapes and cannot be kept as a constant here. A
        shape whose field is absent from Milvus fails every fuzzy merge on that
        field -- `claims` was missing from the list below, so extracting claims
        raised `collection not found` and the ingest wrote no claims.
        """
        from knowledge_bases.registry import extraction_shapes

        return sorted({s.field for s in extraction_shapes() if s.fuzzy_text})

    COLLECTIONS = [
        "chunks",
        "bmad_entities",
        # Skill/Expert catalog embeddings (see knowledge_bases/tools/catalog_index.py)
        "noesis_skills", "noesis_experts",
        # Development KB code entities (see knowledge_bases/development/persist.py)
        "code_entities",
        # Architecture entities are plain components/interfaces, not managed
        # revisions; they get their own collection rather than pretending to
        # be BmadEntity rows.
        "architecture_entities",
        # Metric definitions/calculations for semantic metric search.
        "metric_entities",
    ]

    def __init__(self, timeout: float | None = None):
        # Bounded connect: Milvus is strictly an accelerator (every caller
        # degrades to keyword-only without it -- see
        # `knowledge_bases/tools/catalog_index.py`), so an unreachable server
        # must fail fast rather than stall the write that triggered the
        # reindex. Without this, pymilvus blocks ~10s per attempt, which is
        # paid inline by every catalog/expert write when Milvus is down.
        self.client = MilvusClient(
            uri=settings.milvus_uri,
            timeout=settings.milvus_timeout if timeout is None else timeout,
        )

    def ensure_collections(self):
        dim = settings.embedding_dimension

        for coll_name in sorted(set(self.COLLECTIONS) | set(self._shape_fields())):
            if not self.client.has_collection(collection_name=coll_name):
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )
                # 512: development code-entity ids are `repo:path:qualname`
                # and can run long; existing collections keep 128.
                schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
                schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
                
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="embedding", 
                    index_type="DISKANN",
                    metric_type="COSINE"
                )
                
                self.client.create_collection(
                    collection_name=coll_name,
                    schema=schema,
                    index_params=index_params
                )

    def upsert_vector(self, collection_name: str, id: str, embedding: list[float], metadata: dict | None = None):
        data = {"id": id, "embedding": embedding}
        if metadata:
            data.update(metadata)
        self.client.upsert(collection_name=collection_name, data=[data])

    def search_vectors(self, collection_name: str, embedding: list[float], top_k: int = 5):
        try:
            res = self.client.search(
                collection_name=collection_name,
                data=[embedding],
                limit=top_k,
                output_fields=["*"]
            )
        except Exception as e:
            print(f"DEBUG: Vector search failed for {collection_name}: {e}")
            return []
            
        if not res or not res[0]:
            return []
        
        # Format similar to Neo4j vector search return
        results = []
        for hit in res[0]:
            r = {
                "id": hit["id"],
                "similarity": hit["distance"] # cosine
            }
            if hit.get("entity"):
                for k, v in hit["entity"].items():
                    r[k] = v
            results.append(r)
            
        return results

    def upsert_chunk(self, id: str, embedding: list[float], metadata: dict | None = None):
        self.upsert_vector("chunks", id, embedding, metadata)

    def search_chunks(self, embedding: list[float], top_k: int = 5):
        return self.search_vectors("chunks", embedding, top_k)

    def upsert_claim(self, id: str, embedding: list[float], metadata: dict | None = None):
        self.upsert_vector("claims", id, embedding, metadata)

    def search_claims(self, embedding: list[float], top_k: int = 5):
        return self.search_vectors("claims", embedding, top_k)

    def upsert_business_term(self, id: str, embedding: list[float], metadata: dict | None = None):
        self.upsert_vector("business_terms", id, embedding, metadata)

    def search_business_terms(self, embedding: list[float], top_k: int = 10):
        return self.search_vectors("business_terms", embedding, top_k)

    def delete_vector(self, collection_name: str, id: str) -> None:
        self.client.delete(collection_name=collection_name, ids=[id])

    def wipe(self):
        for coll_name in self.COLLECTIONS:
            if self.client.has_collection(coll_name):
                self.client.drop_collection(coll_name)
