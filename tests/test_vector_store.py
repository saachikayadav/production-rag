from demand_lens.vector_store import LocalVectorStore, PineconeVectorStore


DOCUMENTS = [
    {"id": "c1", "source_id": "s1", "title": "Forecast", "section": "Bias", "content": "Forecast bias measures systematic over forecasting."},
    {"id": "c2", "source_id": "s2", "title": "Inventory", "section": "Risk", "content": "Inventory shortage risk depends on safety stock."},
]


def test_local_vector_store_is_deterministic():
    store = LocalVectorStore()
    first = store.search("workspace", "forecast bias", DOCUMENTS, 2)
    second = store.search("workspace", "forecast bias", DOCUMENTS, 2)
    assert first == second
    assert first[0].chunk_id == "c1"


class FakeResponse:
    def to_dict(self):
        return {"result": {"hits": [{"_id": "c1", "_score": 0.91}]}}


class FakeIndex:
    def __init__(self):
        self.upserts = []
        self.searches = []

    def upsert_records(self, namespace, records):
        self.upserts.append((namespace, records))

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return FakeResponse()


def test_pinecone_provider_uses_integrated_embedding_record_api():
    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.index = FakeIndex()
    store.text_field = "chunk_text"
    store.upsert("workspace-1", DOCUMENTS)
    namespace, records = store.index.upserts[0]
    assert namespace == "workspace-1"
    assert records[0]["_id"] == "c1"
    assert records[0]["chunk_text"] == DOCUMENTS[0]["content"]
    matches = store.search("workspace-1", "forecast bias", DOCUMENTS, 5)
    assert matches[0].chunk_id == "c1"
    assert store.index.searches[0]["query"]["inputs"]["text"] == "forecast bias"
