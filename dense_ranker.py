from sentence_transformers import SentenceTransformer, models
import faiss
import numpy as np


def build_model(model_name):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)
    
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model 
    

class DenseRanker:
    def __init__(self, model: SentenceTransformer, index=None):
        self.model = model
        self.index = index if index is not None else faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        self.documents = {}
        self.doc_ids = []

    @classmethod
    def fit(cls, corpus, model: SentenceTransformer, index=None):
        """Class method to fit a corpus and return a DenseRanker instance."""
        instance = cls(model, index)
        instance.add_documents(corpus)
        return instance

    def add_documents(self, docs):
        """Add documents to the index. Docs should be a dictionary {docid: doc}."""
        self.documents.update(docs)
        self.doc_ids.extend(docs.keys())
        doc_embeddings = self.model.encode(list(docs.values()), convert_to_tensor=True, device=self.model.device).cpu().detach().numpy()
        
        self.index.add(doc_embeddings)
        
    def search(self, queries, top_k=5, return_text=False):
        """Search for the top_k most similar documents to the batch of queries.
            
            Return :[
                        [
                            {
                                "docid": 1,
                                "score": 2,
                                "text": 3
                            },
                        ],
                    ]
        """
        query_embeddings = self.model.encode(queries, convert_to_tensor=True, device=self.model.device).cpu().detach().numpy()
        D, I = self.index.search(query_embeddings, top_k)
        
        results = []
        for query_idx in range(len(queries)):
            query_results = []
            for rank, (doc_idx, score) in enumerate(zip(I[query_idx], D[query_idx])):
                docid = self.doc_ids[doc_idx]
                result = {'docid': docid, 'score': score}
                if return_text:
                    result['text'] = self.documents[docid]
                query_results.append(result)
            results.append(query_results)
        
        return results

    def search_one(self, query, top_k=5, return_text=False):
        """Search for the top_k most similar documents to a single query."""
        return self.search([query], top_k, return_text)[0]

# Example usage
if __name__ == "__main__":
    # Initialize SentenceTransformer model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cuda')

    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "A quick brown fox.",
        "doc3": "The lazy dog.",
        "doc4": "A fast fox and a quick dog.",
        "doc5": "An agile brown fox."
    }

    # Fit the ranker with the corpus
    ranker = DenseRanker.fit(docs, model)

    # Batch search with text return
    queries = ["quick fox", "lazy dog"]
    results = ranker.search(queries, top_k=3, return_text=True)
    for query, query_results in zip(queries, results):
        print(f"Query: {query}")
        for result in query_results:
            print(f"  DocID: {result['docid']}, Score: {result['score']}, Text: {result['text']}")

    # Single search without text return
    query = "quick fox"
    results = ranker.search_one(query, top_k=3, return_text=False)
    print(f"Query: {query}")
    for result in results:
        print(f"  DocID: {result['docid']}, Score: {result['score']}")