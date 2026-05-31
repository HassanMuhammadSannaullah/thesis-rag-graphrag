"""
Simple local vector index using numpy cosine similarity.
Stores embeddings + metadata to disk for resumable indexing.
"""
import json, numpy as np
from pathlib import Path
from src.utils.model_client import embed_texts


class LocalVectorIndex:
    """Flat vector index with caching to disk."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict] = []
        self._load_if_exists()

    def _load_if_exists(self):
        emb_file = self.index_path / "embeddings.npy"
        meta_file = self.index_path / "metadata.jsonl"
        if emb_file.exists() and meta_file.exists():
            self.embeddings = np.load(str(emb_file))
            self.metadata = []
            with open(meta_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))
            print(f"  Loaded index: {len(self.metadata)} vectors from {self.index_path}")

    def save(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(str(self.index_path / "embeddings.npy"), self.embeddings)
        with open(self.index_path / "metadata.jsonl", "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  Saved index: {len(self.metadata)} vectors to {self.index_path}")

    def add(self, texts: list[str], metadatas: list[dict], batch_size: int = 5):
        """Embed and add texts + metadata. Skips already-indexed items."""
        # Check what's already indexed
        existing_ids = {m["id"] for m in self.metadata}
        new_texts = []
        new_metas = []
        for text, meta in zip(texts, metadatas):
            if meta["id"] not in existing_ids:
                new_texts.append(text)
                new_metas.append(meta)

        if not new_texts:
            print(f"  All {len(texts)} items already indexed.")
            return

        print(f"  Indexing {len(new_texts)} new items (skipping {len(texts) - len(new_texts)} existing) ...")

        all_vecs = []
        for i in range(0, len(new_texts), batch_size):
            batch = new_texts[i:i + batch_size]
            # Truncate very long texts for embedding
            batch_truncated = [t[:2000] for t in batch]
            vecs = embed_texts(batch_truncated)
            all_vecs.extend(vecs)
            print(f"    Embedded {min(i + batch_size, len(new_texts))}/{len(new_texts)}")

        new_emb = np.array(all_vecs, dtype=np.float32)

        if self.embeddings is not None and len(self.embeddings) > 0:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        else:
            self.embeddings = new_emb

        self.metadata.extend(new_metas)
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index with a query string. Returns top-k results."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        q_vec = np.array(embed_texts([query[:2000]])[0], dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self.embeddings / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        scores = normed @ q_norm

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            result = dict(self.metadata[idx])
            result["score"] = float(scores[idx])
            results.append(result)
        return results

    @property
    def size(self) -> int:
        return len(self.metadata)
