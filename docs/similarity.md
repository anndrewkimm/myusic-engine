# Weighted multi-seed similarity

The local MVP uses exact cosine search over one explicitly selected embedding definition. It is
small and dependency-free so retrieval behavior can be validated with synthetic vectors before
an embedding model or lawful audio collection is available.

For normalized seed embeddings `e_i` and positive weights `w_i`, the query is:

```text
query = normalize(sum(w_i * normalize(e_i)))
```

The implementation normalizes stored candidates too, ranks by cosine similarity, excludes seed
tracks by default, and uses track ID as a deterministic tie-breaker. A later vector index can
replace the exact scan without changing query semantics.

## Provenance selection

An index requires an `EmbeddingSpec` containing all of:

- a versioned feature name;
- a feature source;
- a source version;
- the expected vector dimensions.

Only observations matching that full definition enter the index. Conflicting sources are never
averaged or silently substituted.

## Filters

Numeric ranges and categorical allowlists select their own exact feature provenance. Filters are
inclusive and required by default. A candidate with a missing or low-confidence required feature
is excluded. Optional filters allow missing features but still reject an observed value outside
the requested condition. Each returned match includes the value, provenance selector, and
confidence that let it pass.

This stage does not claim that synthetic vectors are meaningful audio embeddings. Real retrieval
quality cannot be evaluated until licensed audio or a permitted source supplies actual features.
