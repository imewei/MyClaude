---
name: knowledge-graph-patterns
description: "Build knowledge graphs and ontologies for AI systems including entity-relationship modeling, graph databases (Neo4j), semantic search, ontology design, and knowledge-enhanced retrieval. Use when implementing knowledge graphs, building semantic search, or designing entity models for AI applications."
---

# Knowledge Graph Patterns

## Expert Agent

For knowledge graph design, ontology engineering, and knowledge-enhanced AI systems, delegate to:

- **`context-specialist`**: Manages context retrieval, information synthesis, knowledge graphs, and memory systems.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`

## Graph Database Fundamentals

### Neo4j (Cypher)

```cypher
CREATE (p:Person {name: 'Alice', role: 'researcher'})
CREATE (t:Topic {name: 'Machine Learning', domain: 'CS'})
CREATE (p)-[:STUDIES {since: 2020}]->(t)

-- Traversal: find 2-hop collaborators
MATCH (p:Person)-[:STUDIES]->(t:Topic)<-[:STUDIES]-(c:Person)
WHERE p.name = 'Alice' AND c <> p
RETURN c.name, t.name

-- Shortest path
MATCH path = shortestPath((a:Person {name: 'Alice'})-[*..6]-(b:Person {name: 'Bob'}))
RETURN path
```

### Database Selection

| Feature | Neo4j | ArangoDB | Amazon Neptune |
|---------|-------|----------|----------------|
| Model | Property graph | Multi-model | Property + RDF |
| Query | Cypher | AQL | Gremlin / SPARQL |
| Best for | Complex traversals | Document+graph | Managed cloud |

## Ontology Design (OWL/RDF)

```turtle
@prefix ex: <http://example.org/ontology#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

ex:ResearchPaper a owl:Class ; rdfs:label "Research Paper" .
ex:Author a owl:Class ; rdfs:label "Author" .
ex:hasAuthor a owl:ObjectProperty ;
    rdfs:domain ex:ResearchPaper ; rdfs:range ex:Author .
```

- Define clear class hierarchy (avoid nesting > 5 levels)
- Specify domain and range for all properties
- Add cardinality constraints where applicable
- Validate with a reasoner (HermiT, Pellet) before deployment

## Entity Extraction

```python
import spacy

nlp = spacy.load("en_core_web_trf")

def extract_entities_and_relations(text: str) -> dict:
    """Extract entities and co-occurrence relation candidates."""
    doc = nlp(text)
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char}
        for ent in doc.ents
    ]
    relations = []
    for i, a in enumerate(doc.ents):
        for b in doc.ents[i + 1:]:
            if b.start - a.end < 50:
                relations.append({"source": a.text, "target": b.text})
    return {"entities": entities, "relations": relations}
```

## Knowledge-Enhanced RAG

1. **Index**: Extract entities from documents, build knowledge graph
2. **Query**: Extract entities from query, traverse graph for context
3. **Augment**: Combine graph context with vector-retrieved chunks
4. **Generate**: LLM produces answer with structured knowledge

```python
from neo4j import GraphDatabase

class KnowledgeRAG:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, vector_store):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        entities = extract_entities_and_relations(query)["entities"]
        graph_ctx = []
        with self.driver.session() as session:
            for ent in entities:
                result = session.run(
                    "MATCH (n)-[r*1..2]-(m) WHERE n.name =~ $pat "
                    "RETURN n.name, type(r[0]), m.name LIMIT 20",
                    pat=f"(?i).*{ent['text']}.*",
                )
                graph_ctx.extend(result.data())
        return {"vector": vector_results, "graph": graph_ctx}
```

| Strategy | When to Use | Tradeoff |
|----------|------------|----------|
| Graph-first | Structured queries (who, what, when) | High precision, low recall |
| Vector-first | Semantic similarity queries | High recall, lower precision |
| Hybrid re-rank | General QA | Balanced, higher latency |

## Graph Traversal Patterns

```cypher
-- Community detection (Louvain)
CALL gds.louvain.stream('kg', {relationshipWeightProperty: 'weight'})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS entity, communityId

-- Neighborhood expansion
MATCH (start:Entity {name: $name})
CALL apoc.path.expandConfig(start, {maxLevel: 3, uniqueness: "NODE_GLOBAL"})
YIELD path
RETURN nodes(path), relationships(path), length(path) AS depth
ORDER BY depth ASC
```

## Production Checklist

- [ ] Define node labels, relationship types, and property constraints
- [ ] Create indexes on frequently queried properties
- [ ] Deduplicate entities with fuzzy matching before ingestion
- [ ] Track graph schema versions alongside application versions
- [ ] Monitor query latency, graph size, and traversal depth
- [ ] Schedule regular database backups
- [ ] Validate graph integrity with property-based tests
