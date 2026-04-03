---
name: search-patterns
description: Implement search infrastructure with Elasticsearch, OpenSearch, and Typesense including full-text search, faceted navigation, autocomplete, relevance tuning, and index management. Use when building search features, configuring analyzers, or optimizing search relevance.
---

# Search Patterns

## Expert Agent

For search architecture, index design, and relevance optimization, delegate to:

- **`software-architect`**: Designs search infrastructure with index strategies and service integration.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Technology Selection

| Feature | Elasticsearch | OpenSearch | Typesense |
|---------|--------------|------------|-----------|
| License | Elastic License | Apache 2.0 | GPL-3.0 |
| Managed | Elastic Cloud | AWS OpenSearch | Typesense Cloud |
| Query DSL | Full JSON DSL | Full JSON DSL | Simple API |
| Complexity | High | High | Low |
| Best for | Complex search, analytics | AWS-native, log analytics | Simple search, typo tolerance |


## Index Design

### Mapping Definition (Elasticsearch)

```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "english",
        "fields": {
          "keyword": { "type": "keyword" },
          "suggest": {
            "type": "completion",
            "analyzer": "simple"
          }
        }
      },
      "description": {
        "type": "text",
        "analyzer": "english"
      },
      "category": {
        "type": "keyword"
      },
      "price": {
        "type": "float"
      },
      "tags": {
        "type": "keyword"
      },
      "created_at": {
        "type": "date"
      },
      "location": {
        "type": "geo_point"
      }
    }
  }
}
```

### Field Type Selection

| Data | Type | Use Case |
|------|------|----------|
| Searchable text | `text` | Full-text search with analysis |
| Exact match | `keyword` | Filters, aggregations, sorting |
| Numbers | `integer`, `float` | Range queries, sorting |
| Dates | `date` | Date range queries, histograms |
| Boolean | `boolean` | Filters |
| Nested objects | `nested` | Array of objects with independent queries |


## Full-Text Search Queries

### Multi-Match with Boosting

```json
{
  "query": {
    "bool": {
      "must": {
        "multi_match": {
          "query": "wireless headphones",
          "fields": ["title^3", "description", "tags^2"],
          "type": "best_fields",
          "fuzziness": "AUTO"
        }
      },
      "filter": [
        { "term": { "category": "electronics" } },
        { "range": { "price": { "gte": 20, "lte": 200 } } },
        { "term": { "in_stock": true } }
      ],
      "should": [
        { "term": { "featured": { "value": true, "boost": 2 } } }
      ]
    }
  }
}
```

### Query Types

| Query | Use Case | Example |
|-------|----------|---------|
| `match` | Full-text with analysis | Natural language search |
| `term` | Exact keyword match | Category filters |
| `range` | Numeric/date ranges | Price, date filters |
| `bool` | Combine queries | Complex search with filters |
| `multi_match` | Search across fields | Unified search bar |
| `fuzzy` | Typo tolerance | Misspelling correction |


## Faceted Navigation

```json
{
  "size": 20,
  "query": { "match": { "description": "laptop" } },
  "aggs": {
    "categories": {
      "terms": { "field": "category", "size": 10 }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 500 },
          { "from": 500, "to": 1000 },
          { "from": 1000, "to": 2000 },
          { "from": 2000 }
        ]
      }
    },
    "brands": {
      "terms": { "field": "brand", "size": 20 }
    },
    "avg_price": {
      "avg": { "field": "price" }
    }
  }
}
```


## Index Lifecycle Management

| Phase | Age | Action |
|-------|-----|--------|
| Hot | 0-7 days | Full indexing, primary search |
| Warm | 7-30 days | Read-only, reduced replicas |
| Cold | 30-90 days | Frozen, minimal resources |
| Delete | 90+ days | Remove index |


## Design Checklist

- [ ] Mappings defined explicitly (no dynamic mapping in production)
- [ ] Analyzers configured for language and domain
- [ ] Search queries use `bool` with filters for non-scored criteria
- [ ] Facets implemented for key filterable dimensions
- [ ] Autocomplete configured with completion suggester
- [ ] Relevance tuned with field boosting and function scores
- [ ] Index lifecycle policy configured
- [ ] Pagination uses `search_after` for deep results
- [ ] Monitoring on query latency, indexing rate, and cluster health
