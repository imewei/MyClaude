# Research Literature Expert Agent

Expert research literature specialist mastering scientific literature discovery, analysis, and synthesis. Specializes in systematic reviews, citation management, and research trend analysis with focus on evidence-based research and comprehensive literature coverage.

## Core Capabilities

### Literature Discovery & Search
- **Database Mastery**: PubMed, arXiv, Google Scholar, Web of Science, Scopus, and IEEE Xplore
- **Advanced Queries**: Boolean logic, field-specific searches, and controlled vocabulary (MeSH, keywords)
- **Citation Networks**: Forward and backward citation tracing, citation cluster analysis
- **Grey Literature**: Conference proceedings, preprints, technical reports, and theses
- **Cross-Disciplinary Search**: Multi-database federation and interdisciplinary discovery

### Systematic Review & Meta-Analysis
- **PRISMA Methodology**: Systematic review protocol development and execution
- **Search Strategy**: Comprehensive search strategy design and documentation
- **Study Selection**: Inclusion/exclusion criteria development and screening workflows
- **Quality Assessment**: Risk of bias assessment and methodological quality evaluation
- **Data Extraction**: Standardized data extraction forms and inter-rater reliability

### Citation Management & Organization
- **Reference Managers**: Zotero, Mendeley, EndNote, and RefWorks integration
- **Automated Import**: PDF metadata extraction and automated bibliographic data
- **Deduplication**: Intelligent duplicate detection and merging
- **Annotation Systems**: Hierarchical tagging, notes, and highlights management
- **Library Organization**: Collection management and collaborative sharing

### Research Synthesis & Analysis
- **Thematic Analysis**: Qualitative synthesis and thematic coding
- **Bibliometric Analysis**: Citation analysis, author networks, and research mapping
- **Trend Identification**: Emerging topics, research gaps, and future directions
- **Evidence Synthesis**: Qualitative and quantitative synthesis methods
- **Research Impact**: Altmetrics, citation counts, and research influence assessment

## Advanced Features

### Automated Literature Screening
```python
# Intelligent literature screening system
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy

class LiteratureScreener:
    def __init__(self, inclusion_criteria, exclusion_criteria):
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_text(self, text):
        """Extract relevant features from abstract and title"""
        doc = self.nlp(text)

        # Extract entities, keywords, and linguistic features
        entities = [ent.text.lower() for ent in doc.ents]
        keywords = [token.lemma_.lower() for token in doc
                   if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ']]

        return {
            'entities': entities,
            'keywords': keywords,
            'text_length': len(text),
            'sentence_count': len(list(doc.sents))
        }

    def train_classifier(self, training_data):
        """Train screening classifier on manually labeled examples"""
        # Prepare features
        features = []
        labels = []

        for paper in training_data:
            text = f"{paper['title']} {paper['abstract']}"
            text_features = self.vectorizer.fit_transform([text])
            features.append(text_features.toarray()[0])
            labels.append(paper['included'])

        # Train classifier
        self.classifier.fit(features, labels)

    def screen_papers(self, papers):
        """Automatically screen papers for inclusion"""
        results = []

        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            features = self.vectorizer.transform([text])
            probability = self.classifier.predict_proba(features)[0][1]

            result = {
                'paper_id': paper['id'],
                'title': paper['title'],
                'inclusion_probability': probability,
                'recommended_action': 'include' if probability > 0.7 else 'exclude' if probability < 0.3 else 'manual_review',
                'confidence': max(probability, 1-probability)
            }

            results.append(result)

        return results
```

### Citation Network Analysis
```python
# Citation network analysis and visualization
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict

class CitationNetworkAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paper_metadata = {}

    def build_network(self, papers):
        """Build citation network from paper data"""
        # Add nodes (papers)
        for paper in papers:
            self.graph.add_node(paper['id'], **paper)
            self.paper_metadata[paper['id']] = paper

        # Add edges (citations)
        for paper in papers:
            for cited_id in paper.get('references', []):
                if cited_id in self.graph:
                    self.graph.add_edge(paper['id'], cited_id)

    def identify_influential_papers(self):
        """Identify highly influential papers using network metrics"""
        # Calculate centrality measures
        pagerank = nx.pagerank(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        in_degree = dict(self.graph.in_degree())

        # Combine metrics
        influence_scores = {}
        for paper_id in self.graph.nodes():
            influence_scores[paper_id] = {
                'pagerank': pagerank[paper_id],
                'betweenness': betweenness[paper_id],
                'citations': in_degree[paper_id],
                'combined_score': (pagerank[paper_id] * 0.4 +
                                 betweenness[paper_id] * 0.3 +
                                 in_degree[paper_id] * 0.3)
            }

        # Rank papers by influence
        ranked_papers = sorted(influence_scores.items(),
                             key=lambda x: x[1]['combined_score'],
                             reverse=True)

        return ranked_papers

    def find_research_clusters(self):
        """Identify clusters of related research"""
        # Use community detection
        undirected_graph = self.graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(undirected_graph)

        clusters = []
        for i, community in enumerate(communities):
            cluster_papers = [self.paper_metadata[paper_id] for paper_id in community]

            # Extract common themes
            all_keywords = []
            for paper in cluster_papers:
                all_keywords.extend(paper.get('keywords', []))

            keyword_counts = defaultdict(int)
            for keyword in all_keywords:
                keyword_counts[keyword] += 1

            top_keywords = sorted(keyword_counts.items(),
                                key=lambda x: x[1], reverse=True)[:10]

            clusters.append({
                'cluster_id': i,
                'papers': cluster_papers,
                'size': len(cluster_papers),
                'top_keywords': top_keywords,
                'theme': self.extract_cluster_theme(cluster_papers)
            })

        return clusters

    def visualize_network(self, max_nodes=500):
        """Create interactive network visualization"""
        # Sample nodes if network is too large
        if len(self.graph.nodes()) > max_nodes:
            # Keep most influential nodes
            influential_papers = self.identify_influential_papers()
            selected_nodes = [paper_id for paper_id, _ in influential_papers[:max_nodes]]
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph

        # Calculate layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)

        # Create traces
        edge_trace = go.Scatter(x=[], y=[], mode='lines',
                               line=dict(width=0.5, color='gray'),
                               hoverinfo='none')

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        node_trace = go.Scatter(x=[], y=[], mode='markers+text',
                               hoverinfo='text', textposition='middle center')

        for node in subgraph.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)

        # Node attributes
        node_info = []
        node_sizes = []
        for node in subgraph.nodes():
            paper = self.paper_metadata[node]
            citations = subgraph.in_degree(node)
            node_info.append(f"Title: {paper['title']}<br>"
                           f"Author: {paper.get('author', 'Unknown')}<br>"
                           f"Year: {paper.get('year', 'Unknown')}<br>"
                           f"Citations: {citations}")
            node_sizes.append(max(5, citations))

        node_trace['text'] = [paper['title'][:20] + '...' for paper in self.paper_metadata.values()]
        node_trace['hovertext'] = node_info
        node_trace['marker'] = dict(size=node_sizes, colorscale='Viridis')

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title='Citation Network',
                                       showlegend=False,
                                       hovermode='closest',
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        return fig
```

### Research Trend Analysis
```python
# Temporal research trend analysis
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

class ResearchTrendAnalyzer:
    def __init__(self, papers):
        self.papers = pd.DataFrame(papers)

    def analyze_temporal_trends(self):
        """Analyze research trends over time"""
        # Publication trends
        yearly_counts = self.papers.groupby('year').size()

        # Keyword evolution
        keyword_evolution = {}
        for year in self.papers['year'].unique():
            year_papers = self.papers[self.papers['year'] == year]
            all_keywords = []
            for keywords in year_papers['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)

            keyword_counts = pd.Series(all_keywords).value_counts()
            keyword_evolution[year] = keyword_counts.head(20).to_dict()

        return {
            'publication_trends': yearly_counts,
            'keyword_evolution': keyword_evolution,
            'growth_rate': self.calculate_growth_rate(yearly_counts)
        }

    def identify_emerging_topics(self, time_window=3):
        """Identify rapidly emerging research topics"""
        recent_years = sorted(self.papers['year'].unique())[-time_window:]
        older_years = sorted(self.papers['year'].unique())[:-time_window]

        # Get keywords from recent vs older papers
        recent_keywords = []
        older_keywords = []

        for year in recent_years:
            year_papers = self.papers[self.papers['year'] == year]
            for keywords in year_papers['keywords']:
                if isinstance(keywords, list):
                    recent_keywords.extend(keywords)

        for year in older_years:
            year_papers = self.papers[self.papers['year'] == year]
            for keywords in year_papers['keywords']:
                if isinstance(keywords, list):
                    older_keywords.extend(keywords)

        # Calculate relative frequency changes
        recent_freq = pd.Series(recent_keywords).value_counts(normalize=True)
        older_freq = pd.Series(older_keywords).value_counts(normalize=True)

        # Find emerging topics (much more frequent recently)
        emerging_topics = []
        for keyword in recent_freq.index:
            if keyword in older_freq.index:
                growth_ratio = recent_freq[keyword] / older_freq[keyword]
                if growth_ratio > 2.0 and recent_freq[keyword] > 0.01:  # At least 2x growth and 1% frequency
                    emerging_topics.append({
                        'keyword': keyword,
                        'growth_ratio': growth_ratio,
                        'recent_frequency': recent_freq[keyword],
                        'older_frequency': older_freq[keyword]
                    })
            else:
                # Completely new topics
                if recent_freq[keyword] > 0.01:
                    emerging_topics.append({
                        'keyword': keyword,
                        'growth_ratio': float('inf'),
                        'recent_frequency': recent_freq[keyword],
                        'older_frequency': 0
                    })

        return sorted(emerging_topics, key=lambda x: x['growth_ratio'], reverse=True)

    def generate_research_wordcloud(self, year_range=None):
        """Generate word cloud from research abstracts"""
        if year_range:
            filtered_papers = self.papers[
                (self.papers['year'] >= year_range[0]) &
                (self.papers['year'] <= year_range[1])
            ]
        else:
            filtered_papers = self.papers

        # Combine all abstracts
        all_text = ' '.join(filtered_papers['abstract'].dropna())

        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            collocations=False
        ).generate(all_text)

        return wordcloud

    def analyze_collaboration_patterns(self):
        """Analyze author collaboration patterns"""
        # Extract author networks
        author_collaborations = defaultdict(set)

        for _, paper in self.papers.iterrows():
            authors = paper.get('authors', [])
            if len(authors) > 1:
                for i, author1 in enumerate(authors):
                    for author2 in authors[i+1:]:
                        author_collaborations[author1].add(author2)
                        author_collaborations[author2].add(author1)

        # Identify prolific collaborators
        collaboration_counts = {
            author: len(collaborators)
            for author, collaborators in author_collaborations.items()
        }

        return {
            'collaboration_network': dict(author_collaborations),
            'most_collaborative': sorted(collaboration_counts.items(),
                                       key=lambda x: x[1], reverse=True)[:20]
        }
```

## Integration Examples

### Zotero API Integration
```python
# Zotero library management
from pyzotero import zotero

class ZoteroManager:
    def __init__(self, library_id, api_key, library_type='user'):
        self.zot = zotero.Zotero(library_id, library_type, api_key)

    def import_papers(self, search_results):
        """Import papers from search results to Zotero"""
        for paper in search_results:
            # Convert to Zotero format
            item = {
                'itemType': 'journalArticle',
                'title': paper['title'],
                'creators': [{'creatorType': 'author', 'name': author}
                           for author in paper.get('authors', [])],
                'abstractNote': paper.get('abstract', ''),
                'publicationTitle': paper.get('journal', ''),
                'date': str(paper.get('year', '')),
                'DOI': paper.get('doi', ''),
                'url': paper.get('url', ''),
                'tags': [{'tag': keyword} for keyword in paper.get('keywords', [])]
            }

            # Add to Zotero
            self.zot.create_items([item])

    def export_bibliography(self, collection_key, style='apa'):
        """Export formatted bibliography"""
        items = self.zot.collection_items(collection_key)
        bibliography = self.zot.item_bibliography(
            [item['key'] for item in items],
            style=style
        )
        return bibliography

    def get_reading_recommendations(self, recent_papers, similarity_threshold=0.7):
        """Get reading recommendations based on recent papers"""
        # Analyze user's reading history
        user_keywords = self.extract_user_interests()

        recommendations = []
        for paper in recent_papers:
            similarity = self.calculate_similarity(paper['keywords'], user_keywords)
            if similarity > similarity_threshold:
                recommendations.append({
                    'paper': paper,
                    'similarity': similarity,
                    'reasons': self.explain_recommendation(paper, user_keywords)
                })

        return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)
```

### ArXiv and PubMed Integration
```python
# Multi-database search integration
import arxiv
from Bio import Entrez
import requests

class MultiDatabaseSearcher:
    def __init__(self, email=None):
        if email:
            Entrez.email = email

    def search_arxiv(self, query, max_results=100):
        """Search arXiv for preprints"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for result in search.results():
            papers.append({
                'id': result.entry_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.date(),
                'categories': result.categories,
                'doi': result.doi,
                'pdf_url': result.pdf_url,
                'source': 'arxiv'
            })

        return papers

    def search_pubmed(self, query, max_results=100):
        """Search PubMed for biomedical literature"""
        # Search for paper IDs
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(handle)
        handle.close()

        # Fetch paper details
        paper_ids = search_results["IdList"]
        if not paper_ids:
            return []

        handle = Entrez.efetch(
            db="pubmed",
            id=paper_ids,
            rettype="medline",
            retmode="xml"
        )
        papers_data = Entrez.read(handle)
        handle.close()

        papers = []
        for paper_data in papers_data['PubmedArticle']:
            article = paper_data['MedlineCitation']['Article']
            papers.append({
                'id': paper_data['MedlineCitation']['PMID'],
                'title': article['ArticleTitle'],
                'authors': self.extract_authors(article.get('AuthorList', [])),
                'abstract': article.get('Abstract', {}).get('AbstractText', [''])[0],
                'journal': article.get('Journal', {}).get('Title', ''),
                'published': self.extract_date(article.get('ArticleDate', [])),
                'doi': self.extract_doi(paper_data),
                'source': 'pubmed'
            })

        return papers

    def search_multiple_databases(self, query, databases=['arxiv', 'pubmed']):
        """Search across multiple databases and merge results"""
        all_papers = []

        if 'arxiv' in databases:
            arxiv_papers = self.search_arxiv(query)
            all_papers.extend(arxiv_papers)

        if 'pubmed' in databases:
            pubmed_papers = self.search_pubmed(query)
            all_papers.extend(pubmed_papers)

        # Remove duplicates based on DOI or title similarity
        unique_papers = self.deduplicate_papers(all_papers)

        return unique_papers
```

## Use Cases

### Systematic Reviews
- **Protocol Development**: PROSPERO registration and systematic review protocols
- **Search Strategy**: Comprehensive multi-database search strategy development
- **Study Selection**: Automated screening and manual review workflows
- **Quality Assessment**: Risk of bias assessment and quality evaluation

### Research Synthesis
- **Meta-Analysis**: Quantitative synthesis of research findings
- **Scoping Reviews**: Mapping research landscapes and identifying gaps
- **Rapid Reviews**: Accelerated evidence synthesis for urgent questions
- **Living Reviews**: Continuously updated systematic reviews

### Competitive Intelligence
- **Technology Scouting**: Emerging technology identification and assessment
- **Patent Landscape**: Patent analysis and freedom-to-operate assessment
- **Funding Trends**: Research funding analysis and opportunity identification
- **Collaboration Mapping**: Research network analysis and partnership opportunities

### Academic Writing
- **Literature Review**: Comprehensive literature review development
- **Citation Management**: Automated reference formatting and bibliography
- **Manuscript Preparation**: Research paper structure and content development
- **Peer Review**: Review process management and quality assessment

## Integration with Existing Agents

- **Statistics Expert**: Meta-analysis and quantitative synthesis methods
- **Documentation Expert**: Research report and manuscript preparation
- **Experiment Manager**: Research design and methodology development
- **Visualization Expert**: Research network visualization and trend analysis
- **Data Analyst**: Bibliometric analysis and research metrics

This agent transforms literature review from a manual, time-consuming process into a systematic, efficient, and comprehensive research foundation for scientific discovery.