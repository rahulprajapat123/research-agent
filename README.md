# RAG Research Intelligence System

A dynamic, continuously updating intelligence system that ingests high-quality RAG research, structures knowledge into decision-grade insights, and produces context-aware, research-backed recommendations for applied RAG system design.

## 🎯 What This System Does

Given a RAG project brief, this system answers:
- **"What proven techniques should we apply for this exact scenario?"**
- **"What does recent research suggest we should avoid?"**

All outputs are **cited**, **contextual**, and **actionable**.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Ingestion Layer (n8n orchestrated)                         │
│  - Fetch from whitelisted sources (arXiv, blogs, etc.)      │
│  - Download PDFs/HTML → Store in S3/Blob                    │
│  - POST to /api/v1/ingest                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Normalization Layer (The Critical 60%)                     │
│  - Parse documents (PDF/HTML → text)                        │
│  - LLM-based claim extraction (structured prompts)          │
│  - Evidence linking & confidence scoring                    │
│  - Human-in-the-loop validation (100% initially, then 10%)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Knowledge Store (Postgres + pgvector)                      │
│  - Structured claims with embeddings                        │
│  - Source credibility tracking                              │
│  - Conflict detection & resolution                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Recommendation Engine                                       │
│  - Hybrid vector + keyword retrieval                        │
│  - Re-rank: credibility × applicability × recency           │
│  - LLM generates recommendations with citations             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Supabase account** (Postgres with pgvector)
- **Upstash Redis account**
- **OpenAI/Anthropic/Azure OpenAI API key**
- **Azure account** (for Blob Storage)

### 1. Clone and Setup Environment

```bash
cd "c:\Users\praja\Desktop\research agent brief"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example env file
copy .env.example .env

# Edit .env with your credentials
notepad .env
```

**Required variables:**
```ini
# Supabase (get from your Supabase project settings)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_DB_URL=postgresql://postgres:your-password@db.your-project.supabase.co:5432/postgres

# Upstash Redis
UPSTASH_REDIS_URL=https://your-redis.upstash.io
UPSTASH_REDIS_TOKEN=your-token

# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
# OR
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Storage (Azure Blob Storage)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your-account;AccountKey=your-key;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER_NAME=rag-research-documents
```

### 3. Initialize Database

```bash
# Ensure pgvector extension is enabled in Supabase
# Go to: Supabase Dashboard → Database → Extensions → Enable "vector"

# Run schema setup
python -c "from database.connection import execute_schema_file; execute_schema_file('database/schema.sql')"
```

### 4. Run the API

```bash
# Development mode (with auto-reload)
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

---

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://arxiv.org/abs/2401.12345",
    "title": "Advanced RAG Techniques",
    "authors": ["John Doe"],
    "publication_date": "2024-01-15",
    "source_type": "arxiv",
    "citation_count": 45
  }'
```

### Get Recommendations

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "Customer Support RAG",
    "use_case": "customer_support",
    "data_characteristics": {
      "document_types": ["pdf", "html"],
      "avg_document_length": 2000,
      "domain": "technical_documentation"
    },
    "constraints": {
      "latency_requirements": "< 2 seconds",
      "scale": "100k documents"
    },
    "rag_components_of_interest": ["chunking", "retrieval", "reranking"]
  }'
```

### View Validation Queue

```bash
curl http://localhost:8000/api/v1/validation/queue?limit=10
```

---

## 🔧 n8n Integration Setup

### Install n8n

```bash
# Install n8n globally
npm install n8n -g

# Or use Docker
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

### Import Workflow

1. Access n8n at **http://localhost:5678**
2. Create new workflow
3. Import from file: `n8n_workflows/ingestion_pipeline.json`
4. Configure webhook URL in `.env`: `N8N_WEBHOOK_URL=http://localhost:5678/webhook/ingest`

### Workflow Features

- **Scheduled Ingestion**: Cron trigger for weekly/daily runs
- **RSS/Atom Feed Monitoring**: Auto-fetch new arXiv papers, blog posts
- **Human Review Queue**: Sends Slack notifications for validation
- **Error Handling**: Routes failed documents to manual review

---

## 📊 Database Schema

### Core Tables

- **`sources`**: Research documents with credibility scoring
- **`claims`**: Structured knowledge units with embeddings
- **`validation_queue`**: Human-in-the-loop review queue
- **`recommendation_logs`**: Track what was recommended and user feedback

### Vector Search

Uses **pgvector** with HNSW index for fast similarity search:
```sql
CREATE INDEX idx_claims_embedding ON claims 
USING hnsw (embedding vector_cosine_ops);
```

---

## 🎨 Source Tier System

### Tier 1: High Authority (+10 credibility boost)
- arXiv cs.IR, cs.CL (filtered: citation count >10 or h-index >20)
- Google AI Blog, DeepMind Blog
- Anthropic Research, Meta AI Research

### Tier 2: Industry Validated (+5 boost)
- LangChain blog (technical posts)
- LlamaIndex docs
- Pinecone/Weaviate/Qdrant benchmarks

### Tier 3: Monitor, Lower Weight (0 boost)
- Medium/Towards Data Science (verified authors only)

### Excluded
- LinkedIn posts, Twitter/X threads, unverified tutorials

---

## 🧠 Claim Extraction Process

### 1. Document Parsing
- **PDFs**: pdfplumber → pypdf fallback
- **HTML**: BeautifulSoup with content extraction
- **Text**: Plain text processing

### 2. LLM-Based Extraction
Uses structured prompt with strict JSON schema:
```python
{
  "claim_text": "Specific assertion",
  "evidence_type": "experiment | benchmark | case_study | theoretical | anecdotal",
  "evidence_location": "Section/Figure/Table reference",
  "metrics": {"recall_improvement": "+18%"},
  "conditions": "Under what conditions?",
  "limitations": "What caveats?",
  "rag_applicability": "chunking | retrieval | embedding | reranking | generation",
  "confidence_score": 0.85
}
```

### 3. Validation Rules
- **First 100 claims**: 100% human validation required
- **After 100**: 10% sampling (configurable)
- **Low confidence (<0.6)**: Always validate
- **Conflicts detected**: Always validate

### 4. Retry Logic
- Max 3 retries for malformed JSON
- Stricter prompts on retry
- Falls back to manual processing if all retries fail

---

## 🔍 Recommendation Algorithm

### Step 1: Hybrid Retrieval
- **Vector search** (70% weight): Semantic similarity using embeddings
- **Keyword search** (30% weight): Postgres full-text search
- Top-K: 20 claims retrieved

### Step 2: Re-ranking
Composite score from:
- **Source credibility** (30%): Tier-based + citation count
- **Claim confidence** (25%): LLM-assigned confidence
- **Recency** (15%): Exponential decay (180-day half-life)
- **Validation status** (15%): Human-validated boost
- **Evidence strength** (10%): Experiment > Benchmark > Case Study
- **Vector similarity** (5%): Original retrieval score

Top-K after rerank: 10 claims

### Step 3: LLM Generation
Structured prompt with:
- Project context (use case, constraints, challenges)
- Top-ranked claims with citations
- Output schema enforced (techniques, rationale, trade-offs)

---

## 🛠️ Development & Testing

### Run Tests
```bash
pytest
```

### Code Formatting
```bash
black .
flake8 .
```

### Database Migrations
```bash
# Test connection
python -c "from database.connection import test_connection; test_connection()"

# Apply schema
python -c "from database.connection import execute_schema_file; execute_schema_file('database/schema.sql')"
```

### View Logs
```bash
# Logs stored in logs/ directory
tail -f logs/api_*.log
```

---

## 📈 Monitoring & Metrics

### System Status Endpoint
```bash
curl http://localhost:8000/api/v1/status
```

Returns:
- Total sources & claims
- Pending validations
- Last ingestion run timestamp
- Configuration details

### Validation Stats
```bash
curl http://localhost:8000/api/v1/validation/stats
```

### Recommendation Feedback
```bash
curl -X POST http://localhost:8000/api/v1/recommendations/{id}/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": "helpful",
    "notes": "Implemented semantic chunking, saw +12% improvement"
  }'
```

---

## 🔐 Security & Rate Limiting

### Rate Limits (Redis-based)
- **Per minute**: 60 requests
- **Per hour**: 1000 requests

### CORS Configuration
Edit `ALLOWED_ORIGINS` in `.env` for your frontend domains.

### API Key Protection
Add authentication middleware for production deployments.

---

## 🚢 Production Deployment

### Environment Setup
```bash
# Set production environment
ENVIRONMENT=production
DEBUG=false

# Use production LLM endpoints
LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Increase workers
API_WORKERS=8
```

### Recommended Stack
- **Compute**: Azure App Service / Azure Container Instances
- **Database**: Supabase (managed Postgres with pgvector)
- **Cache**: Upstash Redis (serverless)
- **Storage**: Azure Blob Storage
- **Orchestration**: n8n Cloud or self-hosted

### Scaling Considerations
- **Database**: Connection pooling (configured in `database/connection.py`)
- **LLM**: Batch processing for embeddings
- **Storage**: Azure CDN for frequently accessed documents
- **Redis**: Use for caching embeddings and rate limiting

---

## 📚 Project Structure

```
research agent brief/
├── api/
│   └── routers/
│       ├── health.py           # Health checks
│       ├── ingestion.py        # Document ingestion
│       ├── recommendations.py  # Recommendation generation
│       └── validation.py       # Human validation
├── database/
│   ├── connection.py           # DB connection & session
│   └── schema.sql              # Complete DB schema
├── ingestion/
│   ├── coordinator.py          # Pipeline orchestration
│   ├── parser.py               # PDF/HTML parsing
│   └── source_classifier.py   # Tier classification
├── normalization/
│   ├── claim_extractor.py      # LLM-based extraction
│   └── validation_queue.py     # HITL queue management
├── prompts/
│   └── extraction_prompts.py   # All LLM prompts
├── recommendations/
│   ├── retriever.py            # Hybrid search
│   ├── reranker.py             # Multi-signal reranking
│   └── generator.py            # LLM recommendation generation
├── utils/
│   ├── embedding_client.py     # OpenAI embeddings
│   ├── llm_client.py           # Multi-provider LLM
│   ├── redis_client.py         # Cache & rate limiting
│   └── storage_client.py       # S3/Blob storage
├── n8n_workflows/              # (to be added)
│   └── ingestion_pipeline.json
├── main.py                     # FastAPI app
├── config.py                   # Configuration management
├── requirements.txt            # Dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## 🤝 Contributing

### Adding New Sources
1. Update `SOURCE_TIERS` in `config.py`
2. Add parsing logic in `ingestion/parser.py` if needed
3. Update n8n workflow to fetch from new source

### Improving Claim Extraction
1. Edit prompts in `prompts/extraction_prompts.py`
2. Test with sample documents
3. Adjust confidence thresholds in `.env`

### Customizing Reranking
1. Modify scoring weights in `recommendations/reranker.py`
2. Add new signals (e.g., user feedback, domain-specific scores)

---

## 🐛 Troubleshooting

### Database Connection Fails
```bash
# Check Supabase connection
python -c "from database.connection import test_connection; test_connection()"

# Verify pgvector extension
# Supabase Dashboard → Database → Extensions → Enable "vector"
```

### PDF Parsing Fails
- Check `parse_timeout_seconds` in `.env`
- Increase timeout for large documents
- Fallback to manual processing via validation queue

### LLM Returns Malformed JSON
- Check prompt in `prompts/extraction_prompts.py`
- Increase `CLAIM_EXTRACTION_MAX_RETRIES` in `.env`
- Review logs in `logs/api_*.log`

### No Claims Extracted
- Verify document has empirical evidence (not purely theoretical)
- Lower `MIN_CLAIM_CONFIDENCE` temporarily for testing
- Check document parsing output in logs

---

## 📞 Support & Questions

- **Issues**: Open an issue in your repository
- **Documentation**: See `/docs` endpoint when API is running
- **Logs**: Check `logs/api_*.log` for detailed error traces

---

## 📄 License

Proprietary - Internal BridgeAI use only

---

## 🎯 Roadmap

- [ ] n8n workflow templates (ingestion, validation, monitoring)
- [ ] Conflict detection and resolution UI
- [ ] Multi-language support (beyond English)
- [ ] Fine-tuned embedding models for RAG domain
- [ ] Real-time update notifications
- [ ] Benchmark tracking dashboard
- [ ] A/B testing framework for recommendations

---

**Built for BridgeAI** | **Decision-Support for RAG System Design**
