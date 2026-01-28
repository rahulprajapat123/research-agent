-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Sources table: tracks research sources and their credibility
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    authors TEXT[],
    publication_date DATE,
    source_type VARCHAR(50) NOT NULL, -- arxiv, blog, benchmark, vendor_announcement
    domain VARCHAR(255) NOT NULL,
    tier VARCHAR(10) NOT NULL, -- tier_1, tier_2, tier_3
    credibility_score INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    author_h_index INTEGER,
    raw_file_url TEXT, -- S3/Blob storage URL
    raw_file_size_bytes BIGINT,
    parsed_text TEXT,
    metadata JSONB DEFAULT '{}',
    ingestion_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed, needs_manual
    ingestion_error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_sources_url ON sources(url);
CREATE INDEX idx_sources_domain ON sources(domain);
CREATE INDEX idx_sources_tier ON sources(tier);
CREATE INDEX idx_sources_status ON sources(ingestion_status);
CREATE INDEX idx_sources_publication_date ON sources(publication_date DESC);
CREATE INDEX idx_sources_credibility ON sources(credibility_score DESC);

-- Claims table: structured knowledge units extracted from sources
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    evidence_type VARCHAR(50) NOT NULL, -- experiment, benchmark, case_study, theoretical, anecdotal
    evidence_location TEXT, -- section/figure/table reference
    metrics JSONB, -- quantitative results
    conditions TEXT, -- under what conditions
    limitations TEXT, -- caveats and constraints
    rag_applicability VARCHAR(50) NOT NULL, -- retrieval, chunking, embedding, reranking, generation, evaluation, other
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    embedding vector(1536), -- OpenAI text-embedding-3-small default
    
    -- Validation tracking
    extraction_method VARCHAR(50) DEFAULT 'llm_auto', -- llm_auto, human_validated, human_edited
    validated_by VARCHAR(255),
    validated_at TIMESTAMP,
    validation_notes TEXT,
    
    -- Conflict tracking
    has_conflict BOOLEAN DEFAULT false,
    conflict_with_claim_ids UUID[],
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_claims_source_id ON claims(source_id);
CREATE INDEX idx_claims_evidence_type ON claims(evidence_type);
CREATE INDEX idx_claims_rag_applicability ON claims(rag_applicability);
CREATE INDEX idx_claims_confidence ON claims(confidence_score DESC);
CREATE INDEX idx_claims_extraction_method ON claims(extraction_method);
CREATE INDEX idx_claims_has_conflict ON claims(has_conflict) WHERE has_conflict = true;

-- Vector similarity search index (HNSW for performance)
CREATE INDEX idx_claims_embedding ON claims USING hnsw (embedding vector_cosine_ops);

-- Validation queue: tracks claims requiring human review
CREATE TABLE IF NOT EXISTS validation_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 0, -- higher = more urgent
    reason VARCHAR(255), -- sampling, low_confidence, conflict, malformed
    assigned_to VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending', -- pending, in_review, approved, rejected, edited
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP
);

CREATE INDEX idx_validation_queue_status ON validation_queue(status);
CREATE INDEX idx_validation_queue_priority ON validation_queue(priority DESC);
CREATE INDEX idx_validation_queue_assigned ON validation_queue(assigned_to);

-- Recommendation logs: track what was recommended and why
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_context JSONB NOT NULL,
    retrieved_claim_ids UUID[] NOT NULL,
    reranked_claim_ids UUID[] NOT NULL,
    final_recommendation TEXT NOT NULL,
    reasoning TEXT,
    citations JSONB,
    retrieval_metrics JSONB, -- top-k scores, vector/keyword weights
    llm_model VARCHAR(100),
    llm_tokens_used INTEGER,
    response_time_ms INTEGER,
    user_feedback VARCHAR(50), -- helpful, not_helpful, partially_helpful
    user_feedback_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_recommendation_logs_created_at ON recommendation_logs(created_at DESC);

-- Source quality metrics: track source reliability over time
CREATE TABLE IF NOT EXISTS source_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    claims_extracted INTEGER DEFAULT 0,
    claims_validated INTEGER DEFAULT 0,
    claims_rejected INTEGER DEFAULT 0,
    average_claim_confidence FLOAT,
    times_cited_in_recommendations INTEGER DEFAULT 0,
    user_feedback_positive INTEGER DEFAULT 0,
    user_feedback_negative INTEGER DEFAULT 0,
    UNIQUE(source_id, metric_date)
);

CREATE INDEX idx_source_quality_source_id ON source_quality_metrics(source_id);
CREATE INDEX idx_source_quality_date ON source_quality_metrics(metric_date DESC);

-- System metadata: track ingestion runs and system state
CREATE TABLE IF NOT EXISTS system_metadata (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert initial system metadata
INSERT INTO system_metadata (key, value) VALUES 
    ('last_ingestion_run', '{"status": "never", "timestamp": null}'),
    ('total_sources', '{"count": 0}'),
    ('total_claims', '{"count": 0}'),
    ('validation_stats', '{"pending": 0, "completed": 0}')
ON CONFLICT (key) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_claims_updated_at BEFORE UPDATE ON claims
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
