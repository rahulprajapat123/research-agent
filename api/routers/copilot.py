"""
Project Research Copilot - Backend API

Pipeline Flow:
1. Accept project name + document brief from developer
2. Extract topics/keywords from document brief
3. Check Azure Blob Storage cache for recent results
4. Search research papers on arXiv, OpenReview, and AI/ML/RAG sites
5. Store results in Azure Blob Storage
6. Analyze architecture (LLM vs RAG vs Hybrid vs Fine-tuning)
7. Return comprehensive analysis with 12 sections
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
import asyncio
import re
from loguru import logger

from utils.llm_client import LLMClient
from utils.azure_storage import get_azure_storage

router = APIRouter()
llm_client = LLMClient()
azure_storage = get_azure_storage()

# ============================================================================
# MODELS
# ============================================================================

class CopilotRequest(BaseModel):
    """Input: Project name + document brief"""
    project_name: str = Field(..., description="Name of the project")
    project_brief: str = Field(..., description="Document brief from developer")

class ResearchPaper(BaseModel):
    """Research paper metadata"""
    title: str
    authors: List[str]
    year: int
    url: str
    abstract: str
    source: str  # arxiv, openreview, huggingface, etc.
    relevance_score: float = 0.0  # Ranking score for project relevance

class CopilotResponse(BaseModel):
    """Complete 12-section analysis output"""
    project_name: str
    arxiv_papers: List[ResearchPaper] = Field(default_factory=list, description="Papers downloaded from arXiv")
    project_understanding: List[str]
    assumptions: List[str]
    user_personas_and_use_cases: List[str]
    functional_requirements: Dict[str, List[str]]
    non_functional_requirements: Dict[str, str]
    architecture_decision: Dict[str, str]
    system_design: Dict[str, Any]
    research_digest: Dict[str, Any]
    tech_stack: Dict[str, List[str]]
    milestones: Dict[str, str]
    risks_and_mitigations: List[Dict[str, str]]
    next_steps: List[str]

# ============================================================================
# STEP 1: EXTRACT KEYWORDS/TOPICS FROM BRIEF
# ============================================================================

async def extract_keywords_and_topics(document_brief: str) -> Dict[str, Any]:
    """Extract keywords, topics, and search terms from document brief"""
    
    prompt = f"""Analyze this project brief and extract:
1. Core topics (3-5 main topics)
2. Keywords for research (8-12 specific keywords)
3. Technical domains (e.g., NLP, RAG, LLM, embeddings)
4. Problem domain (e.g., healthcare, legal, education)

Project Brief:
{document_brief}

Return JSON with:
{{
  "core_topics": ["topic1", "topic2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "technical_domains": ["domain1", "domain2", ...],
  "problem_domain": "domain name"
}}"""

    try:
        response = await llm_client.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        import json
        
        # Try to parse as JSON
        try:
            extracted = json.loads(response)
        except json.JSONDecodeError:
            # LLM returned text instead of JSON - parse manually
            logger.warning("LLM returned non-JSON, extracting keywords from text")
            keywords = []
            lines = response.strip().split('\n')
            for line in lines:
                # Extract words that look like keywords
                words = line.replace(',', ' ').replace(';', ' ').split()
                keywords.extend([w.strip('":[]{}') for w in words if len(w) > 3 and w.isalnum()])
            
            extracted = {
                "core_topics": keywords[:5] if keywords else ["AI"],
                "keywords": keywords[:15] if keywords else ["artificial intelligence", "machine learning"],
                "technical_domains": ["AI/ML", "RAG"],
                "problem_domain": "research"
            }
        
        logger.info(f"Extracted {len(extracted.get('keywords', []))} keywords from brief")
        return extracted
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        # Fallback: simple extraction
        return {
            "core_topics": ["AI", "machine learning"],
            "keywords": ["artificial intelligence", "machine learning", "RAG", "LLM"],
            "technical_domains": ["AI/ML"],
            "problem_domain": "general"
        }

# ============================================================================
# STEP 2: SEARCH ARXIV API
# ============================================================================

async def search_arxiv(keywords: List[str], max_results: int = 10) -> List[ResearchPaper]:
    """Search arXiv for papers matching keywords"""
    
    papers = []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create search queries from keywords
            queries = []
            
            # Combine keywords for better results
            for i in range(0, len(keywords), 2):
                query_terms = keywords[i:i+2]
                query = " AND ".join([f'all:"{term}"' for term in query_terms])
                queries.append(query)
            
            # Search each query
            for query in queries[:4]:  # Limit to avoid rate limits
                url = "https://export.arxiv.org/api/query"
                params = {
                    "search_query": query,
                    "start": 0,
                    "max_results": max_results // len(queries[:4]),
                    "sortBy": "relevance",
                    "sortOrder": "descending"
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                    
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.find("atom:name", ns)
                        if name is not None:
                            authors.append(name.text)
                    
                    published = entry.find("atom:published", ns).text
                    year = int(published[:4])
                    
                    url = entry.find("atom:id", ns).text
                    abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                    
                    papers.append(ResearchPaper(
                        title=title,
                        authors=authors,
                        year=year,
                        url=url,
                        abstract=abstract,
                        source="arxiv"
                    ))
                
                await asyncio.sleep(0.5)  # Rate limiting
        
        # Deduplicate by title
        seen = set()
        unique_papers = []
        for paper in papers:
            title_key = paper.title.lower().strip()
            if title_key not in seen:
                seen.add(title_key)
                unique_papers.append(paper)
        
        logger.info(f"Found {len(unique_papers)} unique papers from arXiv")
        return unique_papers[:max_results]
        
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []

# ============================================================================
# STEP 3: SEARCH OPENREVIEW & OTHER AI/ML SITES
# ============================================================================

async def search_ai_ml_sites(keywords: List[str], max_results: int = 5) -> List[ResearchPaper]:
    """
    Search OpenReview and other AI/ML/RAG research sites
    
    Note: This is a placeholder. Real implementation would:
    - Search OpenReview API (requires auth)
    - Search Hugging Face papers
    - Search Papers with Code
    - Search Google Scholar
    """
    
    # For now, we'll note this in the response
    logger.info(f"AI/ML site search requested for: {', '.join(keywords[:3])}")
    
    # Placeholder - in production, integrate with:
    # - OpenReview API: https://api.openreview.net
    # - Papers with Code API: https://paperswithcode.com/api/v1/
    # - Semantic Scholar API: https://api.semanticscholar.org
    
    return []

# ============================================================================
# STEP 3: RANK PAPERS BY RELEVANCE
# ============================================================================

async def rank_papers_by_relevance(
    papers: List[ResearchPaper],
    document_brief: str,
    keywords: List[str]
) -> List[ResearchPaper]:
    """Rank papers by relevance to project brief"""
    
    if not papers:
        return []
    
    # Create concise paper summaries for ranking
    papers_text = "\n\n".join([
        f"Paper {i+1}: {p.title} ({p.year})\nAbstract: {p.abstract[:300]}..."
        for i, p in enumerate(papers[:15])
    ])
    
    keywords_text = ", ".join(keywords)
    
    prompt = f"""Rank these papers by relevance to the project. Consider:
- How well the paper's methods apply to this project
- Recency (favor 2023-2026 papers)
- Practical implementation insights

Project Brief: {document_brief[:500]}
Keywords: {keywords_text}

Papers:
{papers_text}

Output ONLY paper numbers with scores (1-10), format: paper_number,score
Example:
1,9.5
2,7.8
3,8.2"""

    try:
        response = await llm_client.complete(
            prompt=prompt,
            temperature=0.2,
            max_tokens=400
        )
        
        # Parse rankings
        rankings = {}
        for line in response.strip().split("\n"):
            if "," in line:
                try:
                    parts = line.strip().split(",")
                    paper_num = int(parts[0])
                    score = float(parts[1])
                    rankings[paper_num] = score
                except:
                    continue
        
        # Apply scores
        for i, paper in enumerate(papers[:15]):
            paper_idx = i + 1
            if paper_idx in rankings:
                # Store score in the model field
                paper.relevance_score = rankings[paper_idx]
            else:
                # Fallback: use year-based score
                paper.relevance_score = (paper.year - 2015) / 10
        
        # Sort by score
        sorted_papers = sorted(
            papers,
            key=lambda p: (p.relevance_score, p.year),
            reverse=True
        )
        
        return sorted_papers[:12]
        
    except Exception as e:
        logger.error(f"Paper ranking failed: {e}")
        return sorted(papers, key=lambda p: p.year, reverse=True)[:12]

# ============================================================================
# STEP 5: COMPREHENSIVE ANALYSIS (12 SECTIONS)
# ============================================================================

async def generate_comprehensive_analysis(
    project_name: str,
    document_brief: str,
    keywords: Dict[str, Any],
    papers: List[ResearchPaper]
) -> CopilotResponse:
    """Generate all 12 sections of the analysis"""
    
    # Prepare research context with MORE details from papers
    papers_summary = "\n\n".join([
        f"Paper {i+1}: [{p.source.upper()} {p.year}] {p.title}\n"
        f"Abstract: {p.abstract[:400]}...\n"
        f"URL: {p.url}"
        for i, p in enumerate(papers[:12])
    ])
    
    keywords_text = ", ".join(keywords.get("keywords", []))
    topics_text = ", ".join(keywords.get("core_topics", []))
    
    # Master analysis prompt with emphasis on extracting tech from papers
    prompt = f"""You are a Product Research Copilot for developers. Generate a comprehensive build plan based on LATEST research.

CRITICAL: Analyze the research papers below and extract:
- Novel techniques, architectures, and methodologies mentioned
- Specific models, frameworks, and tools referenced
- Latest technologies and implementations from 2023-2026
- Innovative approaches that are state-of-the-art

PROJECT NAME: {project_name}

DOCUMENT BRIEF:
{document_brief}

EXTRACTED TOPICS: {topics_text}
KEYWORDS: {keywords_text}
TECHNICAL DOMAINS: {', '.join(keywords.get('technical_domains', []))}
PROBLEM DOMAIN: {keywords.get('problem_domain', 'general')}

RESEARCH PAPERS FOUND:
{papers_summary}

Generate ALL 12 sections below. Be specific and actionable.

1. PROJECT UNDERSTANDING (2-4 bullets)
- Summarize what this project aims to build

2. ASSUMPTIONS (only if details are missing)
- List assumptions about unclear requirements

3. USER PERSONAS & USE CASES (4-6 bullets)
- Who will use this and for what specific scenarios?

4. FUNCTIONAL REQUIREMENTS
MUST-HAVE (5-8 core features):
SHOULD-HAVE (3-5 nice-to-have features):

5. NON-FUNCTIONAL REQUIREMENTS
Latency: [target response time with justification]
Privacy: [data handling and compliance requirements]
Scale: [expected users/queries per day/month]
Cost: [budget considerations and optimization]

6. ARCHITECTURE DECISION
CHOICE: Choose ONE from [LLM Only | RAG (Embeddings + Retrieval) | Hybrid (RAG + LLM) | Fine-tuning]

REASONING (2-3 sentences):
Why this architecture fits based on:
- Use RAG for: factual grounding, long documents, enterprise knowledge, citation needs, changing information
- Use LLM/Agents for: workflow automation, reasoning over small context, tool-calling, decisioning
- Use Hybrid when: need both structured knowledge AND reasoning/synthesis
- Use Fine-tuning ONLY if: stable repetitive pattern, strict formatting, domain style, AND sufficient data

7. SYSTEM DESIGN
COMPONENTS (5-8 key components):
DATA FLOW (4-6 steps describing how data flows through system):

8. RESEARCH DIGEST
MANDATORY: Analyze ALL papers above and extract:
- Novel techniques and architectures mentioned in papers (be specific)
- Specific models, frameworks, libraries cited (exact names/versions)
- Latest methodologies from 2023-2026 papers
- Implementation insights and code references
- Evaluation metrics and benchmarks used
- Pitfalls and challenges discussed

9. RECOMMENDED TECH STACK
CRITICAL: Base recommendations on research papers above AND latest 2024-2026 technologies.
Extract specific tools/models mentioned in papers. Include versions where relevant.

LLM_LAYER: [Extract from papers: GPT-4o, Claude 3.5 Sonnet, Llama 3.1, Gemini 2.0, Mistral, etc. - use LATEST models from papers]
RETRIEVAL_LAYER: [If RAG: specific embedding models from papers like text-embedding-3-large, BGE, ColBERT; Vector DBs like Qdrant, Weaviate, Pinecone, pgvector]
DATA_STORES: [Modern choices: PostgreSQL with pgvector, MongoDB, Redis Stack, Upstash]
BACKEND: [Modern frameworks: FastAPI + Python 3.12, Node.js + TypeScript, Go]
FRONTEND: [Modern: Next.js 14+, React 18+, Vue 3+, SvelteKit, Tailwind CSS]
TOOLS_LIBRARIES: [Specific libraries from papers: LangChain, LlamaIndex, DSPy, Instructor, etc.]
EVALUATION: [From papers: RAGAS, TruLens, Phoenix, LangSmith, custom eval harnesses]

10. MILESTONES
MVP (Week 1-2): [minimum viable features to validate concept]
V1 (Week 3-4): [core functionality complete]
V2 (Week 5-6): [full feature set with optimization]

11. RISKS & MITIGATIONS
List 4-6 risks with specific mitigations.

12. WHAT TO BUILD NEXT (5 concrete action items)
List exact next steps to start development today.

OUTPUT MUST BE VALID JSON:
{{
  "project_understanding": ["point1", "point2", ...],
  "assumptions": ["assumption1", ...],
  "user_personas_and_use_cases": ["persona1: use case", ...],
  "functional_requirements": {{
    "must_have": ["feature1", ...],
    "should_have": ["feature1", ...]
  }},
  "non_functional_requirements": {{
    "latency": "detailed requirement",
    "privacy": "detailed requirement",
    "scale": "detailed requirement",
    "cost": "detailed requirement"
  }},
  "architecture_decision": {{
    "choice": "Hybrid (RAG + LLM)",
    "reasoning": "detailed reasoning"
  }},
  "system_design": {{
    "components": ["component1", ...],
    "data_flow": ["step1", "step2", ...]
  }},
  "research_digest": {{
    "key_notes": ["insight from paper 1", "technique from paper 2", "finding from paper 3"],
    "novel_techniques": ["specific technique 1", "architecture 2"],
    "specific_models": ["model name 1", "framework 2"],
    "latest_findings": ["2024 finding 1", "2025 technique 2"]
  }},
  "tech_stack": {{
    "llm_layer": ["Specific model from papers with version"],
    "retrieval_layer": ["Embedding model + Vector DB from papers"],
    "data_stores": ["Modern database choices"],
    "backend": ["Framework + Language version"],
    "frontend": ["Modern framework if needed"],
    "tools_libraries": ["Specific libraries from papers"],
    "evaluation": ["Tools mentioned in papers"]
  }},
  "milestones": {{
    "mvp": "description",
    "v1": "description",
    "v2": "description"
  }},
  "risks_and_mitigations": [
    {{"risk": "risk1", "mitigation": "solution1"}},
    ...
  ],
  "next_steps": ["step1", "step2", ...]
}}"""

    try:
        response = await llm_client.complete(
            prompt=prompt,
            temperature=0.8,  # Higher temperature for more creative tech stack suggestions
            max_tokens=4096  # Model maximum
        )
        
        import json
        
        # Try to parse as JSON
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError:
            # If LLM doesn't return JSON, create structured response from text
            logger.warning("LLM returned non-JSON, extracting structured data from papers")
            
            # Extract technologies mentioned in paper abstracts AND titles
            tech_keywords = {
                "models": set(),
                "frameworks": set(),
                "methods": set(),
                "databases": set()
            }
            
            for paper in papers[:12]:
                text_to_analyze = (paper.title + " " + paper.abstract).lower()
                
                # Extract common AI/ML models (2023-2026)
                if "gpt-4" in text_to_analyze or "gpt4" in text_to_analyze: tech_keywords["models"].add("GPT-4o")
                elif "gpt" in text_to_analyze: tech_keywords["models"].add("GPT-3.5")
                if "claude" in text_to_analyze: tech_keywords["models"].add("Claude 3.5 Sonnet")
                if "llama 3" in text_to_analyze or "llama3" in text_to_analyze: tech_keywords["models"].add("Llama 3.1 (Meta)")
                if "gemini" in text_to_analyze: tech_keywords["models"].add("Gemini 2.0 (Google)")
                if "mistral" in text_to_analyze: tech_keywords["models"].add("Mistral Large")
                if "bert" in text_to_analyze: tech_keywords["models"].add("BERT variants")
                if "t5" in text_to_analyze: tech_keywords["models"].add("T5")
                if "roberta" in text_to_analyze: tech_keywords["models"].add("RoBERTa")
                if "sentence-transformer" in text_to_analyze or "sentence transformer" in text_to_analyze: 
                    tech_keywords["models"].add("Sentence Transformers")
                if "colbert" in text_to_analyze: tech_keywords["models"].add("ColBERT")
                
                # Extract frameworks and libraries
                if "langchain" in text_to_analyze: tech_keywords["frameworks"].add("LangChain")
                if "llamaindex" in text_to_analyze: tech_keywords["frameworks"].add("LlamaIndex")
                if "dspy" in text_to_analyze: tech_keywords["frameworks"].add("DSPy")
                if "pytorch" in text_to_analyze: tech_keywords["frameworks"].add("PyTorch")
                if "tensorflow" in text_to_analyze: tech_keywords["frameworks"].add("TensorFlow")
                if "huggingface" in text_to_analyze or "hugging face" in text_to_analyze: 
                    tech_keywords["frameworks"].add("HuggingFace")
                if "fastapi" in text_to_analyze: tech_keywords["frameworks"].add("FastAPI")
                if "instructor" in text_to_analyze: tech_keywords["frameworks"].add("Instructor")
                if "pydantic" in text_to_analyze: tech_keywords["frameworks"].add("Pydantic")
                
                # Extract vector databases and storage
                if "qdrant" in text_to_analyze: tech_keywords["databases"].add("Qdrant")
                if "pinecone" in text_to_analyze: tech_keywords["databases"].add("Pinecone")
                if "weaviate" in text_to_analyze: tech_keywords["databases"].add("Weaviate")
                if "chroma" in text_to_analyze: tech_keywords["databases"].add("ChromaDB")
                if "pgvector" in text_to_analyze: tech_keywords["databases"].add("pgvector")
                if "milvus" in text_to_analyze: tech_keywords["databases"].add("Milvus")
                if "faiss" in text_to_analyze: tech_keywords["databases"].add("FAISS")
                
                # Extract methods and architectures
                if "transformer" in text_to_analyze: tech_keywords["methods"].add("Transformer architecture")
                if "rag" in text_to_analyze or "retrieval-augmented" in text_to_analyze or "retrieval augmented" in text_to_analyze: 
                    tech_keywords["methods"].add("RAG (Retrieval-Augmented Generation)")
                if "embedding" in text_to_analyze: tech_keywords["methods"].add("Vector embeddings")
                if "fine-tun" in text_to_analyze: tech_keywords["methods"].add("Fine-tuning")
                if "prompt engineer" in text_to_analyze or "prompt optim" in text_to_analyze: 
                    tech_keywords["methods"].add("Prompt optimization")
                if "few-shot" in text_to_analyze: tech_keywords["methods"].add("Few-shot learning")
                if "zero-shot" in text_to_analyze: tech_keywords["methods"].add("Zero-shot learning")
                if "chain-of-thought" in text_to_analyze or "cot" in text_to_analyze: 
                    tech_keywords["methods"].add("Chain-of-Thought reasoning")
                if "agent" in text_to_analyze and ("llm" in text_to_analyze or "language model" in text_to_analyze):
                    tech_keywords["methods"].add("LLM Agents")
                if "multi-agent" in text_to_analyze: tech_keywords["methods"].add("Multi-agent systems")
            
            # Determine architecture from project brief
            brief_lower = document_brief.lower()
            architecture_choice = "Hybrid (RAG + LLM)"
            if "chatbot" in brief_lower or "conversation" in brief_lower:
                architecture_choice = "LLM with Memory"
            elif "knowledge" in brief_lower or "document" in brief_lower or "search" in brief_lower:
                architecture_choice = "RAG (Retrieval-Augmented Generation)"
            elif "agent" in brief_lower or "tool" in brief_lower:
                architecture_choice = "LLM Agents with Tools"
            
            # Build dynamic tech stack based on extracted technologies
            llm_models = list(tech_keywords["models"])[:4] if tech_keywords["models"] else ["GPT-4o (OpenAI)", "Claude 3.5 Sonnet"]
            frameworks_list = list(tech_keywords["frameworks"])[:5] if tech_keywords["frameworks"] else ["LangChain", "FastAPI"]
            vector_dbs = list(tech_keywords["databases"])[:3] if tech_keywords["databases"] else ["Qdrant", "Pinecone"]
            methods_list = list(tech_keywords["methods"])[:6]
            
            analysis = {
                "project_understanding": ["Analyzing project requirements and objectives"],
                "assumptions": [],
                "user_personas_and_use_cases": ["Primary users and their use cases"],
                "functional_requirements": {
                    "must_have": ["Core functionality based on project brief"],
                    "should_have": ["Additional features for enhanced experience"]
                },
                "non_functional_requirements": {
                    "latency": "Target response time < 3s",
                    "privacy": "Handle data according to requirements",
                    "scale": "Scale based on user base",
                    "cost": "Optimize for cost-effectiveness"
                },
                "architecture_decision": {
                    "choice": architecture_choice,
                    "reasoning": f"Based on project requirements and research findings: {', '.join(list(tech_keywords['methods'])[:3])}"
                },
                "system_design": {
                    "components": ["Ingestion Pipeline", "Vector Store", "Retrieval Engine", "LLM API", "Backend API", "Frontend UI"],
                    "data_flow": [
                        "1. Ingest and preprocess documents",
                        "2. Generate embeddings and store in vector DB",
                        "3. On query: retrieve relevant context",
                        "4. Augment with LLM for generation",
                        "5. Return response with citations"
                    ]
                },
                "research_digest": {
                    "key_notes": [
                        f"Latest research ({len([p for p in papers if p.year >= 2024])} papers from 2024-2026)",
                        f"Key methodologies: {', '.join(methods_list[:3])}" if methods_list else "Modern AI/ML approaches",
                        f"State-of-art models: {', '.join(list(tech_keywords['models'])[:3])}" if tech_keywords['models'] else "Latest LLM architectures",
                        f"Popular frameworks: {', '.join(list(tech_keywords['frameworks'])[:3])}" if tech_keywords['frameworks'] else "Emerging AI frameworks",
                        f"Vector databases: {', '.join(vector_dbs[:2])}" if vector_dbs else "Modern vector storage solutions"
                    ],
                    "novel_techniques": methods_list if methods_list else ["Transformer-based architectures", "Retrieval-augmented approaches"],
                    "specific_models": list(tech_keywords["models"])[:6] if tech_keywords["models"] else ["GPT-4o", "Claude 3.5"],
                    "latest_findings": [
                        f"Analysis based on {len(papers)} papers from {min(p.year for p in papers) if papers else 2020}-{max(p.year for p in papers) if papers else 2026}",
                        f"Emerging patterns: {', '.join(methods_list[:2])}" if len(methods_list) >= 2 else "Modern AI/ML techniques"
                    ]
                },
                "tech_stack": {
                    "llm_layer": llm_models if llm_models else ["GPT-4o (OpenAI)", "Claude 3.5 Sonnet (Anthropic)", "Llama 3.1 (open-source)"],
                    "retrieval_layer": (
                        ["text-embedding-3-large (OpenAI embeddings)"] + vector_dbs 
                        if ("rag" in brief_lower or "retrieval" in brief_lower or "knowledge" in brief_lower) 
                        else []
                    ),
                    "data_stores": ["PostgreSQL with pgvector extension", "Redis Stack (caching)", "MinIO or S3 (document storage)"],
                    "backend": (frameworks_list if frameworks_list else []) + ["FastAPI (Python 3.12)", "Uvicorn (ASGI server)"],
                    "frontend": (
                        ["Next.js 14+ with App Router", "React 18", "Tailwind CSS", "shadcn/ui components"] 
                        if any(word in brief_lower for word in ["ui", "frontend", "web", "interface", "dashboard"]) 
                        else ["REST API endpoints only"]
                    ),
                    "tools_libraries": list(set(frameworks_list + ["Pydantic v2", "Instructor (structured outputs)", "LiteLLM (multi-provider)"][:4])),
                    "evaluation": ["RAGAS (RAG evaluation)", "LangSmith (tracing)", "Custom eval suite", "A/B testing framework"]
                },
                "milestones": {
                    "mvp": "Basic ingestion and retrieval (Week 1-2)",
                    "v1": "Full pipeline with LLM integration (Week 3-4)",
                    "v2": "Advanced features and optimization (Week 5-6)"
                },
                "risks_and_mitigations": [
                    {"risk": "LLM API costs", "mitigation": "Implement caching and rate limiting"},
                    {"risk": "Latency issues", "mitigation": "Optimize retrieval and use async processing"}
                ],
                "next_steps": [
                    "Set up development environment",
                    "Implement document ingestion pipeline",
                    "Configure vector database",
                    "Build retrieval mechanism",
                    "Integrate LLM for generation"
                ]
            }
        
        # Add research papers to digest
        research_digest = {
            "arxiv_papers": [
                {
                    "title": p.title,
                    "year": p.year,
                    "url": p.url,
                    "key_idea": p.abstract[:200] + "...",
                    "project_takeaway": f"Relevant for {keywords.get('problem_domain', 'implementation')}"
                }
                for p in papers if p.source == "arxiv"
            ],
            "openreview_papers": [
                {
                    "title": p.title,
                    "year": p.year,
                    "url": p.url,
                    "key_idea": p.abstract[:200] + "...",
                    "project_takeaway": "Evaluation insights"
                }
                for p in papers if p.source == "openreview"
            ],
            "key_notes": analysis.get("research_digest", {}).get("key_notes", [])
        }
        
        return CopilotResponse(
            project_name=project_name,
            arxiv_papers=[p for p in papers if p.source == "arxiv"],
            project_understanding=analysis.get("project_understanding", []),
            assumptions=analysis.get("assumptions", []),
            user_personas_and_use_cases=analysis.get("user_personas_and_use_cases", []),
            functional_requirements=analysis.get("functional_requirements", {}),
            non_functional_requirements=analysis.get("non_functional_requirements", {}),
            architecture_decision=analysis.get("architecture_decision", {}),
            system_design=analysis.get("system_design", {}),
            research_digest=research_digest,
            tech_stack=analysis.get("tech_stack", {}),
            milestones=analysis.get("milestones", {}),
            risks_and_mitigations=analysis.get("risks_and_mitigations", []),
            next_steps=analysis.get("next_steps", [])
        )
        
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ============================================================================
# API ENDPOINT
# ============================================================================

@router.post("/analyze", response_model=CopilotResponse)
async def analyze_project(request: CopilotRequest):
    """
    Research Copilot Pipeline: Project Brief → Comprehensive Analysis
    
    Steps:
    1. Check Azure Blob Storage cache
    2. Extract keywords/topics from document brief
    3. Search arXiv API for relevant papers
    4. Search OpenReview & AI/ML sites
    5. Store results in Azure cache
    6. Rank papers by relevance
    7. Generate comprehensive 12-section analysis
    """
    
    try:
        logger.info(f"🤖 Starting analysis for: {request.project_name}")
        
        # Step 1: Check cache first
        if azure_storage.enabled:
            logger.info("Step 1/7: Checking Azure Blob Storage cache")
            cached_papers_dict = azure_storage.get_cached_results(request.project_name, max_age_hours=24)
            if cached_papers_dict:
                logger.info(f"✅ Using {len(cached_papers_dict)} cached papers from Azure Blob")
                # Convert dict papers back to ResearchPaper objects
                cached_papers = [
                    ResearchPaper(
                        title=p.get('title', ''),
                        authors=p.get('authors', []),
                        year=p.get('year', 2024),
                        url=p.get('url', ''),
                        abstract=p.get('abstract', ''),
                        source=p.get('source', 'arxiv'),
                        relevance_score=p.get('relevance_score', 0.0)
                    )
                    for p in cached_papers_dict
                ]
                # Generate analysis from cached papers
                keywords = await extract_keywords_and_topics(request.project_brief)
                logger.info("Step 6/7: Ranking cached papers by relevance")
                ranked_papers = await rank_papers_by_relevance(
                    cached_papers,
                    request.project_brief,
                    keywords.get("keywords", [])
                )
                logger.info("Step 7/7: Generating comprehensive analysis from cache")
                analysis = await generate_comprehensive_analysis(
                    project_name=request.project_name,
                    document_brief=request.project_brief,
                    keywords=keywords,
                    papers=ranked_papers
                )
                logger.info(f"✅ Analysis complete (from cache) for: {request.project_name}")
                return analysis
        else:
            logger.info("Step 1/7: Azure Blob Storage not configured - live search only")
        
        # Step 2: Extract keywords and topics
        logger.info("Step 2/7: Extracting keywords and topics from document brief")
        keywords = await extract_keywords_and_topics(request.project_brief)
        
        # Step 3: Search arXiv API
        logger.info(f"Step 3/7: Searching arXiv for papers on: {', '.join(keywords.get('keywords', [])[:5])}")
        arxiv_papers = await search_arxiv(keywords.get("keywords", []))
        logger.info(f"🔍 Found {len(arxiv_papers)} papers from arXiv")
        
        # Step 4: Search other AI/ML sites
        logger.info("Step 4/7: Searching OpenReview and AI/ML research sites")
        other_papers = await search_ai_ml_sites(keywords.get("keywords", []))
        
        # Combine all sources
        all_papers = arxiv_papers + other_papers
        
        logger.info(f"📚 Total papers found: {len(all_papers)}")
        
        # Step 5: Store in Azure cache
        if azure_storage.enabled and all_papers:
            logger.info("Step 5/7: Storing results in Azure Blob Storage")
            azure_storage.store_search_results(
                project_name=request.project_name,
                papers=all_papers,
                metadata={
                    "keywords": keywords.get("keywords", []),
                    "search_date": datetime.now().isoformat(),
                    "paper_count": len(all_papers)
                }
            )
        else:
            logger.info("Step 5/7: Cache storage skipped (disabled or no papers)")
        
        # Step 6: Rank papers by relevance
        logger.info("Step 6/7: Ranking papers by relevance to project")
        ranked_papers = await rank_papers_by_relevance(
            all_papers,
            request.project_brief,
            keywords.get("keywords", [])
        )
        
        # Step 7: Generate comprehensive analysis
        logger.info("Step 7/7: Generating comprehensive 12-section analysis")
        analysis = await generate_comprehensive_analysis(
            project_name=request.project_name,
            document_brief=request.project_brief,
            keywords=keywords,
            papers=ranked_papers
        )
        
        logger.info(f"✅ Analysis complete for: {request.project_name}")
        return analysis
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Project Research Copilot",
        "pipeline": [
            "1. Extract keywords/topics",
            "2. Search arXiv",
            "3. Search OpenReview & AI/ML sites",
            "4. Rank papers",
            "5. Generate 12-section analysis"
        ]
    }
