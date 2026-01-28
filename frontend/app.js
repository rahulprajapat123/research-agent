// API Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';

// State Management
let currentSection = 'dashboard';
let systemHealth = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    checkSystemHealth();
    loadDashboard();
    setupForms();
    addInteractiveEffects();
    addParticleEffect();
    
    // Auto-refresh system health every 30 seconds
    setInterval(checkSystemHealth, 30000);
});

// Add interactive effects
function addInteractiveEffects() {
    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', createRipple);
    });
    
    // Add card entrance animations
    observeCardAnimations();
    
    // Add smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', smoothScroll);
    });
}

function createRipple(event) {
    const button = event.currentTarget;
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        transform: scale(0);
        animation: ripple 0.6s ease-out;
        pointer-events: none;
    `;
    
    button.style.position = 'relative';
    button.style.overflow = 'hidden';
    button.appendChild(ripple);
    
    setTimeout(() => ripple.remove(), 600);
}

function observeCardAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
                }, index * 100);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.card, .stat-card').forEach(card => {
        card.style.opacity = '0';
        observer.observe(card);
    });
}

function smoothScroll(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function addParticleEffect() {
    // Add subtle floating particles to background
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(style);
}

// Navigation
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show corresponding section
            const sectionId = link.dataset.section;
            showSection(sectionId);
        });
    });
}

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        currentSection = sectionId;
        
        // Load section data
        switch(sectionId) {
            case 'dashboard':
                loadDashboard();
                break;
            case 'sources':
                loadSources();
                break;
            case 'validation':
                loadValidationQueue();
                break;
        }
    }
}

// System Health
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        systemHealth = data;
        
        updateSystemStatus(data);
        if (currentSection === 'dashboard') {
            displayHealthStatus(data);
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateSystemStatus({ status: 'error' });
    }
}

function updateSystemStatus(data) {
    const statusElement = document.getElementById('systemStatus');
    const icon = statusElement.querySelector('i');
    const text = statusElement.querySelector('span');
    
    if (data.status === 'operational') {
        statusElement.classList.add('healthy');
        statusElement.classList.remove('unhealthy');
        text.textContent = 'System Healthy';
    } else {
        statusElement.classList.add('unhealthy');
        statusElement.classList.remove('healthy');
        text.textContent = 'System Error';
    }
}

function displayHealthStatus(data) {
    const healthContainer = document.getElementById('healthStatus');
    
    const services = data.services || {};
    let html = '';
    
    for (const [service, status] of Object.entries(services)) {
        const isHealthy = status === 'healthy';
        html += `
            <div class="health-item">
                <span class="health-label">${formatServiceName(service)}</span>
                <span class="health-status ${isHealthy ? 'healthy' : 'unhealthy'}">
                    ${isHealthy ? '✓ Healthy' : '✗ ' + status}
                </span>
            </div>
        `;
    }
    
    healthContainer.innerHTML = html;
}

function formatServiceName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Dashboard
async function loadDashboard() {
    try {
        // Load stats
        await Promise.all([
            loadSourcesStats(),
            loadValidationStats(),
            checkSystemHealth()
        ]);
        
        // Load recent activity
        loadRecentActivity();
    } catch (error) {
        console.error('Dashboard loading error:', error);
    }
}

async function loadSourcesStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/sources?limit=1000`);
        const data = await response.json();
        
        const count = data.sources?.length || 0;
        animateCounter(document.getElementById('totalSources'), count);
    } catch (error) {
        document.getElementById('totalSources').textContent = 'Error';
    }
}

async function loadValidationStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/validation/queue?limit=1000`);
        const data = await response.json();
        
        animateCounter(document.getElementById('pendingValidation'), data.length || 0);
        animateCounter(document.getElementById('validatedClaims'), Math.floor(Math.random() * 500)); // Mock data
        animateCounter(document.getElementById('recommendationsGenerated'), Math.floor(Math.random() * 50)); // Mock data
    } catch (error) {
        document.getElementById('pendingValidation').textContent = 'Error';
    }
}

function animateCounter(element, target) {
    const duration = 1000;
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

async function loadRecentActivity() {
    const activityContainer = document.getElementById('recentActivity');
    
    try {
        const response = await fetch(`${API_BASE_URL}/sources?limit=10`);
        const data = await response.json();
        
        if (!data.sources || data.sources.length === 0) {
            activityContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No recent activity</p>';
            return;
        }
        
        let html = '<div style="max-height: 300px; overflow-y: auto;">';
        data.sources.forEach(source => {
            html += `
                <div class="health-item">
                    <div>
                        <div style="font-weight: 500; margin-bottom: 0.25rem;">
                            ${source.title || 'Untitled Document'}
                        </div>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">
                            ${formatDate(source.created_at)}
                        </div>
                    </div>
                    <span class="meta-badge status-${source.status}">
                        ${source.status}
                    </span>
                </div>
            `;
        });
        html += '</div>';
        
        activityContainer.innerHTML = html;
    } catch (error) {
        activityContainer.innerHTML = '<p style="color: var(--danger-color);">Failed to load activity</p>';
    }
}

// Forms Setup
function setupForms() {
    // Pipeline Form
    const pipelineForm = document.getElementById('pipelineForm');
    if (pipelineForm) {
        pipelineForm.addEventListener('submit', handlePipelineSubmit);
    }
    
    // Recommendation Form
    const recommendationForm = document.getElementById('recommendationForm');
    recommendationForm.addEventListener('submit', handleRecommendationSubmit);
    
    // Ingestion Form
    const ingestionForm = document.getElementById('ingestionForm');
    ingestionForm.addEventListener('submit', handleIngestionSubmit);
}

// Full Pipeline Execution
async function handlePipelineSubmit(e) {
    e.preventDefault();
    
    showLoading();
    
    try {
        // Get auto-ingest sources
        const sourcesText = document.getElementById('autoIngestSources').value;
        const sources = sourcesText
            .split('\n')
            .map(s => s.trim())
            .filter(s => s && (s.startsWith('http://') || s.startsWith('https://')));
        
        const projectBrief = {
            project_name: document.getElementById('pipelineProjectName').value,
            use_case: document.getElementById('pipelineUseCase').value,
            document_types: document.getElementById('pipelineDocTypes').value.split(',').map(s => s.trim()),
            avg_document_length: parseInt(document.getElementById('pipelineDocLength').value) || 2000,
            domain: document.getElementById('pipelineDomain').value || 'general',
            language: document.getElementById('pipelineLanguage').value || 'english',
            budget: document.getElementById('pipelineBudget').value,
            latency_requirements: document.getElementById('pipelineLatency').value,
            scale: document.getElementById('pipelineScale').value,
            current_challenges: document.getElementById('pipelineChallenges').value
                .split('\n')
                .map(s => s.trim())
                .filter(Boolean),
            rag_components_of_interest: Array.from(document.querySelectorAll('input[name="pipelineComponents"]:checked'))
                .map(cb => cb.value),
            auto_ingest_sources: sources.length > 0 ? sources : null
        };
        
        const response = await fetch(`${API_BASE_URL}/pipeline/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(projectBrief)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Pipeline execution failed');
        }
        
        const data = await response.json();
        displayPipelineResult(data);
        showToast('🚀 Pipeline started successfully!', 'success');
        
    } catch (error) {
        console.error('Pipeline error:', error);
        showToast(error.message || 'Failed to execute pipeline', 'error');
    } finally {
        hideLoading();
    }
}

function displayPipelineResult(data) {
    const resultDiv = document.getElementById('pipelineResult');
    resultDiv.classList.remove('hidden');
    
    const steps = data.pipeline_steps || {};
    const recommendations = steps.step_4_recommendations?.recommendations || [];
    
    let html = `
        <div class="card">
            <div class="card-header">
                <h2><i class="fas fa-chart-line"></i> Pipeline Execution Status</h2>
                <span class="meta-badge status-${data.status}">
                    ${data.status === 'processing' ? '⏳ Processing' : '✅ Completed'}
                </span>
            </div>
            <div class="card-body">
                <div style="background: var(--hover-bg); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
                    <h3 style="margin-bottom: 0.5rem;">${data.project_brief.project_name}</h3>
                    <p style="color: var(--text-secondary);">${data.message}</p>
                </div>
                
                <!-- Pipeline Steps -->
                <div style="display: grid; gap: 1rem; margin-bottom: 2rem;">
                    <!-- Step 1: Ingestion -->
                    <div class="source-item">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">1</div>
                            <div style="flex: 1;">
                                <strong>Ingestion Layer</strong>
                                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                                    Status: ${steps.step_1_ingestion?.status || 'N/A'}
                                </div>
                            </div>
                            <span class="meta-badge">
                                ${steps.step_1_ingestion?.sources_queued || 0} sources queued
                            </span>
                        </div>
                        ${steps.step_1_ingestion?.sources?.length > 0 ? `
                            <div style="margin-top: 0.5rem;">
                                ${steps.step_1_ingestion.sources.map(s => `
                                    <div style="font-size: 0.875rem; padding: 0.5rem; background: white; border-radius: 0.375rem; margin-bottom: 0.5rem;">
                                        <i class="fas fa-link"></i> ${s.url}
                                        <span class="meta-badge" style="margin-left: 0.5rem;">${s.status}</span>
                                    </div>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                    
                    <!-- Step 2: Normalization -->
                    <div class="source-item">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #10b981, #059669); border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">2</div>
                            <div style="flex: 1;">
                                <strong>Normalization Layer</strong>
                                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                                    Parse → Extract Claims → Generate Embeddings
                                </div>
                            </div>
                            <span class="meta-badge">
                                ${steps.step_2_normalization?.status || 'N/A'}
                            </span>
                        </div>
                    </div>
                    
                    <!-- Step 3: Knowledge Store -->
                    <div class="source-item">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #f59e0b, #d97706); border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">3</div>
                            <div style="flex: 1;">
                                <strong>Knowledge Store</strong>
                                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                                    Postgres + pgvector
                                </div>
                            </div>
                            <span class="meta-badge">
                                ${steps.step_3_knowledge_store?.claims_retrieved || 0} claims retrieved
                            </span>
                        </div>
                    </div>
                    
                    <!-- Step 4: Recommendations -->
                    <div class="source-item">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #8b5cf6, #7c3aed); border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">4</div>
                            <div style="flex: 1;">
                                <strong>Recommendation Engine</strong>
                                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                                    Hybrid retrieval → Re-rank → LLM generation
                                </div>
                            </div>
                            <span class="meta-badge status-${steps.step_4_recommendations?.status}">
                                ${steps.step_4_recommendations?.status || 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
    `;
    
    // Show recommendations if available
    if (recommendations.length > 0) {
        html += `
            <h3 style="margin: 2rem 0 1rem;"><i class="fas fa-star"></i> Generated Recommendations</h3>
            ${recommendations.map((rec, index) => `
                <div class="recommendation-item">
                    <div class="recommendation-header">
                        <h4 class="recommendation-title">${index + 1}. ${rec.technique}</h4>
                        <span class="impact-badge impact-${rec.implementation_priority?.toLowerCase() || 'medium'}">
                            ${rec.implementation_priority || 'Medium'} Priority
                        </span>
                    </div>
                    <p class="recommendation-description">${rec.description}</p>
                    <div class="recommendation-rationale">
                        <strong>Rationale:</strong> ${rec.rationale}
                    </div>
                    ${rec.estimated_impact ? `
                        <div style="margin-top: 0.5rem;">
                            <strong>Estimated Impact:</strong> ${rec.estimated_impact}
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        `;
        
        // Show citations
        if (steps.step_4_recommendations?.citations?.length > 0) {
            html += `
                <div style="margin-top: 2rem; padding: 1.5rem; background: var(--hover-bg); border-radius: 0.5rem;">
                    <h3 style="margin-bottom: 1rem;"><i class="fas fa-bookmark"></i> Citations</h3>
                    ${steps.step_4_recommendations.citations.map((citation, index) => `
                        <div style="margin-bottom: 0.75rem;">
                            [${index + 1}] <strong>${citation.title || 'Research Paper'}:</strong>
                            ${citation.url ? `<a href="${citation.url}" target="_blank">${citation.url}</a>` : citation.source || ''}
                        </div>
                    `).join('')}
                </div>
            `;
        }
    } else if (data.status === 'processing') {
        html += `
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 0.5rem;">
                <strong>⏳ Pipeline Processing</strong>
                <p style="margin: 0.5rem 0 0; color: var(--text-secondary);">
                    Documents are being ingested and processed. Recommendations will be generated after claims are extracted and validated.
                    Check back in a few minutes or monitor the Sources and Validation Queue sections.
                </p>
            </div>
        `;
    }
    
    // Next steps
    if (data.next_steps && data.next_steps.filter(Boolean).length > 0) {
        html += `
            <div style="margin-top: 2rem; padding: 1.5rem; background: #dbeafe; border-radius: 0.5rem;">
                <h3 style="margin-bottom: 1rem;"><i class="fas fa-tasks"></i> Next Steps</h3>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    ${data.next_steps.filter(Boolean).map(step => `<li style="margin-bottom: 0.5rem;">${step}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Recommendations
async function handleRecommendationSubmit(e) {
    e.preventDefault();
    
    showLoading();
    
    try {
        // Gather form data
        const projectContext = {
            project_name: document.getElementById('projectName').value,
            use_case: document.getElementById('useCase').value,
            data_characteristics: {
                document_types: document.getElementById('documentTypes').value.split(',').map(s => s.trim()).filter(Boolean),
                avg_document_length: parseInt(document.getElementById('avgDocLength').value) || 2000,
                domain: document.getElementById('domain').value || 'general',
                language: document.getElementById('language').value || 'english'
            },
            constraints: {
                budget: document.getElementById('budget').value,
                latency_requirements: document.getElementById('latencyReq').value,
                scale: document.getElementById('scale').value,
                existing_infrastructure: document.getElementById('infrastructure').value
            },
            current_challenges: document.getElementById('challenges').value
                .split('\n')
                .map(s => s.trim())
                .filter(Boolean),
            rag_components_of_interest: Array.from(document.querySelectorAll('input[name="components"]:checked'))
                .map(cb => cb.value)
        };
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(projectContext)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate recommendations');
        }
        
        const data = await response.json();
        displayRecommendations(data);
        showToast('Recommendations generated successfully!', 'success');
        
    } catch (error) {
        console.error('Recommendation error:', error);
        showToast(error.message || 'Failed to generate recommendations', 'error');
    } finally {
        hideLoading();
    }
}

function displayRecommendations(data) {
    const resultDiv = document.getElementById('recommendationsResult');
    const contentDiv = document.getElementById('recommendationsContent');
    
    resultDiv.classList.remove('hidden');
    
    let html = `
        <div class="recommendation-summary" style="background: var(--hover-bg); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem;"><i class="fas fa-file-alt"></i> Summary</h3>
            <p style="line-height: 1.8;">${data.summary || 'No summary available'}</p>
        </div>
    `;
    
    if (data.recommendations && data.recommendations.length > 0) {
        html += '<h3 style="margin-bottom: 1rem;"><i class="fas fa-list"></i> Recommendations</h3>';
        
        data.recommendations.forEach((rec, index) => {
            const impactClass = rec.implementation_priority?.toLowerCase() || 'medium';
            
            html += `
                <div class="recommendation-item">
                    <div class="recommendation-header">
                        <h4 class="recommendation-title">${index + 1}. ${rec.technique}</h4>
                        <span class="impact-badge impact-${impactClass}">
                            ${rec.implementation_priority || 'Medium'} Priority
                        </span>
                    </div>
                    <p class="recommendation-description">${rec.description}</p>
                    <div class="recommendation-rationale">
                        <strong>Rationale:</strong> ${rec.rationale}
                    </div>
                    ${rec.estimated_impact ? `
                        <div style="margin-top: 0.5rem;">
                            <strong>Estimated Impact:</strong> ${rec.estimated_impact}
                        </div>
                    ` : ''}
                    ${rec.supporting_claims && rec.supporting_claims.length > 0 ? `
                        <div class="supporting-claims">
                            <h4>Supporting Evidence:</h4>
                            ${rec.supporting_claims.map(claim => `
                                <div class="claim-reference">
                                    • ${claim.claim_text || claim.source || 'Research finding'}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        });
    }
    
    if (data.citations && data.citations.length > 0) {
        html += `
            <div class="card" style="margin-top: 2rem;">
                <div class="card-header">
                    <h3><i class="fas fa-bookmark"></i> Citations</h3>
                </div>
                <div class="card-body">
                    ${data.citations.map((citation, index) => `
                        <div style="margin-bottom: 0.75rem;">
                            [${index + 1}] <strong>${citation.title || 'Research Paper'}:</strong>
                            ${citation.url ? `<a href="${citation.url}" target="_blank">${citation.url}</a>` : citation.source || ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    contentDiv.innerHTML = html;
    
    // Scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Ingestion
async function handleIngestionSubmit(e) {
    e.preventDefault();
    
    showLoading();
    
    try {
        const formData = {
            url: document.getElementById('docUrl').value,
            title: document.getElementById('docTitle').value || null,
            source_type: document.getElementById('sourceType').value,
            authors: document.getElementById('authors').value
                ? document.getElementById('authors').value.split(',').map(s => s.trim())
                : null,
            publication_date: document.getElementById('pubDate').value || null,
            citation_count: parseInt(document.getElementById('citationCount').value) || null,
            author_h_index: parseInt(document.getElementById('authorHIndex').value) || null,
            metadata: {}
        };
        
        const response = await fetch(`${API_BASE_URL}/ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to ingest document');
        }
        
        const data = await response.json();
        displayIngestionResult(data);
        showToast('Document submitted successfully!', 'success');
        
        // Reset form
        e.target.reset();
        
    } catch (error) {
        console.error('Ingestion error:', error);
        showToast(error.message || 'Failed to ingest document', 'error');
    } finally {
        hideLoading();
    }
}

function displayIngestionResult(data) {
    const resultDiv = document.getElementById('ingestionResult');
    resultDiv.classList.remove('hidden');
    
    resultDiv.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h2><i class="fas fa-check-circle"></i> Submission Successful</h2>
            </div>
            <div class="card-body">
                <div class="health-item">
                    <span class="health-label">Source ID</span>
                    <span style="font-family: monospace;">${data.source_id}</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Status</span>
                    <span class="meta-badge status-${data.status}">${data.status}</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Tier</span>
                    <span class="meta-badge tier-${data.tier.toLowerCase()}">${data.tier}</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Credibility Score</span>
                    <span>${data.credibility_score}/100</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Message</span>
                    <span>${data.message}</span>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Sources
async function loadSources() {
    const sourcesList = document.getElementById('sourcesList');
    sourcesList.innerHTML = '<div class="loading">Loading sources...</div>';
    
    try {
        const tier = document.getElementById('filterTier').value;
        const status = document.getElementById('filterStatus').value;
        
        let url = `${API_BASE_URL}/sources?limit=50`;
        if (tier) url += `&tier=${tier}`;
        if (status) url += `&status=${status}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.sources || data.sources.length === 0) {
            sourcesList.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 2rem;">No sources found</p>';
            return;
        }
        
        let html = '';
        data.sources.forEach(source => {
            html += `
                <div class="source-item">
                    <div class="source-header">
                        <div>
                            <h3 class="source-title">${source.title || 'Untitled Document'}</h3>
                            <a href="${source.url}" target="_blank" class="source-url">
                                <i class="fas fa-external-link-alt"></i> ${source.url}
                            </a>
                        </div>
                        <span class="meta-badge tier-${source.tier?.toLowerCase() || 'c'}">
                            Tier ${source.tier || 'C'}
                        </span>
                    </div>
                    <div class="source-meta">
                        <span class="meta-badge status-${source.status}">
                            <i class="fas fa-circle"></i> ${source.status}
                        </span>
                        <span class="meta-badge">
                            <i class="fas fa-star"></i> Score: ${source.credibility_score}
                        </span>
                        <span class="meta-badge">
                            <i class="fas fa-clock"></i> ${formatDate(source.created_at)}
                        </span>
                    </div>
                </div>
            `;
        });
        
        sourcesList.innerHTML = html;
        
    } catch (error) {
        console.error('Sources loading error:', error);
        sourcesList.innerHTML = '<p style="color: var(--danger-color); text-align: center; padding: 2rem;">Failed to load sources</p>';
    }
}

// Validation Queue
async function loadValidationQueue() {
    const queueContainer = document.getElementById('validationQueue');
    queueContainer.innerHTML = '<div class="loading">Loading validation queue...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/validation/queue?limit=20`);
        const data = await response.json();
        
        if (!data || data.length === 0) {
            queueContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 2rem;">No pending validations</p>';
            return;
        }
        
        let html = '';
        data.forEach(item => {
            const priorityClass = item.priority >= 80 ? 'high' : item.priority >= 50 ? 'medium' : 'low';
            
            html += `
                <div class="validation-item">
                    <div class="validation-header">
                        <div>
                            <strong>Claim from:</strong> ${item.source_title || 'Unknown Source'}
                            <br>
                            <small style="color: var(--text-secondary);">
                                <i class="fas fa-link"></i> 
                                <a href="${item.source_url}" target="_blank">${item.source_url}</a>
                            </small>
                        </div>
                        <span class="priority-badge priority-${priorityClass}">
                            Priority: ${item.priority}
                        </span>
                    </div>
                    
                    <div class="claim-text">
                        ${item.claim_text}
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <strong>Evidence Type:</strong> ${item.evidence_type}
                        </div>
                        <div>
                            <strong>Confidence:</strong> ${Math.round(item.confidence_score * 100)}%
                        </div>
                        <div>
                            <strong>Reason:</strong> ${item.reason}
                        </div>
                    </div>
                    
                    <div class="validation-actions">
                        <button class="btn btn-success" onclick="validateClaim('${item.id}', 'approved')">
                            <i class="fas fa-check"></i> Approve
                        </button>
                        <button class="btn btn-danger" onclick="validateClaim('${item.id}', 'rejected')">
                            <i class="fas fa-times"></i> Reject
                        </button>
                        <button class="btn btn-secondary" onclick="editClaim('${item.id}')">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                    </div>
                </div>
            `;
        });
        
        queueContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Validation queue loading error:', error);
        queueContainer.innerHTML = '<p style="color: var(--danger-color); text-align: center; padding: 2rem;">Failed to load validation queue</p>';
    }
}

async function validateClaim(validationId, decision) {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/validation/${validationId}/submit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                decision: decision,
                reviewer_notes: null
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit validation');
        }
        
        showToast(`Claim ${decision} successfully!`, 'success');
        loadValidationQueue();
        
    } catch (error) {
        console.error('Validation error:', error);
        showToast('Failed to submit validation', 'error');
    } finally {
        hideLoading();
    }
}

function editClaim(claimId) {
    showToast('Edit functionality coming soon!', 'info');
}

// Utility Functions
function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString();
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' : 
                 'fa-info-circle';
    
    toast.innerHTML = `
        <i class="fas ${icon} toast-icon"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Add progress bar
    const progress = document.createElement('div');
    progress.style.cssText = `
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        background: ${type === 'success' ? 'var(--success-color)' : 
                     type === 'error' ? 'var(--danger-color)' : 
                     'var(--primary-color)'};
        width: 100%;
        animation: shrink 5s linear;
    `;
    toast.appendChild(progress);
    
    // Add dismiss button
    const dismissBtn = document.createElement('button');
    dismissBtn.innerHTML = '<i class="fas fa-times"></i>';
    dismissBtn.style.cssText = `
        background: none;
        border: none;
        color: var(--text-secondary);
        cursor: pointer;
        padding: 0.25rem;
        margin-left: auto;
        transition: color 0.2s;
    `;
    dismissBtn.onmouseover = () => dismissBtn.style.color = 'var(--text-primary)';
    dismissBtn.onmouseout = () => dismissBtn.style.color = 'var(--text-secondary)';
    dismissBtn.onclick = () => {
        toast.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    };
    toast.appendChild(dismissBtn);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
    
    // Add animation styles
    if (!document.getElementById('toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
            @keyframes shrink {
                from { width: 100%; }
                to { width: 0%; }
            }
            @keyframes slideOutRight {
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

function showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

// Export for inline onclick handlers
window.loadSources = loadSources;
window.loadValidationQueue = loadValidationQueue;
window.validateClaim = validateClaim;
window.editClaim = editClaim;
