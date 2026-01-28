"""
Azure Blob Storage client for research paper caching
Stores arXiv papers and search results for faster retrieval
"""
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from loguru import logger
from typing import Optional, Dict, List, Any
import json
from datetime import datetime, timedelta
from config import get_settings

settings = get_settings()


class AzureStorageClient:
    """Azure Blob Storage client for paper caching"""
    
    def __init__(self):
        """Initialize Azure Blob Storage client"""
        self.connection_string = settings.azure_storage_connection_string
        self.container_name = settings.azure_storage_container_name
        
        if not self.connection_string:
            logger.warning("Azure Storage not configured - caching disabled")
            self.enabled = False
            return
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            # Create container if it doesn't exist
            try:
                self.container_client.create_container()
                logger.info(f"Created Azure Blob container: {self.container_name}")
            except ResourceExistsError:
                logger.info(f"Azure Blob container exists: {self.container_name}")
            
            self.enabled = True
            logger.success("✅ Azure Blob Storage initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage: {e}")
            self.enabled = False
    
    def _get_cache_key(self, project_name: str, search_type: str = "arxiv") -> str:
        """Generate cache key for project"""
        # Sanitize project name for blob name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name.lower())
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"cache/{search_type}/{safe_name}_{timestamp}.json"
    
    def store_search_results(
        self,
        project_name: str,
        papers: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store arXiv search results in blob storage"""
        if not self.enabled:
            return False
        
        try:
            blob_name = self._get_cache_key(project_name)
            
            # Convert papers to dict if they're Pydantic models
            papers_data = []
            for paper in papers:
                if hasattr(paper, 'model_dump'):
                    papers_data.append(paper.model_dump())
                elif hasattr(paper, 'dict'):
                    papers_data.append(paper.dict())
                elif isinstance(paper, dict):
                    papers_data.append(paper)
                else:
                    papers_data.append(dict(paper))
            
            cache_data = {
                "project_name": project_name,
                "timestamp": datetime.now().isoformat(),
                "paper_count": len(papers),
                "papers": papers_data,
                "metadata": metadata or {}
            }
            
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                json.dumps(cache_data, indent=2),
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )
            
            logger.info(f"✅ Stored {len(papers)} papers for '{project_name}' in Azure Blob")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store results in Azure Blob: {e}")
            return False
    
    def get_cached_results(
        self,
        project_name: str,
        max_age_hours: int = 24
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search results from blob storage"""
        if not self.enabled:
            return None
        
        try:
            blob_name = self._get_cache_key(project_name)
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Check if blob exists and is recent
            properties = blob_client.get_blob_properties()
            last_modified = properties.last_modified
            age = datetime.now(last_modified.tzinfo) - last_modified
            
            if age > timedelta(hours=max_age_hours):
                logger.info(f"Cache expired for '{project_name}' (age: {age})")
                return None
            
            # Download and parse
            blob_data = blob_client.download_blob().readall()
            cache_data = json.loads(blob_data)
            
            papers = cache_data.get("papers", [])
            logger.info(f"✅ Retrieved {len(papers)} cached papers for '{project_name}'")
            return papers
            
        except ResourceNotFoundError:
            logger.info(f"No cached results for '{project_name}'")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached results: {e}")
            return None
    
    def store_paper_pdf(
        self,
        paper_id: str,
        pdf_content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store paper PDF in blob storage"""
        if not self.enabled:
            return False
        
        try:
            blob_name = f"papers/{paper_id}.pdf"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Store PDF
            blob_client.upload_blob(
                pdf_content,
                overwrite=True,
                content_settings={'content_type': 'application/pdf'},
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Stored PDF for paper '{paper_id}' in Azure Blob")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store PDF: {e}")
            return False
    
    def get_paper_pdf(self, paper_id: str) -> Optional[bytes]:
        """Retrieve paper PDF from blob storage"""
        if not self.enabled:
            return None
        
        try:
            blob_name = f"papers/{paper_id}.pdf"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            pdf_content = blob_client.download_blob().readall()
            logger.info(f"✅ Retrieved PDF for paper '{paper_id}'")
            return pdf_content
            
        except ResourceNotFoundError:
            logger.info(f"PDF not found for paper '{paper_id}'")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve PDF: {e}")
            return None
    
    def list_cached_projects(self, limit: int = 100) -> List[str]:
        """List all cached project names"""
        if not self.enabled:
            return []
        
        try:
            blobs = self.container_client.list_blobs(name_starts_with="cache/")
            project_names = []
            
            for blob in blobs:
                if limit and len(project_names) >= limit:
                    break
                # Extract project name from blob path
                name_parts = blob.name.split("/")
                if len(name_parts) >= 3:
                    project_names.append(name_parts[2].split("_")[0])
            
            return project_names
            
        except Exception as e:
            logger.error(f"Failed to list cached projects: {e}")
            return []
    
    def clear_cache(self, project_name: Optional[str] = None) -> bool:
        """Clear cache for specific project or all projects"""
        if not self.enabled:
            return False
        
        try:
            if project_name:
                # Clear specific project
                blob_name = self._get_cache_key(project_name)
                blob_client = self.container_client.get_blob_client(blob_name)
                blob_client.delete_blob()
                logger.info(f"✅ Cleared cache for '{project_name}'")
            else:
                # Clear all cache
                blobs = self.container_client.list_blobs(name_starts_with="cache/")
                for blob in blobs:
                    self.container_client.delete_blob(blob.name)
                logger.info("✅ Cleared all cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# Global instance
_storage_client: Optional[AzureStorageClient] = None


def get_azure_storage() -> AzureStorageClient:
    """Get or create Azure Storage client instance"""
    global _storage_client
    if _storage_client is None:
        _storage_client = AzureStorageClient()
    return _storage_client


# Example usage
if __name__ == "__main__":
    # Test Azure Storage
    storage = get_azure_storage()
    
    if storage.enabled:
        # Test storing search results
        test_papers = [
            {
                "id": "2301.00001",
                "title": "Test Paper on AI",
                "authors": ["John Doe"],
                "abstract": "This is a test paper about AI research.",
                "published": "2023-01-01",
                "arxiv_url": "https://arxiv.org/abs/2301.00001"
            }
        ]
        
        storage.store_search_results(
            project_name="Test AI Project",
            papers=test_papers,
            metadata={"search_query": "artificial intelligence"}
        )
        
        # Test retrieving
        cached = storage.get_cached_results("Test AI Project")
        print(f"Retrieved {len(cached) if cached else 0} cached papers")
        
        # List projects
        projects = storage.list_cached_projects()
        print(f"Cached projects: {projects}")
    else:
        print("Azure Storage not enabled - configure AZURE_STORAGE_CONNECTION_STRING")
