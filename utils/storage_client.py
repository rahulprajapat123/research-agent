"""
Storage client for Azure Blob/S3/MinIO
"""
from azure.storage.blob import BlobServiceClient, ContentSettings
from config import get_settings
from loguru import logger

settings = get_settings()


class StorageClient:
    """Object storage client (Azure/S3/MinIO)"""
    
    def __init__(self):
        if settings.storage_type == "azure":
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            )
            self.container_name = settings.azure_storage_container_name
            self._ensure_container_exists()
        
        elif settings.storage_type == "s3":
            try:
                import boto3
                from botocore.exceptions import ClientError
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                self.bucket = settings.s3_bucket_name
            except ImportError:
                raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        else:
            raise NotImplementedError(f"Storage type {settings.storage_type} not yet implemented")
    
    def _ensure_container_exists(self):
        """Ensure Azure container exists (Azure only)"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created Azure container: {self.container_name}")
        except Exception as e:
            logger.warning(f"Container check/create: {e}")
    
    def upload(self, file_content: bytes, object_key: str) -> str:
        """
        Upload file to storage
        
        Returns the storage URL
        """
        try:
            if settings.storage_type == "azure":
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=object_key
                )
                
                # Detect content type
                content_type = "application/pdf" if object_key.endswith(".pdf") else "application/octet-stream"
                
                blob_client.upload_blob(
                    file_content,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type)
                )
                
                url = blob_client.url
                logger.info(f"Uploaded to Azure Blob: {url}")
                return url
            
            elif settings.storage_type == "s3":
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=file_content
                )
                url = f"s3://{self.bucket}/{object_key}"
                logger.info(f"Uploaded to S3: {url}")
                return url
            
        except Exception as e:
            logger.error(f"Storage upload error: {e}")
            raise
    
    def download(self, object_key: str) -> bytes:
        """Download file from storage"""
        try:
            if settings.storage_type == "azure":
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=object_key
                )
                return blob_client.download_blob().readall()
            
            elif settings.storage_type == "s3":
                response = self.client.get_object(
                    Bucket=self.bucket,
                    Key=object_key
                )
                return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Storage download error: {e}")
            raise
    
    def delete(self, object_key: str):
        """Delete file from storage"""
        try:
            if settings.storage_type == "azure":
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=object_key
                )
                blob_client.delete_blob()
                logger.info(f"Deleted from Azure Blob: {object_key}")
            
            elif settings.storage_type == "s3":
                self.client.delete_object(
                    Bucket=self.bucket,
                    Key=object_key
                )
                logger.info(f"Deleted from S3: {object_key}")
            
        except Exception as e:
            logger.error(f"Storage delete error: {e}")
            raise
