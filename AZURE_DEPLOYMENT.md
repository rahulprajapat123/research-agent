# 🚀 Azure Production Deployment Guide
## Project Research Copilot - Complete Hosting Setup

---

## 📋 Overview

This guide covers complete production deployment to Azure, including:
- ✅ Azure Container Registry (ACR) setup
- ✅ Azure App Service deployment  
- ✅ Azure Blob Storage integration
- ✅ Environment configuration
- ✅ CI/CD pipeline (optional)
- ✅ Monitoring & scaling

**Architecture**: Stateless FastAPI application using live arXiv API + Azure Blob Storage for caching

---

## 🔧 Prerequisites

### 1. Azure Account Setup
```bash
# Install Azure CLI (Windows)
winget install Microsoft.AzureCLI

# Login to Azure
az login

# Set your subscription
az account set --subscription "Your-Subscription-Name"
```

### 2. Local Requirements
- Docker Desktop installed
- Python 3.11+
- Git
- OpenAI API key

---

## 📦 Step 1: Prepare Application

### 1.1 Install Production Dependencies
```bash
cd "c:\Users\praja\Desktop\research agent brief"

# Install dependencies
pip install -r requirements-prod.txt
```

### 1.2 Test Locally with Docker
```bash
# Build Docker image
docker build -t research-copilot:latest .

# Test container locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-openai-key \
  -e ENVIRONMENT=production \
  research-copilot:latest

# Verify: http://localhost:8000/docs
```

---

## ☁️ Step 2: Azure Resource Setup

### 2.1 Create Resource Group
```bash
# Variables
RESOURCE_GROUP="research-copilot-rg"
LOCATION="eastus"  # or your preferred region
APP_NAME="research-copilot"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION
```

### 2.2 Create Azure Container Registry (ACR)
```bash
# Create ACR (lowercase name required)
ACR_NAME="researchcopilotacr"

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --location $LOCATION

# Enable admin access
az acr update -n $ACR_NAME --admin-enabled true

# Get ACR credentials
az acr credential show --name $ACR_NAME
```

### 2.3 Create Azure Blob Storage
```bash
# Create storage account (lowercase, no hyphens)
STORAGE_ACCOUNT="researchcopilotstore"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2

# Create container for research papers
az storage container create \
  --name research-papers \
  --account-name $STORAGE_ACCOUNT \
  --public-access off

# Get connection string
az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --output tsv
```

---

## 🐳 Step 3: Build and Push Docker Image

### 3.1 Login to ACR
```bash
az acr login --name $ACR_NAME
```

### 3.2 Build and Push Image
```bash
# Tag image for ACR
docker tag research-copilot:latest $ACR_NAME.azurecr.io/research-copilot:latest

# Push to ACR
docker push $ACR_NAME.azurecr.io/research-copilot:latest

# Verify
az acr repository list --name $ACR_NAME --output table
```

---

## 🌐 Step 4: Deploy to Azure App Service

### 4.1 Create App Service Plan
```bash
# Create Linux-based plan (F1 Free tier for testing)
az appservice plan create \
  --name ${APP_NAME}-plan \
  --resource-group $RESOURCE_GROUP \
  --sku B1 \
  --is-linux

# For production, use P1V2 or higher:
# --sku P1V2
```

### 4.2 Create Web App
```bash
# Create web app from ACR image
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan ${APP_NAME}-plan \
  --name $APP_NAME \
  --deployment-container-image-name $ACR_NAME.azurecr.io/research-copilot:latest

# Configure ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

az webapp config container set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $ACR_NAME.azurecr.io/research-copilot:latest \
  --docker-registry-server-url https://$ACR_NAME.azurecr.io \
  --docker-registry-server-user $ACR_USERNAME \
  --docker-registry-server-password $ACR_PASSWORD
```

### 4.3 Configure Environment Variables
```bash
# Get your storage connection string (from step 2.3)
STORAGE_CONN_STRING="<your-connection-string>"

# Set app settings
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    APP_NAME="Project Research Copilot" \
    ENVIRONMENT="production" \
    DEBUG="false" \
    LOG_LEVEL="INFO" \
    OPENAI_API_KEY="your-openai-api-key" \
    AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN_STRING" \
    AZURE_STORAGE_CONTAINER_NAME="research-papers" \
    LLM_PROVIDER="openai" \
    LLM_MODEL="gpt-4-turbo-preview" \
    API_WORKERS="4" \
    WEBSITES_PORT="8000"

# CRITICAL: Set the port
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings WEBSITES_PORT=8000
```

---

## 🔒 Step 5: Security Configuration

### 5.1 Enable HTTPS Only
```bash
az webapp update \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --https-only true
```

### 5.2 Configure CORS (if needed for frontend)
```bash
az webapp cors add \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --allowed-origins "https://your-frontend-domain.com"
```

### 5.3 Enable Application Insights (Monitoring)
```bash
# Create Application Insights
az monitor app-insights component create \
  --app ${APP_NAME}-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app ${APP_NAME}-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)

# Add to app settings
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

---

## 🚀 Step 6: Deploy and Verify

### 6.1 Restart App
```bash
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### 6.2 Get Application URL
```bash
echo "https://${APP_NAME}.azurewebsites.net"
```

### 6.3 Test Endpoints
```bash
# Health check
curl https://${APP_NAME}.azurewebsites.net/api/v1/health

# Test copilot (POST request)
curl -X POST https://${APP_NAME}.azurewebsites.net/api/v1/copilot/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "AI-Powered Chatbot",
    "project_brief": "Build a customer service chatbot using GPT-4"
  }'
```

### 6.4 Access API Documentation
Open in browser:
```
https://research-copilot.azurewebsites.net/docs
```

---

## 📊 Step 7: Monitoring & Scaling

### 7.1 View Logs
```bash
# Stream live logs
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Download logs
az webapp log download --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### 7.2 Scale Up (Vertical)
```bash
# Upgrade to higher tier for more CPU/RAM
az appservice plan update \
  --name ${APP_NAME}-plan \
  --resource-group $RESOURCE_GROUP \
  --sku P2V2
```

### 7.3 Scale Out (Horizontal)
```bash
# Add more instances
az appservice plan update \
  --name ${APP_NAME}-plan \
  --resource-group $RESOURCE_GROUP \
  --number-of-workers 3
```

### 7.4 Enable Auto-Scaling (Production)
```bash
# Create autoscale rule
az monitor autoscale create \
  --resource-group $RESOURCE_GROUP \
  --resource ${APP_NAME}-plan \
  --resource-type Microsoft.Web/serverfarms \
  --name ${APP_NAME}-autoscale \
  --min-count 2 \
  --max-count 5 \
  --count 2

# Add CPU-based scale rule
az monitor autoscale rule create \
  --resource-group $RESOURCE_GROUP \
  --autoscale-name ${APP_NAME}-autoscale \
  --condition "CpuPercentage > 70 avg 5m" \
  --scale out 1
```

---

## 🔄 Step 8: CI/CD Pipeline (Optional)

### GitHub Actions Deployment

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]

env:
  ACR_NAME: researchcopilotacr
  IMAGE_NAME: research-copilot
  RESOURCE_GROUP: research-copilot-rg
  APP_NAME: research-copilot

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Login to ACR
      run: az acr login --name ${{ env.ACR_NAME }}
    
    - name: Build and push image
      run: |
        docker build -t ${{ env.ACR_NAME }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }} .
        docker push ${{ env.ACR_NAME }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: ${{ env.APP_NAME }}
        images: ${{ env.ACR_NAME }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

**Setup GitHub Secrets:**
```bash
# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-sp" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth

# Add output JSON to GitHub Secrets as AZURE_CREDENTIALS
```

---

## 💰 Cost Optimization

### Development/Testing
- **App Service Plan**: B1 Basic (~$13/month)
- **ACR**: Basic (~$5/month)
- **Storage**: Pay-as-you-go (~$0.02/GB)
- **Total**: ~$20-30/month

### Production (Medium Scale)
- **App Service Plan**: P1V2 (~$90/month) with 2-5 auto-scale instances
- **ACR**: Standard (~$20/month)
- **Storage**: ~$5-10/month
- **Application Insights**: First 5GB free, then ~$2/GB
- **Total**: ~$150-300/month

### Cost-Saving Tips
```bash
# Stop app when not in use (dev)
az webapp stop --name $APP_NAME --resource-group $RESOURCE_GROUP

# Delete unused resources
az acr repository delete --name $ACR_NAME --image research-copilot:old-tag

# Use B-series VMs for development
# Use Azure Reserved Instances for 40% savings on production
```

---

## 🐛 Troubleshooting

### Issue: Container fails to start
```bash
# Check container logs
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Common fixes:
# 1. Verify WEBSITES_PORT=8000 is set
# 2. Check Docker image runs locally
# 3. Verify environment variables are set correctly
```

### Issue: 502 Bad Gateway
```bash
# Check application logs for startup errors
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Verify health endpoint
curl https://${APP_NAME}.azurewebsites.net/api/v1/health

# Restart app
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### Issue: Slow API responses
```bash
# Check App Insights for performance metrics
# Upgrade to higher SKU
az appservice plan update --name ${APP_NAME}-plan --resource-group $RESOURCE_GROUP --sku P2V2

# Enable auto-scaling (see step 7.4)
```

---

## 🔍 Performance Optimization

### 1. Enable Azure CDN (for static assets)
```bash
az cdn profile create \
  --name ${APP_NAME}-cdn \
  --resource-group $RESOURCE_GROUP \
  --sku Standard_Microsoft
```

### 2. Use Azure Redis Cache (for LLM response caching)
```bash
az redis create \
  --name ${APP_NAME}-redis \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Basic \
  --vm-size c0
```

### 3. Configure Connection Pooling
Add to app settings:
```bash
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    OPENAI_MAX_RETRIES="3" \
    OPENAI_TIMEOUT="30"
```

---

## 📚 Additional Resources

- [Azure App Service Docs](https://learn.microsoft.com/azure/app-service/)
- [Azure Container Registry](https://learn.microsoft.com/azure/container-registry/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)

---

## ✅ Deployment Checklist

- [ ] Azure CLI installed and logged in
- [ ] Docker Desktop running
- [ ] OpenAI API key obtained
- [ ] Resource group created
- [ ] ACR created and image pushed
- [ ] Azure Blob Storage created
- [ ] App Service created and configured
- [ ] Environment variables set
- [ ] HTTPS enabled
- [ ] Health check passing
- [ ] API documentation accessible
- [ ] Application Insights enabled
- [ ] Auto-scaling configured (production)
- [ ] Custom domain configured (optional)
- [ ] CI/CD pipeline setup (optional)

---

## 🎉 Success!

Your Project Research Copilot is now live at:
```
https://research-copilot.azurewebsites.net
```

**Next Steps:**
1. Test all endpoints via `/docs`
2. Monitor Application Insights for performance
3. Set up alerts for errors/high CPU
4. Configure custom domain (optional)
5. Implement rate limiting for production use

---

**Support:** For issues, check logs with `az webapp log tail` or open an issue on GitHub.
