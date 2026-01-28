# Azure Deployment Script for Research Copilot
# Run this script after installing Azure CLI: winget install Microsoft.AzureCLI

# Configuration Variables
$RESOURCE_GROUP = "research-copilot-rg"
$LOCATION = "eastus"
$APP_NAME = "research-copilot-$(Get-Random -Maximum 9999)"  # Unique name
$ACR_NAME = "researchcopilot$(Get-Random -Maximum 9999)"  # Lowercase, unique
$STORAGE_ACCOUNT = "researchstorage$(Get-Random -Maximum 99999)"  # Lowercase, unique

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Azure Deployment for Research Copilot" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Login to Azure
Write-Host "[1/8] Logging into Azure..." -ForegroundColor Yellow
az login
if ($LASTEXITCODE -ne 0) {
    Write-Host "Azure login failed. Please install Azure CLI first: winget install Microsoft.AzureCLI" -ForegroundColor Red
    exit 1
}

# Set subscription (optional - comment out if you only have one subscription)
# az account set --subscription "Your-Subscription-Name"

# Step 2: Create Resource Group
Write-Host "`n[2/8] Creating resource group: $RESOURCE_GROUP..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 3: Create Azure Container Registry
Write-Host "`n[3/8] Creating Azure Container Registry: $ACR_NAME..." -ForegroundColor Yellow
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --location $LOCATION `
    --admin-enabled true

# Get ACR credentials
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv

# Step 4: Build and Push Docker Image
Write-Host "`n[4/8] Building Docker image..." -ForegroundColor Yellow
az acr build `
    --registry $ACR_NAME `
    --image research-copilot:latest `
    --file Dockerfile `
    .

# Step 5: Create Azure Blob Storage
Write-Host "`n[5/8] Creating Azure Blob Storage: $STORAGE_ACCOUNT..." -ForegroundColor Yellow
az storage account create `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION `
    --sku Standard_LRS `
    --kind StorageV2

# Create container
az storage container create `
    --name research-papers `
    --account-name $STORAGE_ACCOUNT `
    --public-access off

# Get connection string
$STORAGE_CONN_STRING = az storage account show-connection-string `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --output tsv

# Step 6: Create App Service Plan
Write-Host "`n[6/8] Creating App Service Plan..." -ForegroundColor Yellow
az appservice plan create `
    --name "$APP_NAME-plan" `
    --resource-group $RESOURCE_GROUP `
    --sku B1 `
    --is-linux

# Step 7: Create Web App
Write-Host "`n[7/8] Creating Web App: $APP_NAME..." -ForegroundColor Yellow
az webapp create `
    --resource-group $RESOURCE_GROUP `
    --plan "$APP_NAME-plan" `
    --name $APP_NAME `
    --deployment-container-image-name "$ACR_NAME.azurecr.io/research-copilot:latest"

# Configure ACR credentials
az webapp config container set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-custom-image-name "$ACR_NAME.azurecr.io/research-copilot:latest" `
    --docker-registry-server-url "https://$ACR_NAME.azurecr.io" `
    --docker-registry-server-user $ACR_USERNAME `
    --docker-registry-server-password $ACR_PASSWORD

# Step 8: Configure Environment Variables
Write-Host "`n[8/8] Configuring environment variables..." -ForegroundColor Yellow
Write-Host "Please enter your OpenAI API key:" -ForegroundColor Yellow
$OPENAI_KEY = Read-Host

az webapp config appsettings set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --settings `
        APP_NAME="Project Research Copilot" `
        ENVIRONMENT="production" `
        DEBUG="false" `
        LOG_LEVEL="INFO" `
        OPENAI_API_KEY="$OPENAI_KEY" `
        AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN_STRING" `
        AZURE_STORAGE_CONTAINER_NAME="research-papers" `
        LLM_PROVIDER="openai" `
        LLM_MODEL="gpt-4-turbo-preview" `
        API_WORKERS="2" `
        PORT="8000" `
        ALLOWED_ORIGINS="*"

# Enable HTTPS only
az webapp update `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --https-only true

# Restart the app
Write-Host "`nRestarting web app..." -ForegroundColor Yellow
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

# Display results
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Deployment Summary:" -ForegroundColor Cyan
Write-Host "  Resource Group: $RESOURCE_GROUP"
Write-Host "  App Name: $APP_NAME"
Write-Host "  Container Registry: $ACR_NAME"
Write-Host "  Storage Account: $STORAGE_ACCOUNT"

Write-Host "`nYour application is live at:" -ForegroundColor Green
Write-Host "  https://$APP_NAME.azurewebsites.net" -ForegroundColor White

Write-Host "`nAPI Documentation:" -ForegroundColor Cyan
Write-Host "  https://$APP_NAME.azurewebsites.net/docs" -ForegroundColor White

Write-Host "`nTo view logs:" -ForegroundColor Yellow
Write-Host "  az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP" -ForegroundColor White

Write-Host "`nDeployment info saved to:" -ForegroundColor Cyan
Write-Host "  azure-deployment-info.txt" -ForegroundColor White

# Save deployment info
@"
Azure Deployment Information
=============================
Deployed: $(Get-Date)

Resource Group: $RESOURCE_GROUP
App Name: $APP_NAME
Container Registry: $ACR_NAME
Storage Account: $STORAGE_ACCOUNT

Application URL: https://$APP_NAME.azurewebsites.net
API Docs: https://$APP_NAME.azurewebsites.net/docs

Commands:
---------
# View logs
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Stop app
az webapp stop --name $APP_NAME --resource-group $RESOURCE_GROUP

# Start app
az webapp start --name $APP_NAME --resource-group $RESOURCE_GROUP

# Delete resources (to avoid charges)
az group delete --name $RESOURCE_GROUP --yes --no-wait
"@ | Out-File -FilePath "azure-deployment-info.txt" -Encoding utf8

Write-Host "`nDone! Your Research Copilot is now running on Azure!`n" -ForegroundColor Green
