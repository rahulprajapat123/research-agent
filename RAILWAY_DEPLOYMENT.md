# Railway Deployment Guide

This guide will help you deploy the RAG Research Intelligence System to Railway from GitHub.

## Prerequisites

1. A GitHub account
2. A Railway account (sign up at https://railway.app)
3. OpenAI API key
4. Azure Blob Storage account (for paper caching)

## Deployment Steps

### 1. Push Code to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Ready for Railway deployment"

# Add GitHub remote (replace with your repository URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy on Railway

1. Go to https://railway.app and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub account (if not already connected)
5. Select your repository
6. Railway will automatically detect the Python project

### 3. Configure Environment Variables

In Railway dashboard, go to your project → **Variables** tab and add:

#### Required Variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection string
- `AZURE_STORAGE_CONTAINER_NAME` - Container name (e.g., `research-papers`)

#### Optional but Recommended:
- `ENVIRONMENT=production`
- `DEBUG=false`
- `LOG_LEVEL=INFO`
- `ALLOWED_ORIGINS=*` (or specify your domain)
- `LLM_PROVIDER=openai`
- `LLM_MODEL=gpt-4`

### 4. Deploy

Railway will automatically:
- Install dependencies from `requirements.txt`
- Use the `Procfile` to start the application
- Assign a public URL to your application

### 5. Access Your Application

Once deployed, Railway will provide a URL like:
`https://your-app-name.railway.app`

- **Frontend UI**: `https://your-app-name.railway.app/`
- **API Docs**: `https://your-app-name.railway.app/docs`
- **Health Check**: `https://your-app-name.railway.app/api/v1/health`

## Getting Azure Blob Storage

1. Go to https://portal.azure.com
2. Create a **Storage Account**
3. Create a **Container** named `research-papers`
4. Get connection string from **Access keys** section
5. Add to Railway environment variables

## Getting OpenAI API Key

1. Go to https://platform.openai.com
2. Navigate to **API keys**
3. Create a new key
4. Add to Railway environment variables

## Troubleshooting

### Check Logs
In Railway dashboard, click on your service → **Logs** tab

### Common Issues

1. **Port binding error**: Railway automatically sets `$PORT` - no action needed
2. **Missing dependencies**: Check `requirements.txt` is complete
3. **API key errors**: Verify environment variables are set correctly
4. **Health check timeout**: Initial startup may take 2-3 minutes

### Redeploy

Railway automatically redeploys when you push to GitHub:
```bash
git add .
git commit -m "Update deployment"
git push
```

## Custom Domain (Optional)

1. In Railway dashboard, go to **Settings** → **Domains**
2. Click **Add Domain**
3. Follow instructions to configure DNS

## Cost Estimate

- **Railway**: Free tier includes $5/month credit (sufficient for small apps)
- **OpenAI API**: Pay per use (~$0.01-0.10 per request)
- **Azure Storage**: ~$0.018/GB per month

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: Create an issue on GitHub
