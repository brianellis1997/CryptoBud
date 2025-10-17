# CryptoBud Deployment Guide

This guide will help you deploy CryptoBud to various cloud platforms.

## Prerequisites

Before deploying, ensure:
1. Models are trained and saved in `saved_models/` directory
2. All dependencies are listed in `requirements.txt`
3. GitHub repository is set up and pushed

## Option 1: Streamlit Cloud (Recommended - Free)

Streamlit Cloud is the easiest way to deploy your dashboard for free.

### Steps:

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `brianellis1997/CryptoBud`
   - Set the main file path: `dashboard/app.py`
   - Click "Deploy"

4. **Configure secrets (if needed)**
   - In Streamlit Cloud dashboard, go to App settings > Secrets
   - Add any API keys or configuration

### Important Notes:
- Free tier includes 1GB storage and limited compute
- Apps sleep after inactivity but wake up on access
- Models must be included in the repository (consider using Git LFS for large files)

## Option 2: Railway

Railway offers $5 free credits per month.

### Steps:

1. **Create a `railway.toml` file** (already done)

2. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

3. **Login and deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

4. **Set environment variables**
   ```bash
   railway variables set PORT=8501
   ```

### Procfile for Railway:
Create a file named `Procfile`:
```
web: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```

## Option 3: Render

Render offers a free tier with some limitations.

### Steps:

1. **Push code to GitHub**

2. **Create a new Web Service on Render**
   - Go to [render.com](https://render.com)
   - Click "New" > "Web Service"
   - Connect your GitHub repository

3. **Configure the service**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0`

4. **Deploy**
   - Click "Create Web Service"

## Option 4: Google Cloud Run

For more control and scalability.

### Steps:

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8080

   CMD streamlit run dashboard/app.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Build and deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/cryptobud
   gcloud run deploy --image gcr.io/PROJECT_ID/cryptobud --platform managed
   ```

## Handling Large Model Files

If your model files are too large for GitHub (>100MB):

### Option A: Git LFS
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"
git add .gitattributes
git add saved_models/
git commit -m "Add models with LFS"
git push
```

### Option B: Download during deployment
Add a script to download pre-trained models from cloud storage (S3, Google Cloud Storage) during deployment.

## Environment Variables

Create a `.streamlit/secrets.toml` file for sensitive data:
```toml
[api]
coingecko_key = "your_api_key_here"
```

Never commit this file! It's already in `.gitignore`.

## Post-Deployment Checklist

- [ ] App loads without errors
- [ ] Real-time data fetching works
- [ ] Models load successfully
- [ ] Predictions are generated
- [ ] Charts display correctly
- [ ] Auto-refresh works (if enabled)
- [ ] Mobile responsive design works

## Troubleshooting

### "Module not found" errors
- Ensure all dependencies are in `requirements.txt`
- Check Python version compatibility (use 3.11)

### "Model file not found" errors
- Verify model files are in the repository
- Check file paths are relative, not absolute
- Consider using Git LFS for large files

### Out of memory errors
- Reduce model size or use quantization
- Deploy to a platform with more resources
- Implement lazy loading for models

### Slow loading times
- Cache data and predictions using `@st.cache_data`
- Optimize model inference
- Use smaller models for deployment

## Monitoring

Set up monitoring to track:
- App performance
- Error rates
- API usage
- User engagement

## Updating the App

To update your deployed app:
```bash
git add .
git commit -m "Update features"
git push origin main
```

Most platforms will automatically redeploy on push to main branch.

## Cost Considerations

- **Streamlit Cloud**: Free for public apps
- **Railway**: $5/month free credit
- **Render**: Free tier available with limitations
- **Google Cloud Run**: Pay-per-use, can be kept in free tier with low traffic

## Support

For issues or questions:
- Check the [GitHub Issues](https://github.com/brianellis1997/CryptoBud/issues)
- Refer to [Streamlit Docs](https://docs.streamlit.io)
- Review platform-specific documentation
