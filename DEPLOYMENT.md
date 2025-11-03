# Streamlit Cloud Deployment Guide

## Prerequisites

- GitHub account
- Repository pushed to GitHub: https://github.com/siv3sh/LeadWalntu
- Streamlit Cloud account (free - sign up at https://share.streamlit.io)

## Deployment Files

The following files are configured for Streamlit Cloud deployment:

### 1. `.python-version`
Specifies Python version (3.11.5) for cloud environment.

### 2. `requirements.txt`
All dependencies with pinned versions:
- streamlit==1.28.0
- pandas==2.0.3
- numpy==1.24.3
- beautifulsoup4==4.12.2
- textstat==0.7.3
- scikit-learn==1.3.0
- sentence-transformers==2.2.2
- matplotlib==3.7.2
- seaborn==0.12.2
- plotly==5.17.0
- nltk==3.8.1
- requests==2.31.0
- torch==2.0.1
- pillow==10.0.0

### 3. `packages.txt`
System-level dependencies (build-essential).

### 4. `.streamlit/config.toml`
Streamlit configuration for professional theme and server settings.

## Step-by-Step Deployment

### Step 1: Access Streamlit Cloud

1. Visit: https://share.streamlit.io
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub repositories

### Step 2: Create New App

1. Click "New app" button
2. Select deployment type: "From existing repo"

### Step 3: Configure Deployment

Fill in the following settings:

**Repository:**
```
siv3sh/LeadWalntu
```

**Branch:**
```
main
```

**Main file path:**
```
app.py
```

**App URL (optional):**
```
leadwalntu
```
(This will create: https://leadwalntu.streamlit.app)

### Step 4: Advanced Settings (Optional)

Click "Advanced settings" if you need to configure:

- **Python version**: 3.11 (auto-detected from .python-version)
- **Secrets**: None required for this app
- **Resources**: Default (1 vCPU, 800 MB RAM) is sufficient

### Step 5: Deploy

1. Click "Deploy!" button
2. Wait for deployment (typically 2-5 minutes)
3. Watch the build logs for any issues

## Expected Build Process

The deployment will:

1. ✅ Clone repository from GitHub
2. ✅ Set up Python 3.11 environment
3. ✅ Install system packages (build-essential)
4. ✅ Install Python packages from requirements.txt
5. ✅ Download NLTK data (punkt tokenizer)
6. ✅ Download Sentence Transformer model (all-MiniLM-L6-v2)
7. ✅ Start Streamlit server
8. ✅ App becomes available at your URL

**Build time**: Approximately 3-5 minutes

## Deployment URL

Once deployed, your app will be available at:
```
https://leadwalntu.streamlit.app
```
(or your custom URL)

## Post-Deployment

### Update README

After successful deployment, update README.md with the live URL:

```markdown
## Live Demo

🚀 **Live Application**: https://leadwalntu.streamlit.app

Try the SEO Content Quality Analyzer live!
```

### Monitor Performance

Access your app dashboard at:
```
https://share.streamlit.io/
```

Monitor:
- **Usage stats**: Visitors, page views
- **Logs**: Real-time application logs
- **Resources**: CPU and memory usage
- **Errors**: Any runtime issues

### Automatic Redeployment

The app automatically redeploys when you push to the main branch:

1. Make changes locally
2. Commit and push to GitHub
3. Streamlit Cloud detects changes
4. Automatic rebuild and redeploy (1-2 minutes)

## Data Handling for Cloud

### Important Notes

⚠️ **Large Data Files**

The repository excludes large data files (data.csv, extracted_content.csv, features.csv) via .gitignore.

For the cloud deployment:

**Option 1: Use Sample Data (Recommended for Demo)**
- The app includes sample_data.csv for testing
- Users can upload their own HTML files
- Live Analysis page works without any data files

**Option 2: Process Data Separately**
- Run the Jupyter notebook locally first
- Upload processed files (features.csv, duplicates.csv) to GitHub
- Update .gitignore to track these files

**Option 3: Use Streamlit Secrets for Data URLs**
- Store data on external hosting (Google Drive, S3, etc.)
- Add download URLs to Streamlit secrets
- Modify app.py to fetch data on startup

### Current Configuration

The deployed app will:
- ✅ Load sample data if available
- ✅ Show "no data" message for dashboard
- ✅ Provide "Process Data Now" option (for small datasets)
- ✅ Fully functional Live Analysis page (no data required)

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError"
- **Solution**: Ensure all packages in requirements.txt have correct versions
- Check build logs for specific missing package

**Issue**: "Memory limit exceeded"
- **Solution**: Process data in smaller batches
- Consider upgrading to Streamlit Cloud paid plan

**Issue**: "Build timeout"
- **Solution**: Usually resolves on retry
- Check if requirements.txt has version conflicts

**Issue**: App loads slowly
- **Solution**: First load downloads transformer models (~80MB)
- Subsequent loads are cached and faster

### Getting Help

1. **Build Logs**: Check detailed logs in Streamlit Cloud dashboard
2. **Community Forum**: https://discuss.streamlit.io
3. **Documentation**: https://docs.streamlit.io/streamlit-community-cloud
4. **GitHub Issues**: https://github.com/siv3sh/LeadWalntu/issues

## Security Best Practices

### What's Already Configured

✅ No API keys in repository  
✅ Large data files excluded  
✅ Secrets placeholder in .gitignore  
✅ CORS disabled for security  
✅ No sensitive data exposed  

### If You Add Features

If you add external APIs or databases:

1. **Never commit secrets to Git**
2. Use Streamlit Secrets Manager:
   - Go to app settings in Streamlit Cloud
   - Add secrets in TOML format
   - Access in code: `st.secrets["api_key"]`

3. **Example secrets.toml format**:
```toml
# Don't commit this file!
api_key = "your-api-key-here"
database_url = "postgresql://..."
```

## Performance Optimization

### Caching

The app already uses Streamlit caching:

```python
@st.cache_data
def load_data():
    # Cached after first load
    
@st.cache_resource
def load_model():
    # Shared across users
```

### Resource Limits

**Free Tier**:
- 1 vCPU
- 800 MB RAM
- Unlimited public apps
- Auto-sleep after inactivity

**Paid Tier** (if needed):
- More CPU/RAM
- No auto-sleep
- Priority support

## Maintenance

### Regular Updates

1. **Dependencies**: Update packages periodically
2. **Security**: Monitor for vulnerabilities
3. **Data**: Refresh sample data if needed
4. **Documentation**: Keep README current

### Monitoring

Set up monitoring:
- Check logs weekly
- Monitor error rates
- Track usage patterns
- Update based on user feedback

## Success Checklist

- [ ] Repository pushed to GitHub
- [ ] All deployment files committed (.python-version, packages.txt, config.toml)
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] Deployment URL accessible
- [ ] All pages working correctly
- [ ] Live Analysis functional
- [ ] README updated with live URL
- [ ] App tested in production

## Next Steps After Deployment

1. ✅ Test all features in production
2. ✅ Update README with live demo link
3. ✅ Share URL in project submission
4. ✅ Monitor initial usage and errors
5. ✅ Gather feedback and iterate

---

**Deployment Support**: For issues, check logs first, then consult Streamlit documentation or community forum.

**Repository**: https://github.com/siv3sh/LeadWalntu  
**Streamlit Cloud**: https://share.streamlit.io  
**Documentation**: https://docs.streamlit.io/streamlit-community-cloud

---

Good luck with your deployment! 🚀
