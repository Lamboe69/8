# USL Clinical Screening System - Render Deployment Guide

## 🚀 Quick Deployment

The system is now ready for deployment on Render! Here's how to deploy it:

### Step 1: Connect to GitHub
1. Go to [Render.com](https://render.com) and sign in/sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Search for and select the repository: `Lamboe69/8`
5. Choose the branch: `master`

### Step 2: Configure the Service
Use these settings:

**Basic Settings:**
- **Name:** `usl-clinical-screening` (or your preferred name)
- **Runtime:** `Python`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python server.py`

**Environment Variables:**
- **PORT:** `$PORT` (automatically set by Render)
- **PYTHON_VERSION:** `3.9.18`

**Advanced Settings:**
- **Instance Type:** `Free` (or `Starter` for better performance)
- **Health Check Path:** `/health`
- **Auto-Deploy:** `No` (for manual control)

### Step 3: Deploy
1. Click "Create Web Service"
2. Wait for the build to complete (may take 5-10 minutes due to 3D model files)
3. Once deployed, you'll get a URL like: `https://usl-clinical-screening.onrender.com`

## 📋 System Features

Once deployed, the system provides:

- 🏥 **Complete USL Medical Interpreter** with real-time screening
- 🤖 **3D Avatar System** using FBX/OBJ models from `usl_models/` folder
- 📷 **Video & Camera Recognition** for USL sign language input
- 🩺 **Infectious Disease Screening** (Malaria, TB, Typhoid, Cholera, etc.)
- 🌍 **Multi-language Support** (English, Runyankole, Luganda)
- 📄 **FHIR Bundle Export** for electronic health records
- 🚨 **Emergency Protocol Detection** with triage scoring

## 🔧 Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py

# Open http://localhost:5000 in your browser
```

## 📁 File Structure

```
├── complete_usl_system.html    # Main application (client-side)
├── server.py                   # Flask server for deployment
├── app_updated.py             # Streamlit version (alternative)
├── render.yaml                # Render deployment config
├── requirements.txt           # Python dependencies
├── usl_models/                # 3D avatar models and textures
│   └── male avatar/
│       ├── fbx Clean.fbx      # FBX model file
│       ├── obj file.obj       # OBJ model file
│       ├── obj file.mtl       # Material file
│       └── textures/          # Texture files
└── *.html                     # Test and development files
```

## 🎯 Deployment Benefits

**Why Flask over Static Hosting:**
- Large 3D model files (>100MB) exceed static hosting limits
- Better caching and compression for 3D assets
- Health check endpoints for monitoring
- CORS handling for API calls

**Performance Optimizations:**
- Flask serves static files efficiently
- 3D models loaded on-demand
- Client-side Three.js rendering (no server GPU needed)
- Free tier suitable for demo/pilot deployments

## 🚨 Troubleshooting

**Build Failures:**
- Check that `requirements.txt` includes all dependencies
- Ensure Python version compatibility (3.9.18 recommended)

**Model Loading Issues:**
- 3D models are loaded client-side via Three.js
- No server-side GPU requirements
- Models served as static files

**Performance Issues:**
- Free tier has memory limits (~512MB)
- Consider upgrading to Starter tier for production use
- 3D models are large; optimize textures if needed

## 🔄 Updating the Deployment

To update the deployed service:

1. Make changes to the code
2. Commit and push to GitHub
3. Go to Render dashboard → Manual Deploy → Deploy latest commit

## 📊 Monitoring

The deployment includes health check endpoints:
- `/health` - Basic health check
- `/api/health` - Detailed system status

## 🎉 Success!

Once deployed, share the Render URL with users. The system will be accessible worldwide and can handle the complete USL clinical screening workflow with real 3D avatars!

---

*Built with: Three.js, Flask, Python, Real-time ML processing, FHIR standards*
