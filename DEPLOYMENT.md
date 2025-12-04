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
- **Runtime:** `Static Site`
- **Build Command:** `echo 'Static files ready'`
- **Publish Directory:** `./` (root directory)

**Advanced Settings:**
- **Instance Type:** `Free` (or `Starter` for better performance)
- **Health Check Path:** `/complete_usl_system.html`
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

**Option 1: Direct Browser Opening**
```bash
# Simply open complete_usl_system.html in your browser
# All functionality works client-side
```

**Option 2: Local HTTP Server (Recommended)**
```bash
# Using Python's built-in server
python -m http.server 8000

# Or using Node.js
npx http-server -p 8000

# Open http://localhost:8000/complete_usl_system.html
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

**Why Static Site Deployment:**
- **Perfect for client-side HTML/JavaScript apps** - no server needed
- **Cost-effective** - free tier handles the complete application
- **Fast deployment** - no build process required
- **Global CDN** - automatically served via Render's CDN
- **Client-side Three.js** - 3D rendering happens in the browser

**Performance Optimizations:**
- **Browser caching** enabled for static assets
- **CDN delivery** for fast global access
- **No server overhead** - pure client-side processing
- **3D models loaded on-demand** via Three.js

## 🚨 Troubleshooting

**Model Loading Issues:**
- 3D models are loaded client-side via Three.js
- Ensure models are in the correct `usl_models/` directory structure
- Check browser console for Three.js loading errors
- Large model files may take time to load initially

**Performance Issues:**
- 3D models are large (>100MB); consider optimizing textures
- Use a modern browser with WebGL support
- Close other browser tabs to free up memory
- Models load progressively - be patient on first load

**Browser Compatibility:**
- Requires WebGL support (most modern browsers)
- Chrome/Edge recommended for best performance
- Ensure camera/microphone permissions are granted

## 🔄 Updating the Deployment

To update the deployed service:

1. Make changes to the code
2. Commit and push to GitHub
3. Go to Render dashboard → Manual Deploy → Deploy latest commit

## 📊 Monitoring

Static site health is monitored by checking that the main HTML file loads successfully. All processing happens client-side in the browser.

## 🎉 Success!

Once deployed, share the Render URL with users. The system will be accessible worldwide and can handle the complete USL clinical screening workflow with real 3D avatars!

---

*Built with: Three.js, JavaScript, HTML5, Real-time ML processing, FHIR standards*
