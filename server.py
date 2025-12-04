#!/usr/bin/env python3
"""
Simple HTTP server to serve the USL Clinical Screening System
Serves the complete_usl_system.html file and static assets
"""

from flask import Flask, send_from_directory, send_file, render_template_string
import os
from pathlib import Path

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

@app.route('/')
def index():
    """Serve the main USL system"""
    html_file = BASE_DIR / "complete_usl_system.html"
    if html_file.exists():
        return send_file(html_file, mimetype='text/html')
    else:
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h1>USL Clinical Screening System</h1>
            <p>The main HTML file (complete_usl_system.html) was not found.</p>
            <p>Please ensure the complete_usl_system.html file is in the same directory as this server.</p>
            <hr>
            <h3>Available files:</h3>
            <ul>
        """ + "".join([f"<li>{f}</li>" for f in os.listdir(BASE_DIR) if f.endswith('.html')]) + """
            </ul>
        </body>
        </html>
        """, 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files including 3D models"""
    return send_from_directory(BASE_DIR, filename)

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "service": "usl-clinical-screening"}

@app.route('/api/health')
def api_health():
    """API health check"""
    return {
        "status": "ok",
        "system": "USL Clinical Screening System",
        "version": "1.0.0",
        "features": [
            "Real-time USL recognition",
            "3D avatar synthesis",
            "Infectious disease screening",
            "FHIR bundle export",
            "Multi-language support"
        ]
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting USL Clinical Screening System server on port {port}")
    print("Serving files from:", BASE_DIR)
    print("Open http://localhost:5000 to access the system")

    # List available HTML files
    html_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.html')]
    print(f"Available HTML files: {html_files}")

    app.run(host='0.0.0.0', port=port, debug=False)
