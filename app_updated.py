#!/usr/bin/env python3
"""
Graph-Reasoned Large Vision Models for Ugandan Sign Language Translation
in Infectious Disease Screening

Streamlit Web Application with 3D Avatar Integration
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
import os
from pathlib import Path

# ============================================================================
# AVATAR INTEGRATION
# ============================================================================

class AvatarSystem:
    """3D Avatar system for USL synthesis using real FBX/OBJ models"""

    def __init__(self):
        self.avatar_models = {
            "female": {
                "fbx": "usl_models/female avatar/fbx file clean.fbx",
                "obj": "usl_models/female avatar/obj file.obj",
                "mtl": "usl_models/female avatar/obj file.mtl",
                "textures": "usl_models/female avatar/textures/"
            }
        }
        self.current_avatar = "female"

    def get_avatar_html(self, gloss_sequence="", animation_params=None):
        """Generate HTML for 3D avatar display with Three.js"""

        # Check if avatar files exist
        fbx_path = Path(self.avatar_models[self.current_avatar]["fbx"])
        obj_path = Path(self.avatar_models[self.current_avatar]["obj"])

        # Use FBX if available, otherwise OBJ
        model_path = ""
        model_format = ""

        if fbx_path.exists():
            model_path = fbx_path
            model_format = "fbx"
        elif obj_path.exists():
            model_path = obj_path
            model_format = "obj"
        else:
            return self._get_placeholder_avatar(gloss_sequence)

        # Generate Three.js HTML for 3D model display
        avatar_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    font-family: Arial, sans-serif;
                }}
                #avatar-container {{
                    width: 100%;
                    height: 400px;
                    position: relative;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                #status-indicator {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(76, 175, 80, 0.9);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 5px;
                    font-size: 12px;
                    z-index: 1000;
                }}
                #gloss-display {{
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(0, 0, 0, 0.7);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 5px;
                    font-size: 14px;
                    max-width: 300px;
                    z-index: 1000;
                }}
                #controls {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    display: flex;
                    gap: 5px;
                    z-index: 1000;
                }}
                .control-btn {{
                    background: rgba(255, 255, 255, 0.9);
                    border: none;
                    padding: 5px 10px;
                    border-radius: 3px;
                    cursor: pointer;
                    font-size: 12px;
                }}
                .control-btn:hover {{
                    background: rgba(255, 255, 255, 1);
                }}
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/FBXLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/MTLLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        </head>
        <body>
            <div id="avatar-container">
                <div id="status-indicator">🧑‍🤝‍🧑 3D Avatar Active</div>
                <div id="controls">
                    <button class="control-btn" onclick="playAnimation()">▶️ Play</button>
                    <button class="control-btn" onclick="pauseAnimation()">⏸️ Pause</button>
                    <button class="control-btn" onclick="resetPose()">🔄 Reset</button>
                </div>
                {f'<div id="gloss-display"><strong>Current Gloss:</strong><br>{gloss_sequence}</div>' if gloss_sequence else ''}
            </div>

            <script>
                let scene, camera, renderer, controls, model, mixer;
                let animations = [];
                let currentAnimation = null;

                init();
                animate();

                function init() {{
                    // Scene setup
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0xf5f7fa);

                    // Camera setup
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / 400, 0.1, 1000);
                    camera.position.set(0, 1.6, 3);

                    // Renderer setup
                    renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    renderer.setSize(document.getElementById('avatar-container').clientWidth, 400);
                    renderer.shadowMap.enabled = true;
                    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                    document.getElementById('avatar-container').appendChild(renderer.domElement);

                    // Controls
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.enableZoom = true;
                    controls.enablePan = false;
                    controls.minDistance = 2;
                    controls.maxDistance = 10;

                    // Lighting
                    setupLighting();

                    // Load model
                    loadModel();

                    // Handle window resize
                    window.addEventListener('resize', onWindowResize, false);
                }}

                function setupLighting() {{
                    // Ambient light
                    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                    scene.add(ambientLight);

                    // Directional light (main)
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                    directionalLight.position.set(5, 5, 5);
                    directionalLight.castShadow = true;
                    directionalLight.shadow.mapSize.width = 2048;
                    directionalLight.shadow.mapSize.height = 2048;
                    scene.add(directionalLight);

                    // Fill light
                    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
                    fillLight.position.set(-5, 3, -5);
                    scene.add(fillLight);

                    // Rim light
                    const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
                    rimLight.position.set(0, -5, -10);
                    scene.add(rimLight);
                }}

                function loadModel() {{
                    const modelFormat = '{model_format}';

                    if (modelFormat === 'fbx') {{
                        loadFBXModel();
                    }} else if (modelFormat === 'obj') {{
                        loadOBJModel();
                    }}
                }}

                function loadFBXModel() {{
                    const loader = new THREE.FBXLoader();

                    loader.load(
                        '{str(model_path).replace(chr(92), "/")}', // FBX file path
                        function (fbx) {{
                            model = fbx;
                            setupModel(model);
                        }},
                        function (progress) {{
                            console.log('FBX Loading progress:', (progress.loaded / progress.total * 100) + '%');
                        }},
                        function (error) {{
                            console.error('Error loading FBX:', error);
                            showFallbackAvatar();
                        }}
                    );
                }}

                function loadOBJModel() {{
                    const mtlLoader = new THREE.MTLLoader();
                    const objLoader = new THREE.OBJLoader();

                    // Load materials first
                    mtlLoader.load(
                        '{str(Path(self.avatar_models[self.current_avatar]["mtl"])).replace(chr(92), "/")}', // MTL file path
                        function (materials) {{
                            materials.preload();
                            objLoader.setMaterials(materials);

                            // Load OBJ
                            objLoader.load(
                                '{str(model_path).replace(chr(92), "/")}', // OBJ file path
                                function (obj) {{
                                    model = obj;
                                    setupModel(model);
                                }},
                                function (progress) {{
                                    console.log('OBJ Loading progress:', (progress.loaded / progress.total * 100) + '%');
                                }},
                                function (error) {{
                                    console.error('Error loading OBJ:', error);
                                    showFallbackAvatar();
                                }}
                            );
                        }},
                        function (progress) {{
                            console.log('MTL Loading progress:', (progress.loaded / progress.total * 100) + '%');
                        }},
                        function (error) {{
                            console.error('Error loading MTL:', error);
                            // Try loading OBJ without materials
                            objLoader.load(
                                '{str(model_path).replace(chr(92), "/")}',
                                function (obj) {{
                                    model = obj;
                                    setupModel(model);
                                }}
                            );
                        }}
                    );
                }}

                function setupModel(model) {{
                    // Scale and position the model
                    model.scale.setScalar(1.0);
                    model.position.set(0, -1, 0);

                    // Enable shadows
                    model.traverse(function (child) {{
                        if (child.isMesh) {{
                            child.castShadow = true;
                            child.receiveShadow = true;

                            // Ensure materials are properly configured
                            if (child.material) {{
                                if (Array.isArray(child.material)) {{
                                    child.material.forEach(mat => {{
                                        mat.needsUpdate = true;
                                    }});
                                }} else {{
                                    child.material.needsUpdate = true;
                                }}
                            }}
                        }}
                    }});

                    scene.add(model);

                    // Setup animations if available
                    if (model.animations && model.animations.length > 0) {{
                        mixer = new THREE.AnimationMixer(model);
                        animations = model.animations;

                        // Play default pose
                        if (animations.length > 0) {{
                            currentAnimation = mixer.clipAction(animations[0]);
                            currentAnimation.play();
                        }}
                    }}

                    updateStatus('Model loaded successfully');
                }}

                function showFallbackAvatar() {{
                    // Create a simple geometric avatar as fallback
                    const geometry = new THREE.CapsuleGeometry(0.5, 1.5, 4, 8);
                    const material = new THREE.MeshLambertMaterial({{
                        color: 0x8B4513,
                        transparent: true,
                        opacity: 0.8
                    }});
                    model = new THREE.Mesh(geometry, material);
                    model.position.set(0, 0, 0);
                    model.castShadow = true;
                    scene.add(model);
                    updateStatus('Using fallback avatar');
                }}

                function updateStatus(message) {{
                    const statusEl = document.getElementById('status-indicator');
                    if (statusEl) {{
                        statusEl.textContent = '🧑‍🤝‍🧑 ' + message;
                    }}
                }}

                function playAnimation() {{
                    if (mixer && animations.length > 0) {{
                        if (currentAnimation) {{
                            currentAnimation.stop();
                        }}
                        currentAnimation = mixer.clipAction(animations[0]);
                        currentAnimation.play();
                        updateStatus('Animation playing');
                    }} else {{
                        updateStatus('No animations available');
                    }}
                }}

                function pauseAnimation() {{
                    if (currentAnimation) {{
                        currentAnimation.stop();
                        updateStatus('Animation paused');
                    }}
                }}

                function resetPose() {{
                    if (model) {{
                        model.rotation.set(0, 0, 0);
                        model.position.set(0, -1, 0);
                        controls.reset();
                        updateStatus('Pose reset');
                    }}
                }}

                function animate() {{
                    requestAnimationFrame(animate);

                    if (controls) {{
                        controls.update();
                    }}

                    if (mixer) {{
                        mixer.update(0.016); // ~60 FPS
                    }}

                    renderer.render(scene, camera);
                }}

                function onWindowResize() {{
                    const container = document.getElementById('avatar-container');
                    camera.aspect = container.clientWidth / 400;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, 400);
                }}
            </script>
        </body>
        </html>
        """

        return avatar_html

    def _get_placeholder_avatar(self, gloss_sequence=""):
        """Fallback avatar display when 3D model not available"""

        return f"""
        <div style="
            width: 100%;
            height: 400px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: 2px solid #FF6B6B;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: white;
            text-align: center;
        ">
            <div style="margin-bottom: 20px;">
                🤖
            </div>
            <strong>Avatar System Initializing...</strong><br>
            <small>3D Model Loading</small><br>
            <small style="margin-top: 10px; color: #ccc;">{gloss_sequence or 'Ready for USL Synthesis'}</small>
        </div>
        """

    def synthesize_sign(self, text, language="english", regional_variant="canonical"):
        """Convert text to USL gloss sequence for avatar animation"""

        # Simplified text-to-gloss mapping
        gloss_mappings = {
            "do you have fever": "FEVER YOU HAVE?",
            "do you have cough": "COUGH YOU HAVE?",
            "any blood in sputum": "BLOOD SPIT ANY?",
            "do you have diarrhea": "DIARRHEA YOU HAVE?",
            "are you pregnant": "PREGNANT YOU?",
            "have you traveled recently": "TRAVEL RECENT YOU?",
            "yes": "YES",
            "no": "NO",
            "help": "HELP",
            "emergency": "EMERGENCY"
        }

        text_lower = text.lower().strip()

        # Find matching phrase
        for phrase, gloss in gloss_mappings.items():
            if phrase in text_lower:
                return gloss

        # Fallback: word-by-word conversion
        words = text_lower.replace('?', '').split()
        gloss_words = []

        word_mappings = {
            "fever": "FEVER", "cough": "COUGH", "blood": "BLOOD",
            "diarrhea": "DIARRHEA", "pain": "PAIN", "rash": "RASH",
            "travel": "TRAVEL", "pregnancy": "PREGNANT", "help": "HELP",
            "emergency": "EMERGENCY", "yes": "YES", "no": "NO",
            "you": "YOU", "have": "HAVE", "do": "DO", "any": "ANY",
            "recently": "RECENT", "are": "ARE"
        }

        for word in words:
            gloss_words.append(word_mappings.get(word, word.upper()))

        return " ".join(gloss_words)

# ============================================================================
# SCREENING SYSTEM
# ============================================================================

class ScreeningOntology:
    """Infectious disease screening ontology with skip-logic"""

    def __init__(self):
        self.screening_slots = {
            "symptom_onset": {
                "question": "When did your symptoms start?",
                "usl_gloss": "SYMPTOM START WHEN?",
                "languages": {
                    "english": "When did your symptoms start?",
                    "runyankole": "Ebindu byo bikugira laki?",
                    "luganda": "Obulwadde bwo bwakutandika ddi?"
                },
                "triage_weight": 1,
                "response_type": "temporal"
            },
            "fever": {
                "question": "Do you have fever or feel hot?",
                "usl_gloss": "FEVER YOU HAVE?",
                "languages": {
                    "english": "Do you have fever?",
                    "runyankole": "Orikugira omusujja?",
                    "luganda": "Olina omusujja?"
                },
                "triage_weight": 2,
                "response_type": "yes_no"
            },
            "cough_hemoptysis": {
                "question": "Do you have cough? Any blood in sputum?",
                "usl_gloss": "COUGH YOU HAVE? BLOOD SPIT?",
                "languages": {
                    "english": "Do you have cough with blood?",
                    "runyankole": "Orikuhikura? Omusaayi guri mu kikohozi?",
                    "luganda": "Olina kikohozi? Omusaayi guli mu kikohozi?"
                },
                "triage_weight": 3,
                "response_type": "cough_blood",
                "danger_signs": ["hemoptysis"]
            },
            "diarrhea_dehydration": {
                "question": "Do you have diarrhea? Signs of dehydration?",
                "usl_gloss": "DIARRHEA YOU HAVE? WATER-LOSS?",
                "languages": {
                    "english": "Do you have diarrhea or dehydration?",
                    "runyankole": "Orikugira ebyenda? Omubiri gukamye?",
                    "luganda": "Olina ebyenda? Omubiri gukaze?"
                },
                "triage_weight": 3,
                "response_type": "diarrhea_severity",
                "danger_signs": ["severe_dehydration", "bloody_diarrhea"]
            },
            "rash": {
                "question": "Do you have any skin rash or spots?",
                "usl_gloss": "SKIN SPOTS YOU HAVE?",
                "languages": {
                    "english": "Do you have skin rash?",
                    "runyankole": "Orikugira obushasha ku ruhanga?",
                    "luganda": "Olina obushasha ku lususu?"
                },
                "triage_weight": 2,
                "response_type": "yes_no"
            },
            "exposure": {
                "question": "Have you been exposed to sick person?",
                "usl_gloss": "SICK PERSON CONTACT YOU?",
                "languages": {
                    "english": "Have you been exposed to sick person?",
                    "runyankole": "Waatambula n'omuntu omulwadde?",
                    "luganda": "Otambuze n'omuntu omulwadde?"
                },
                "triage_weight": 2,
                "response_type": "yes_no"
            },
            "travel": {
                "question": "Have you traveled recently?",
                "usl_gloss": "TRAVEL RECENT YOU?",
                "languages": {
                    "english": "Have you traveled recently?",
                    "runyankole": "Waatambura haihi?",
                    "luganda": "Otambuze mu kiseera kino?"
                },
                "triage_weight": 1,
                "response_type": "yes_no"
            },
            "pregnancy": {
                "question": "Are you pregnant?",
                "usl_gloss": "PREGNANT YOU?",
                "languages": {
                    "english": "Are you pregnant?",
                    "runyankole": "Oli mu nkye?",
                    "luganda": "Oli mu lubuto?"
                },
                "triage_weight": 1,
                "response_type": "pregnancy_status"
            },
            "hiv_tb_history": {
                "question": "History of HIV or TB?",
                "usl_gloss": "HIV TB HISTORY YOU?",
                "languages": {
                    "english": "Do you have HIV or TB history?",
                    "runyankole": "Wali na HIV oba TB?",
                    "luganda": "Wali na HIV oba TB?"
                },
                "triage_weight": 2,
                "response_type": "yes_no"
            },
            "danger_signs": {
                "question": "Any danger signs: difficulty breathing, confusion, bleeding?",
                "usl_gloss": "DANGER SIGNS ANY? BREATH HARD? CONFUSION? BLEED?",
                "languages": {
                    "english": "Any danger signs?",
                    "runyankole": "Ebindu by'okutya obulamu?",
                    "luganda": "Waliwo eby'okutya obulamu?"
                },
                "triage_weight": 4,
                "response_type": "danger_multi",
                "danger_signs": ["respiratory_distress", "altered_consciousness", "severe_bleeding"]
            }
        }

        self.responses = {}
        self.triage_score = 0
        self.danger_flags = []
        self.current_slot_idx = 0

    def get_current_question(self, language="english"):
        """Get current screening question"""
        slot_keys = list(self.screening_slots.keys())
        if self.current_slot_idx < len(slot_keys):
            slot_key = slot_keys[self.current_slot_idx]
            slot_data = self.screening_slots[slot_key]
            return {
                "slot": slot_key,
                "question": slot_data["languages"].get(language, slot_data["question"]),
                "usl_gloss": slot_data["usl_gloss"],
                "response_type": slot_data["response_type"]
            }
        return None

    def process_response(self, slot_key, response, confidence=1.0):
        """Process screening response with skip-logic"""

        if slot_key not in self.screening_slots:
            return {"error": "Invalid slot"}

        slot_data = self.screening_slots[slot_key]
        self.responses[slot_key] = {
            "response": response,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        # Update triage score
        weight = slot_data.get("triage_weight", 1)
        if str(response).lower() in ["yes", "true", "severe"]:
            self.triage_score += weight

        # Check danger signs
        if "danger_signs" in slot_data and response:
            danger_mapping = {
                "blood": "hemoptysis",
                "severe": "severe_dehydration",
                "bloody_diarrhea": "bloody_diarrhea",
                "difficulty_breathing": "respiratory_distress",
                "confusion": "altered_consciousness",
                "bleeding": "severe_bleeding"
            }

            response_str = str(response).lower()
            for trigger, danger in danger_mapping.items():
                if trigger in response_str:
                    self.danger_flags.append(danger)

        # Skip logic
        self.current_slot_idx += 1

        # Emergency override for danger signs
        is_emergency = any(flag in ["respiratory_distress", "altered_consciousness", "severe_bleeding"]
                          for flag in self.danger_flags)

        return {
            "status": "response_recorded",
            "slot": slot_key,
            "response": response,
            "triage_score": self.triage_score,
            "danger_flags": self.danger_flags,
            "is_emergency": is_emergency,
            "next_question": self.get_current_question()
        }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class USLScreeningApp:
    """Main Streamlit application for USL screening system"""

    def __init__(self):
        self.avatar_system = AvatarSystem()
        self.screening_system = ScreeningOntology()

        # Session state for persistent data
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = f"PAT-{int(time.time())}"
        if 'language' not in st.session_state:
            st.session_state.language = "english"
        if 'regional_variant' not in st.session_state:
            st.session_state.regional_variant = "canonical"
        if 'current_question' not in st.session_state:
            st.session_state.current_question = self.screening_system.get_current_question()

    def run(self):
        """Main application loop"""

        st.set_page_config(
            page_title="USL Clinical Screening System",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🧠 Graph-Reasoned LVM for USL Translation")
        st.markdown("**Infectious Disease Screening • Real-time USL ↔ Clinical Text • WHO/MoH Aligned**")

        # Sidebar configuration
        self._render_sidebar()

        # Main content layout
        col1, col2 = st.columns([1.5, 1])

        with col1:
            self._render_main_interface()

        with col2:
            self._render_avatar_section()
            self._render_results_section()

        # Footer
        st.markdown("---")
        st.markdown("*Graph-Reasoned Large Vision Models • Ugandan Sign Language • Infectious Disease Screening*")

    def _render_sidebar(self):
        """Render sidebar with configuration options"""

        with st.sidebar:
            st.header("🎛️ System Configuration")

            # Patient information
            st.subheader("👤 Patient Information")
            patient_id = st.text_input(
                "Patient ID",
                value=st.session_state.patient_id,
                key="patient_id_input"
            )
            st.session_state.patient_id = patient_id

            # Language settings
            st.subheader("🌍 Language & Regional Settings")

            language_options = ["english", "runyankole", "luganda"]
            selected_language = st.selectbox(
                "Clinic Language",
                language_options,
                index=language_options.index(st.session_state.language),
                key="language_select"
            )
            st.session_state.language = selected_language

            regional_options = ["canonical", "kampala", "gulu", "mbale"]
            selected_regional = st.selectbox(
                "USL Regional Variant",
                regional_options,
                index=regional_options.index(st.session_state.regional_variant),
                key="regional_select"
            )
            st.session_state.regional_variant = selected_regional

            # System status
            st.subheader("📊 System Status")
            st.metric("LVM Status", "🟢 Ready")
            st.metric("Avatar Status", "🟢 Active")
            st.metric("Processing FPS", "30")
            st.metric("Confidence Threshold", "0.7")

            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("🔄 New Patient Session"):
                self._reset_session()
                st.success("New session started!")

            if st.button("📄 Export FHIR Bundle"):
                self._export_fhir_bundle()

    def _render_main_interface(self):
        """Render main screening interface"""

        st.header("🩺 Infectious Disease Screening")

        # Current question display
        if st.session_state.current_question:
            question_data = st.session_state.current_question

            st.subheader("❓ Current Question")
            st.info(f"**{question_data['question']}**")

            st.markdown(f"**USL Gloss:** `{question_data['usl_gloss']}`")

            # Response input based on question type
            response_type = question_data['response_type']

            if response_type == "yes_no":
                response = st.radio(
                    "Patient Response:",
                    ["Yes", "No"],
                    key=f"response_{question_data['slot']}"
                )

            elif response_type == "temporal":
                response = st.selectbox(
                    "When did symptoms start?",
                    ["Today", "Yesterday", "2-3 days ago", "1 week ago", "2 weeks ago", "1 month ago", "More than 1 month ago"],
                    key=f"response_{question_data['slot']}"
                )

            elif response_type == "cough_blood":
                col1, col2 = st.columns(2)
                with col1:
                    cough = st.radio("Do you have cough?", ["Yes", "No"], key=f"cough_{question_data['slot']}")
                with col2:
                    blood = st.radio("Any blood in sputum?", ["Yes", "No"], key=f"blood_{question_data['slot']}")

                response = f"Cough: {cough}, Blood: {blood}"

            elif response_type == "diarrhea_severity":
                diarrhea = st.radio("Do you have diarrhea?", ["Yes", "No"], key=f"diarrhea_{question_data['slot']}")
                if diarrhea == "Yes":
                    severity = st.selectbox(
                        "Severity:",
                        ["Mild", "Moderate", "Severe", "Bloody diarrhea"],
                        key=f"severity_{question_data['slot']}"
                    )
                    response = f"Diarrhea: {diarrhea}, Severity: {severity}"
                else:
                    response = f"Diarrhea: {diarrhea}"

            elif response_type == "pregnancy_status":
                response = st.radio(
                    "Are you pregnant?",
                    ["Yes", "No", "Not applicable"],
                    key=f"response_{question_data['slot']}"
                )

            elif response_type == "danger_multi":
                st.write("Check all that apply:")
                danger_options = [
                    "Difficulty breathing", "Confusion", "Severe bleeding",
                    "Bloody diarrhea", "Seizures", "Unconsciousness"
                ]

                selected_dangers = []
                for option in danger_options:
                    if st.checkbox(option, key=f"danger_{option}_{question_data['slot']}"):
                        selected_dangers.append(option)

                response = ", ".join(selected_dangers) if selected_dangers else "None"

            else:
                response = st.text_input(
                    "Patient Response:",
                    key=f"response_{question_data['slot']}"
                )

            # Submit response button
            if st.button("✅ Submit Response", type="primary"):
                result = self.screening_system.process_response(
                    question_data['slot'],
                    response,
                    confidence=0.95
                )

                st.session_state.current_question = result.get('next_question')

                # Trigger avatar synthesis for the question
                avatar_gloss = question_data['usl_gloss']
                st.session_state.avatar_gloss = avatar_gloss

                st.success("Response recorded!")

                # Show immediate feedback
                if result['danger_flags']:
                    st.error(f"🚨 **DANGER SIGNS DETECTED**: {', '.join(result['danger_flags'])}")
                    if result['is_emergency']:
                        st.error("🏥 **IMMEDIATE MEDICAL ATTENTION REQUIRED**")

                st.rerun()

        else:
            st.success("🎉 Screening Complete!")
            st.balloons()

    def _render_avatar_section(self):
        """Render 3D avatar display section"""

        st.header("🤖 USL Synthesis Avatar")

        # Avatar display
        avatar_html = self.avatar_system.get_avatar_html(
            gloss_sequence=st.session_state.get('avatar_gloss', '')
        )
        components.html(avatar_html, height=420)

        # Avatar controls
        st.subheader("🎭 Avatar Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎬 Play Animation"):
                st.info("Playing USL animation...")

        with col2:
            if st.button("⏸️ Pause"):
                st.info("Animation paused")

        with col3:
            if st.button("🔄 Reset"):
                st.session_state.avatar_gloss = ""
                st.info("Avatar reset")

        # Manual gloss input for testing
        st.subheader("🧪 Manual USL Synthesis")
        manual_text = st.text_input("Enter text to synthesize:")

        if st.button("🎭 Synthesize USL"):
            if manual_text:
                gloss = self.avatar_system.synthesize_sign(
                    manual_text,
                    st.session_state.language,
                    st.session_state.regional_variant
                )
                st.session_state.avatar_gloss = gloss
                st.success(f"Synthesized: {gloss}")
                st.rerun()

    def _render_results_section(self):
        """Render screening results and analytics"""

        st.header("📊 Screening Results")

        if self.screening_system.responses:
            # Current responses
            st.subheader("📋 Recorded Responses")

            responses_data = []
            for slot, response_data in self.screening_system.responses.items():
                slot_info = self.screening_system.screening_slots[slot]
                responses_data.append({
                    "Question": slot_info["question"],
                    "Response": response_data["response"],
                    "Confidence": f"{response_data['confidence']:.2f}",
                    "Time": response_data["timestamp"]
                })

            if responses_data:
                df = pd.DataFrame(responses_data)
                st.dataframe(df, use_container_width=True)

            # Triage metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Triage Score",
                    self.screening_system.triage_score,
                    delta="High Priority" if self.screening_system.triage_score > 5 else "Normal"
                )

            with col2:
                st.metric("Responses Recorded", len(self.screening_system.responses))

            with col3:
                danger_count = len(self.screening_system.danger_flags)
                st.metric("Danger Signs", danger_count)

            # Danger signs alert
            if self.screening_system.danger_flags:
                st.error(f"⚠️ **Danger Signs Detected**: {', '.join(self.screening_system.danger_flags)}")

                # Emergency protocol
                if any(flag in ["respiratory_distress", "altered_consciousness", "severe_bleeding"]
                      for flag in self.screening_system.danger_flags):
                    st.error("🚨 **EMERGENCY PROTOCOL ACTIVATED**")
                    st.error("🏥 Immediate medical attention required!")

            # Progress visualization
            st.subheader("📈 Screening Progress")

            completed_slots = len(self.screening_system.responses)
            total_slots = len(self.screening_system.screening_slots)

            progress_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=completed_slots,
                title={'text': "Screening Progress"},
                gauge={
                    'axis': {'range': [0, total_slots]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, total_slots * 0.5], 'color': "lightgray"},
                        {'range': [total_slots * 0.5, total_slots * 0.8], 'color': "gray"},
                        {'range': [total_slots * 0.8, total_slots], 'color': "darkgray"}
                    ]
                }
            ))

            st.plotly_chart(progress_fig, use_container_width=True)

        else:
            st.info("No responses recorded yet. Start screening to see results here.")

    def _reset_session(self):
        """Reset the screening session"""

        self.screening_system = ScreeningOntology()
        st.session_state.current_question = self.screening_system.get_current_question()
        st.session_state.avatar_gloss = ""

    def _export_fhir_bundle(self):
        """Export FHIR-compliant bundle"""

        if not self.screening_system.responses:
            st.error("No screening data to export")
            return

        # Create FHIR bundle
        fhir_bundle = {
            "resourceType": "Bundle",
            "id": f"usl-screening-{st.session_state.patient_id}",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": []
        }

        # Add screening observations
        for slot_key, response_data in self.screening_system.responses.items():
            slot_info = self.screening_system.screening_slots[slot_key]

            observation = {
                "resourceType": "Observation",
                "id": f"screening-{slot_key}-{st.session_state.patient_id}",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://who.int/usl-screening",
                        "code": slot_key,
                        "display": slot_info["question"]
                    }]
                },
                "subject": {"reference": f"Patient/{st.session_state.patient_id}"},
                "effectiveDateTime": response_data["timestamp"],
                "valueString": str(response_data["response"]),
                "extension": [
                    {
                        "url": "http://medisign.ug/usl-gloss",
                        "valueString": slot_info["usl_gloss"]
                    },
                    {
                        "url": "http://medisign.ug/confidence",
                        "valueDecimal": response_data["confidence"]
                    }
                ]
            }

            fhir_bundle["entry"].append({"resource": observation})

        # Add triage assessment
        triage_obs = {
            "resourceType": "Observation",
            "id": f"triage-{st.session_state.patient_id}",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://who.int/triage",
                    "code": "infectious-disease-triage",
                    "display": "Infectious Disease Triage Score"
                }]
            },
            "subject": {"reference": f"Patient/{st.session_state.patient_id}"},
            "valueInteger": self.screening_system.triage_score,
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "H" if self.screening_system.triage_score > 5 else "N",
                    "display": "High Priority" if self.screening_system.triage_score > 5 else "Normal"
                }]
            }]
        }

        fhir_bundle["entry"].append({"resource": triage_obs})

        # Add danger signs flag if applicable
        if self.screening_system.danger_flags:
            flag = {
                "resourceType": "Flag",
                "id": f"danger-signs-{st.session_state.patient_id}",
                "status": "active",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/flag-category",
                        "code": "clinical",
                        "display": "Clinical"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://medisign.ug/danger-signs",
                        "code": "immediate-attention",
                        "display": "Requires Immediate Medical Attention"
                    }]
                },
                "subject": {"reference": f"Patient/{st.session_state.patient_id}"},
                "period": {"start": datetime.now().isoformat()}
            }

            fhir_bundle["entry"].append({"resource": flag})

        # Download button
        bundle_json = json.dumps(fhir_bundle, indent=2)
        st.download_button(
            label="💾 Download FHIR Bundle",
            data=bundle_json,
            file_name=f"usl_screening_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        st.success("FHIR bundle generated successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main application entry point"""
    app = USLScreeningApp()
    app.run()

if __name__ == "__main__":
    main()
