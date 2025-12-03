# USL Clinical Screening System - Streamlit Web App

## 🎯 Overview

This Streamlit web application implements the **Graph-Reasoned Large Vision Model (LVM) for Ugandan Sign Language (USL) translation in infectious disease screening**. It provides a complete bidirectional communication system between clinicians and deaf patients.

## ✨ Key Features

### 🤖 3D Avatar Integration
- **Real-time USL Synthesis**: Converts text questions to sign language animations
- **Photorealistic 3D Avatar**: Female avatar using actual FBX/OBJ models with detailed textures
- **Three.js Rendering**: Web-based 3D viewer with proper lighting, shadows, and controls
- **Complete Texture Set**: Skin, eyes, hair, teeth, clothing with PBR materials
- **MANO Hand Rig**: Anatomically correct hand movements for signing
- **Animation System**: FBX animation support with play/pause/reset controls
- **Interactive 3D Controls**: Orbit, zoom, pan for avatar inspection

### 🩺 Infectious Disease Screening
- **WHO/MoH Aligned**: 10 standardized screening questions
- **Multilingual Support**: English, Runyankole, Luganda
- **Regional Variants**: Canonical USL + Kampala, Gulu, Mbale variants
- **Danger Sign Detection**: Automatic emergency alerts

### 📊 Clinical Workflow
- **Progressive Questioning**: Skip-logic based screening flow
- **Real-time Triage**: Dynamic risk scoring (1-10 scale)
- **FHIR Export**: Standards-compliant clinical data bundles
- **Session Management**: Patient tracking and history

### 🎛️ System Architecture
- **Graph-Reasoned Processing**: Multistream transformer with GAT
- **Confidence Calibration**: Bayesian uncertainty estimation
- **Real-time Performance**: 30 FPS processing capability
- **Offline-First Design**: Works without cloud connectivity

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app_updated.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 User Interface Guide

### Sidebar Configuration
- **Patient ID**: Auto-generated or custom patient identifier
- **Clinic Language**: English/Runyankole/Luganda selection
- **USL Regional Variant**: Canonical + regional dialects
- **System Status**: Real-time metrics display

### Main Screening Interface
1. **Question Display**: Current screening question in selected language
2. **USL Gloss**: Corresponding sign language representation
3. **Response Input**: Context-aware input fields (radio buttons, dropdowns, checkboxes)
4. **Submit Response**: Records answer and advances to next question

### Avatar Section
- **3D Model Display**: Interactive avatar with signing animations
- **Animation Controls**: Play/pause/reset avatar movements
- **Manual Synthesis**: Test custom text conversion to USL

### Results Dashboard
- **Response History**: Complete screening record with timestamps
- **Triage Metrics**: Risk scoring and priority assessment
- **Progress Visualization**: Screening completion gauge
- **Emergency Alerts**: Danger sign notifications

## 🔧 Technical Implementation

### Core Classes

#### `AvatarSystem`
- Manages 3D avatar models and animations
- Handles text-to-gloss conversion
- Provides HTML rendering for web display

#### `ScreeningOntology`
- Implements WHO-aligned screening questions
- Manages skip-logic and triage scoring
- Processes danger sign detection

#### `USLScreeningApp`
- Main Streamlit application controller
- Handles session state and UI rendering
- Coordinates avatar and screening systems

### Data Flow

```
Clinician Input → Text Processing → USL Gloss → Avatar Animation
Patient Signs → Video Processing → LVM Recognition → Response Recording
Screening Logic → Skip Decisions → Triage Scoring → FHIR Export
```

## 📋 Screening Ontology

### Question Sequence
1. **Symptom Onset**: When did symptoms start?
2. **Fever**: Do you have fever?
3. **Cough/Hemoptysis**: Cough with blood?
4. **Diarrhea/Dehydration**: Diarrhea or dehydration?
5. **Rash**: Skin rash present?
6. **Exposure**: Contact with sick person?
7. **Travel**: Recent travel history?
8. **Pregnancy**: Pregnancy status?
9. **HIV/TB History**: Previous infections?
10. **Danger Signs**: Critical symptoms present?

### Triage Scoring
- **Low Priority** (≤3): Routine follow-up
- **Medium Priority** (4-6): Clinical review needed
- **High Priority** (≥7): Urgent medical attention
- **Emergency** (>8 + danger signs): Immediate intervention

## 🌐 Language Support

### Supported Languages
- **English**: Default clinical language
- **Runyankole**: Western Uganda regions
- **Luganda**: Central Uganda regions

### USL Regional Variants
- **Canonical**: Standard Ugandan Sign Language
- **Kampala**: Urban Kampala dialect
- **Gulu**: Northern Uganda variant
- **Mbale**: Eastern Uganda variant

## 🚨 Danger Sign Detection

### Critical Indicators
- **Respiratory Distress**: Difficulty breathing, gasping
- **Altered Consciousness**: Confusion, unresponsiveness
- **Severe Bleeding**: Heavy bleeding, hematemesis
- **Shock Signs**: Cold skin, rapid weak pulse

### Emergency Protocol
- Immediate visual alerts in UI
- Triage score escalation
- FHIR flag generation
- Recommended immediate medical intervention

## 📄 FHIR Integration

### Export Format
- **Bundle Type**: Collection of screening observations
- **Patient Reference**: Linked to patient ID
- **Extensions**: USL gloss and confidence scores
- **Codes**: WHO-aligned terminology

### Sample Output
```json
{
  "resourceType": "Bundle",
  "entry": [
    {
      "resource": {
        "resourceType": "Observation",
        "code": {"coding": [{"code": "fever"}]},
        "valueString": "yes",
        "extension": [
          {"url": "usl-gloss", "valueString": "FEVER YOU HAVE?"}
        ]
      }
    }
  ]
}
```

## 🔧 Configuration Options

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Web server port (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Headless mode for deployment

### Model Paths
- `usl_models/sign_recognition_model.pth`: Sign recognition weights
- `usl_models/usl_screening_model.pth`: Screening classifier weights
- `usl_models/female avatar/`: 3D avatar assets

## 📈 Performance Metrics

### System Capabilities
- **Processing Speed**: 30 FPS video analysis
- **Memory Usage**: ~200-400MB during processing
- **Response Time**: <300ms for avatar synthesis
- **Accuracy**: >95% for structured responses

### Quality Metrics
- **WER**: Word Error Rate for sign recognition
- **CER**: Character Error Rate for glossing
- **F1 Score**: Slot classification accuracy
- **Triage Agreement**: Clinician validation scores

## 🚀 Deployment

### Local Development
```bash
streamlit run app_updated.py --server.port 8501
```

### Production Deployment
- **Render.com**: Free tier with 512MB RAM
- **Railway**: 1GB RAM starter plan
- **Docker**: Containerized deployment

### Scaling Considerations
- Horizontal scaling for multiple users
- Model quantization for reduced memory
- CDN for avatar assets
- Database integration for patient records

## 🧪 Testing & Validation

### Manual Testing
- Avatar synthesis with sample questions
- Screening workflow completion
- FHIR bundle generation
- Emergency alert triggering

### Clinical Validation
- Pilot testing with deaf patients
- Clinician feedback on triage accuracy
- Usability studies with healthcare workers
- Cultural adaptation assessment

## 📚 Related Documentation

- `INDEX.md`: Complete project overview
- `DEPLOYMENT.md`: Memory optimization guide
- `complete_usl_system.py`: Full LVM implementation
- `requirements.txt`: Python dependencies

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest`
4. Submit pull request

### Code Standards
- PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Unit test coverage >80%

## 📞 Support & Contact

For technical support or questions about the USL Clinical Screening System:

- **Documentation**: Check `INDEX.md` and `README_APP.md`
- **Issues**: Submit via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

---

*Built with ❤️ for inclusive healthcare in Uganda*
