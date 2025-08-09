// Global variables
let selectedFile = null;
const API_BASE_URL = 'http://localhost:8000';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultStatus = document.getElementById('resultStatus');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const recommendation = document.getElementById('recommendation');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadBtn = document.getElementById('downloadBtn');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // File input events
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Button events
    analyzeBtn.addEventListener('click', analyzeImage);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    downloadBtn.addEventListener('click', downloadReport);
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        displayImagePreview(file);
    } else {
        showError('Please select a valid image file (JPG, PNG, GIF)');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidImageFile(file)) {
            selectedFile = file;
            displayImagePreview(file);
        } else {
            showError('Please select a valid image file (JPG, PNG, GIF)');
        }
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    return validTypes.includes(file.type);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        
        // Hide upload area and show preview
        uploadArea.parentElement.style.display = 'none';
        previewSection.style.display = 'block';
        previewSection.classList.add('fade-in');
    };
    reader.readAsDataURL(file);
}

// Analysis functions
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch(`${API_BASE_URL}/predict/`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
        hideLoadingState();
    }
}

function showLoadingState() {
    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');
}

function hideLoadingState() {
    loadingSection.style.display = 'none';
}

function displayResults(result) {
    hideLoadingState();
    
    const isPneumonia = result.prediction === "Pneumonia Detected";
    const confidence = result.confidence;
    
    // Update result icon and status
    resultIcon.className = `result-icon ${isPneumonia ? 'pneumonia' : 'normal'}`;
    resultIcon.innerHTML = isPneumonia ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-check-circle"></i>';
    
    resultTitle.textContent = 'Analysis Complete';
    resultStatus.textContent = result.prediction;
    resultStatus.className = `result-status ${isPneumonia ? 'pneumonia' : 'normal'}`;
    
    // Update confidence meter
    const confidencePercent = Math.round(confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceFill.className = `confidence-fill ${isPneumonia ? 'pneumonia' : 'normal'}`;
    
    // Update recommendation
    if (isPneumonia) {
        recommendation.textContent = 'The AI model has detected signs consistent with pneumonia. Please consult with a healthcare professional immediately for proper medical evaluation and treatment.';
    } else {
        recommendation.textContent = 'The AI model indicates the chest X-ray appears normal with no obvious signs of pneumonia. However, always follow up with your healthcare provider for comprehensive medical assessment.';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
}

// Utility functions
function showError(message) {
    alert(`Error: ${message}`);
}

function resetAnalysis() {
    // Reset all sections
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    uploadArea.parentElement.style.display = 'flex';
    
    // Reset variables
    selectedFile = null;
    fileInput.value = '';
    
    // Remove animation classes
    document.querySelectorAll('.fade-in').forEach(el => {
        el.classList.remove('fade-in');
    });
}

function downloadReport() {
    if (!selectedFile) return;
    
    // Get current results
    const prediction = resultStatus.textContent;
    const confidence = confidenceValue.textContent;
    const recommendationText = recommendation.textContent;
    const timestamp = new Date().toLocaleString();
    
    // Create report content
    const reportContent = `
PNEUMONIA DETECTION REPORT
Generated: ${timestamp}
File: ${selectedFile.name}

ANALYSIS RESULTS:
Prediction: ${prediction}
Confidence: ${confidence}

RECOMMENDATION:
${recommendationText}

DISCLAIMER:
This report is generated by an AI model for educational purposes only. 
Always consult with qualified healthcare professionals for medical diagnosis and treatment.
    `.trim();
    
    // Create and download file
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pneumonia-report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Error handling for API connectivity
window.addEventListener('load', async function() {
    try {
        const response = await fetch(`${API_BASE_URL}/docs`, { method: 'HEAD' });
        if (!response.ok) {
            throw new Error('API not accessible');
        }
    } catch (error) {
        console.warn('API server may not be running. Please ensure the FastAPI server is started.');
        
        // Show a subtle warning in the UI
        const warningDiv = document.createElement('div');
        warningDiv.innerHTML = `
            <div style="background: #f8d7da; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center;">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Note:</strong> Please ensure the backend server is running at ${API_BASE_URL}
            </div>
        `;
        document.querySelector('.header').after(warningDiv.firstElementChild);
    }
});