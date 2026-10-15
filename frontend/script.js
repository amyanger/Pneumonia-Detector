// Global variables
let selectedFile = null;
let lastResult = null;
const API_BASE_URL = 'http://localhost:8000';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // Matches the backend limit
const VALID_TYPES = ['image/jpeg', 'image/png'];

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
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
const pneumoniaProbability = document.getElementById('pneumoniaProbability');
const recommendation = document.getElementById('recommendation');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadBtn = document.getElementById('downloadBtn');
const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');
const errorDismiss = document.getElementById('errorDismiss');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // File input events
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    // One handler for the whole area (including the Choose File button).
    // The input lives inside uploadArea, so ignore its own bubbled click.
    uploadArea.addEventListener('click', (event) => {
        if (event.target === fileInput) return;
        fileInput.click();
    });

    // Button events
    analyzeBtn.addEventListener('click', analyzeImage);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    downloadBtn.addEventListener('click', downloadReport);
    errorDismiss.addEventListener('click', hideError);
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectFile(file);
    }
    // Allow choosing the same file again after an error
    fileInput.value = '';
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
        selectFile(files[0]);
    }
}

function selectFile(file) {
    if (!VALID_TYPES.includes(file.type)) {
        showError('Please select a valid image file (JPG or PNG)');
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        showError('File is too large. The maximum size is 10 MB.');
        return;
    }
    hideError();
    selectedFile = file;
    displayImagePreview(file);
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;

        // Hide upload area and show preview
        uploadArea.parentElement.style.display = 'none';
        previewSection.style.display = 'flex';
        previewSection.classList.add('fade-in');
    };
    reader.onerror = function() {
        showError('Could not read the selected file');
    };
    reader.readAsDataURL(file);
}

// Analysis functions
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    if (analyzeBtn.disabled) return;

    hideError();
    analyzeBtn.disabled = true;
    showLoadingState();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE_URL}/predict/`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(await getErrorDetail(response));
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Analysis error:', error);
        // fetch rejects with a TypeError when the server is unreachable
        const message = error instanceof TypeError
            ? `Could not reach the server at ${API_BASE_URL}`
            : error.message;
        showError(`Analysis failed: ${message}`);
        hideLoadingState();
        // Return to the preview so the user can retry
        previewSection.style.display = 'flex';
    } finally {
        analyzeBtn.disabled = false;
    }
}

// Reads FastAPI's {"detail": ...} error body. detail is usually a string,
// but validation errors (422) return a list of objects with a msg field.
async function getErrorDetail(response) {
    try {
        const data = await response.json();
        if (typeof data.detail === 'string') {
            return data.detail;
        }
        if (Array.isArray(data.detail)) {
            const messages = data.detail.map(item => item && item.msg).filter(Boolean);
            if (messages.length > 0) {
                return messages.join('; ');
            }
        }
    } catch (e) {
        // Body was not JSON, fall through to the generic message
    }
    return `Server returned status ${response.status}`;
}

function showLoadingState() {
    previewSection.style.display = 'none';
    loadingSection.style.display = 'flex';
    loadingSection.classList.add('fade-in');
}

function hideLoadingState() {
    loadingSection.style.display = 'none';
}

function displayResults(result) {
    hideLoadingState();
    lastResult = result;

    const isPneumonia = result.prediction === "Pneumonia Detected";
    // confidence is the probability of the predicted label
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

    if (typeof result.pneumonia_probability === 'number') {
        pneumoniaProbability.textContent = `Pneumonia probability: ${Math.round(result.pneumonia_probability * 100)}%`;
    } else {
        pneumoniaProbability.textContent = '';
    }

    // Update recommendation
    if (isPneumonia) {
        recommendation.textContent = 'The AI model has detected signs consistent with pneumonia. Please consult with a healthcare professional immediately for proper medical evaluation and treatment.';
    } else {
        recommendation.textContent = 'The AI model indicates the chest X-ray appears normal with no obvious signs of pneumonia. However, always follow up with your healthcare provider for comprehensive medical assessment.';
    }

    // Show results section
    resultsSection.style.display = 'flex';
    resultsSection.classList.add('fade-in');
}

// Utility functions
function showError(message) {
    errorMessage.textContent = message;
    errorBanner.style.display = 'flex';
}

function hideError() {
    errorBanner.style.display = 'none';
    errorMessage.textContent = '';
}

function resetAnalysis() {
    // Reset all sections
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    uploadArea.parentElement.style.display = 'flex';
    hideError();

    // Reset variables
    selectedFile = null;
    lastResult = null;
    fileInput.value = '';
    previewImage.removeAttribute('src');
    analyzeBtn.disabled = false;

    // Remove animation classes
    document.querySelectorAll('.fade-in').forEach(el => {
        el.classList.remove('fade-in');
    });
}

function downloadReport() {
    if (!selectedFile || !lastResult) return;

    // Get current results
    const prediction = resultStatus.textContent;
    const confidence = confidenceValue.textContent;
    const probabilityLine = pneumoniaProbability.textContent;
    const recommendationText = recommendation.textContent;
    const timestamp = new Date().toLocaleString();

    // Create report content
    const reportContent = `
PNEUMONIA DETECTION REPORT
Generated: ${timestamp}
File: ${selectedFile.name}

ANALYSIS RESULTS:
Prediction: ${prediction}
Confidence: ${confidence}${probabilityLine ? `\n${probabilityLine}` : ''}

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
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('API not accessible');
        }
    } catch (error) {
        console.warn('API server may not be running. Please ensure the FastAPI server is started.');

        // Show a subtle warning in the UI
        const warningDiv = document.createElement('div');
        warningDiv.className = 'notice-banner';
        warningDiv.setAttribute('role', 'status');

        const icon = document.createElement('i');
        icon.className = 'fas fa-exclamation-triangle';
        const label = document.createElement('strong');
        label.textContent = 'Note:';
        const text = document.createElement('span');
        text.textContent = `Please ensure the backend server is running at ${API_BASE_URL}`;

        warningDiv.append(icon, label, text);
        document.querySelector('.header').after(warningDiv);
    }
});
