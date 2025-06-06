const axios = require('axios');
const fs = require('fs');
const path = require('path');

console.log('=== DEBUG INFO ===');
console.log('Axios available:', typeof axios);
console.log('FS available:', typeof fs);

let selectedFile = null;
let upscaledImageData = null;

// API base URL (your Python backend)
const API_BASE = 'http://127.0.0.1:8000';

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded successfully');
    
    // Get DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const upscaleBtn = document.getElementById('upscaleBtn');
    const clearBtn = document.getElementById('clearBtn');
    const previewImg = document.getElementById('previewImg');
    const imagePreview = document.getElementById('imagePreview');
    const emptyState = document.getElementById('emptyState');
    const imageInfo = document.getElementById('imageInfo');
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultSection = document.getElementById('resultSection');
    const originalImg = document.getElementById('originalImg');
    const upscaledImg = document.getElementById('upscaledImg');
    const downloadBtn = document.getElementById('downloadBtn');
    
    console.log('Drop zone found:', !!dropZone);
    console.log('File input found:', !!fileInput);
    
    if (!dropZone || !fileInput) {
        console.error('Required elements not found!');
        return;
    }
    
    // Setup Event Listeners
    setupEventListeners();
    
    // Check backend connection
    checkBackend();
    
    function setupEventListeners() {
        console.log('Setting up event listeners...');
        
        // File drop zone events
        dropZone.addEventListener('click', () => {
            console.log('Drop zone clicked!');
            fileInput.click();
        });
        
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
        
        // File input change
        fileInput.addEventListener('change', handleFileSelect);
        
        // Button events
        upscaleBtn.addEventListener('click', handleUpscale);
        clearBtn.addEventListener('click', handleClear);
        downloadBtn.addEventListener('click', handleDownload);
        
        console.log('Event listeners set up successfully');
    }

    function handleDragOver(e) {
        e.preventDefault();
        console.log('Drag over');
        dropZone.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        console.log('Drag leave');
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        console.log('File dropped');
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        console.log('Files dropped:', files.length);
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        console.log('File selected from input');
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        console.log('Processing file:', file.name, file.type, file.size);
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file');
            return;
        }
        
        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File too large. Please select an image under 10MB');
            return;
        }
        
        selectedFile = file;
        displayImagePreview(file);
        enableButtons();
    }

    function displayImagePreview(file) {
        console.log('Displaying preview for:', file.name);
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            imagePreview.classList.remove('hidden');
            emptyState.classList.add('hidden');
            
            // Show image info
            const size = (file.size / 1024 / 1024).toFixed(2);
            imageInfo.textContent = `${file.name} (${size} MB)`;
            
            // Hide previous results
            resultSection.classList.add('hidden');
            
            console.log('Preview displayed successfully');
        };
        reader.readAsDataURL(file);
    }

    function enableButtons() {
        upscaleBtn.disabled = false;
        clearBtn.disabled = false;
        console.log('Buttons enabled');
    }

    function disableButtons() {
        upscaleBtn.disabled = true;
        clearBtn.disabled = true;
        console.log('Buttons disabled');
    }

    async function handleUpscale() {
        if (!selectedFile) return;
        
        console.log('Starting upscale process...');
        
        try {
            disableButtons();
            showProgress();
            
            // Create FormData
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            updateProgress(10, 'Uploading image...');
            
            // Send to backend
            const response = await axios.post(`${API_BASE}/upscale`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                responseType: 'blob',
                onUploadProgress: (progressEvent) => {
                    const uploadProgress = Math.round((progressEvent.loaded * 50) / progressEvent.total);
                    updateProgress(10 + uploadProgress, 'Uploading image...');
                }
            });
            
            updateProgress(70, 'Processing with AI...');
            
            // Simulate processing time for better UX
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            updateProgress(100, 'Complete!');
            
            // Handle the response
            upscaledImageData = response.data;
            displayResults();
            
            console.log('Upscale completed successfully');
            
        } catch (error) {
            console.error('Upscaling failed:', error);
            alert('Upscaling failed. Make sure the Python backend is running.');
        } finally {
            hideProgress();
            enableButtons();
        }
    }

    function showProgress() {
        progressContainer.style.display = 'block';
        updateProgress(0, 'Starting...');
    }

    function hideProgress() {
        progressContainer.style.display = 'none';
    }

    function updateProgress(percent, text) {
        progressFill.style.width = percent + '%';
        progressText.textContent = text;
    }

    function displayResults() {
        // Show original image
        originalImg.src = previewImg.src;
        
        // Show upscaled image
        const upscaledUrl = URL.createObjectURL(upscaledImageData);
        upscaledImg.src = upscaledUrl;
        
        // Show results section
        resultSection.classList.remove('hidden');
        
        console.log('Results displayed');
    }

    async function handleDownload() {
        if (!upscaledImageData) return;
        
        try {
            // Create download link
            const url = URL.createObjectURL(upscaledImageData);
            const a = document.createElement('a');
            a.href = url;
            a.download = `upscaled_${selectedFile.name}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            // Show success message
            const originalText = downloadBtn.textContent;
            downloadBtn.textContent = '✅ Downloaded!';
            setTimeout(() => {
                downloadBtn.textContent = originalText;
            }, 2000);
            
            console.log('Download completed');
            
        } catch (error) {
            console.error('Download failed:', error);
            alert('Download failed');
        }
    }

    function handleClear() {
        console.log('Clearing selection');
        selectedFile = null;
        upscaledImageData = null;
        fileInput.value = '';
        
        // Reset UI
        imagePreview.classList.add('hidden');
        emptyState.classList.remove('hidden');
        resultSection.classList.add('hidden');
        
        disableButtons();
    }

    // Check if backend is running
    async function checkBackend() {
        try {
            const response = await axios.get(API_BASE);
            console.log('Backend connected:', response.data.message);
        } catch (error) {
            console.warn('Backend not running. Please start your Python backend.');
        }
    }
});