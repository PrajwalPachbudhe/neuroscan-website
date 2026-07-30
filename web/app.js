/**
 * NeuroScan AI — Brain Tumor MRI Classifier
 * ==========================================
 * Client-side brain tumor classification using TensorFlow.js.
 * Runs entirely in the browser — no server required.
 */

(() => {
    'use strict';

    // ─── Configuration ──────────────────────────────────────────
    const MODEL_URL = 'model/model.json';
    const IMG_SIZE = 224;
    const CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary'];
    const CLASS_DISPLAY_NAMES = {
        glioma: 'Glioma',
        meningioma: 'Meningioma',
        notumor: 'No Tumor',
        pituitary: 'Pituitary Tumor',
    };

    // Tumor information database
    const TUMOR_INFO = {
        glioma: {
            description: 'Gliomas are tumors that arise from glial cells in the brain. They are the most common type of primary brain tumor.',
            severity: 'High',
            location: 'Can occur anywhere in the brain or spinal cord',
            treatment: 'Surgery, radiation therapy, chemotherapy',
            icon: '🔴',
        },
        meningioma: {
            description: 'Meningiomas develop from the meninges, the membranes surrounding the brain and spinal cord. Most are benign.',
            severity: 'Low to Moderate',
            location: 'Surface of the brain (meninges)',
            treatment: 'Observation, surgery, radiation therapy',
            icon: '🟠',
        },
        pituitary: {
            description: 'Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas.',
            severity: 'Low to Moderate',
            location: 'Pituitary gland (base of brain)',
            treatment: 'Medication, surgery, radiation therapy',
            icon: '🟣',
        },
        notumor: {
            description: 'No tumor was detected in this MRI scan. The brain appears normal based on the AI analysis.',
            severity: 'None',
            location: 'N/A',
            treatment: 'No treatment needed',
            icon: '🟢',
        },
    };

    // ─── State ──────────────────────────────────────────────────
    let model = null;
    let isModelLoaded = false;
    let isDemoMode = false;
    let uploadedImage = null;

    // ─── DOM Elements ───────────────────────────────────────────
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const previewImage = document.getElementById('preview-image');
    const previewFilename = document.getElementById('preview-filename');
    const scanLine = document.getElementById('scan-line');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const resetBtn = document.getElementById('reset-btn');
    const emptyState = document.getElementById('empty-state');
    const results = document.getElementById('results');
    const diagnosis = document.getElementById('diagnosis');
    const diagnosisName = document.getElementById('diagnosis-name');
    const diagnosisConfidence = document.getElementById('diagnosis-confidence');
    const infoSection = document.getElementById('info-section');
    const infoGrid = document.getElementById('info-grid');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    // ─── Model Loading ──────────────────────────────────────────
    async function loadModel() {
        updateStatus('loading', 'Loading AI model...');

        try {
            model = await tf.loadLayersModel(MODEL_URL);
            isModelLoaded = true;
            isDemoMode = false;
            updateStatus('ready', 'Model loaded — Ready to analyze');
            console.log('✅ TensorFlow.js model loaded successfully');

            // Enable analyze button if image is uploaded
            if (uploadedImage) {
                analyzeBtn.disabled = false;
            }
        } catch (error) {
            console.warn('⚠️ Could not load model, switching to demo mode:', error.message);
            isDemoMode = true;
            isModelLoaded = true; // Mark as "loaded" so demo works
            updateStatus('demo', 'Demo mode — No trained model found');

            if (uploadedImage) {
                analyzeBtn.disabled = false;
            }
        }
    }

    function updateStatus(state, message) {
        statusDot.className = 'status-bar__dot ' + state;
        statusText.textContent = message;
    }

    // ─── File Upload Handling ───────────────────────────────────
    function initUploadHandlers() {
        // Click to upload
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // Analyze button
        analyzeBtn.addEventListener('click', () => analyzeImage());

        // Reset button
        resetBtn.addEventListener('click', () => resetState());
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG, WEBP)');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be under 10MB');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage = e.target.result;
            showPreview(file.name);
        };
        reader.readAsDataURL(file);
    }

    function showPreview(filename) {
        previewImage.src = uploadedImage;
        previewFilename.textContent = filename;
        uploadZone.style.display = 'none';
        preview.classList.add('active');

        // Enable analyze button if model is loaded
        analyzeBtn.disabled = !isModelLoaded;

        // Hide previous results
        results.classList.remove('active');
        emptyState.style.display = 'block';
        infoSection.style.display = 'none';
    }

    function resetState() {
        uploadedImage = null;
        fileInput.value = '';
        uploadZone.style.display = '';
        preview.classList.remove('active');
        results.classList.remove('active');
        emptyState.style.display = 'block';
        infoSection.style.display = 'none';
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = '🔍 Analyze Scan';

        // Reset probability bars
        CLASS_NAMES.forEach((cls) => {
            const bar = document.getElementById('bar-' + cls);
            const val = document.getElementById('prob-' + cls);
            if (bar) bar.style.width = '0%';
            if (val) val.textContent = '0%';
        });

        // Reset diagnosis
        diagnosis.className = 'diagnosis';
        diagnosisName.textContent = '—';
        diagnosisConfidence.textContent = '—';
    }

    // ─── Image Analysis ─────────────────────────────────────────
    async function analyzeImage() {
        if (!uploadedImage || !isModelLoaded) return;

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtnText.innerHTML = '<div class="spinner"></div> Analyzing...';
        scanLine.classList.add('active');

        // Small delay for UX (show the scan animation)
        await sleep(800);

        let predictions;
        if (isDemoMode) {
            predictions = generateDemoPredictions();
        } else {
            predictions = await runInference();
        }

        // Hide scan line
        scanLine.classList.remove('active');

        // Display results
        displayResults(predictions);

        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtnText.textContent = '🔍 Re-analyze';
    }

    async function runInference() {
        return tf.tidy(() => {
            // Create an image element for TF processing
            const img = new Image();
            img.src = uploadedImage;

            // Convert image to tensor
            let tensor = tf.browser.fromPixels(previewImage)
                .resizeNearestNeighbor([IMG_SIZE, IMG_SIZE])
                .toFloat()
                .div(255.0)
                .expandDims(0);

            // Run prediction
            const output = model.predict(tensor);
            const probabilities = output.dataSync();

            return Array.from(probabilities);
        });
    }

    function generateDemoPredictions() {
        // Generate realistic-looking demo predictions
        const dominant = Math.floor(Math.random() * CLASS_NAMES.length);
        const probs = CLASS_NAMES.map((_, i) => {
            if (i === dominant) return 0.7 + Math.random() * 0.25; // 70-95%
            return Math.random() * 0.15; // 0-15%
        });

        // Normalize to sum to 1
        const sum = probs.reduce((a, b) => a + b, 0);
        return probs.map((p) => p / sum);
    }

    // ─── Results Display ────────────────────────────────────────
    function displayResults(probabilities) {
        // Hide empty state, show results
        emptyState.style.display = 'none';
        results.classList.add('active');

        // Find the top prediction
        let maxIdx = 0;
        let maxProb = 0;
        probabilities.forEach((prob, i) => {
            if (prob > maxProb) {
                maxProb = prob;
                maxIdx = i;
            }
        });

        const topClass = CLASS_NAMES[maxIdx];
        const topName = CLASS_DISPLAY_NAMES[topClass];
        const topConfidence = (maxProb * 100).toFixed(1);

        // Update diagnosis badge
        diagnosis.className = 'diagnosis diagnosis--' + topClass;
        diagnosisName.textContent = topName;
        diagnosisConfidence.textContent = topConfidence + '%';

        // Animate probability bars (with stagger)
        CLASS_NAMES.forEach((cls, i) => {
            const percentage = (probabilities[i] * 100).toFixed(1);
            const bar = document.getElementById('bar-' + cls);
            const val = document.getElementById('prob-' + cls);

            // Reset first
            bar.style.width = '0%';
            val.textContent = '0%';

            // Animate after staggered delay
            setTimeout(() => {
                bar.style.width = percentage + '%';
                val.textContent = percentage + '%';
            }, 150 + i * 120);
        });

        // Show tumor info
        showTumorInfo(topClass);
    }

    function showTumorInfo(tumorClass) {
        const info = TUMOR_INFO[tumorClass];
        if (!info) return;

        infoSection.style.display = 'block';
        infoGrid.innerHTML = `
            <div class="info-card">
                <div class="info-card__icon">📋</div>
                <div class="info-card__label">Description</div>
                <div class="info-card__value">${info.description}</div>
            </div>
            <div class="info-card">
                <div class="info-card__icon">⚠️</div>
                <div class="info-card__label">Severity Level</div>
                <div class="info-card__value">${info.severity}</div>
            </div>
            <div class="info-card">
                <div class="info-card__icon">📍</div>
                <div class="info-card__label">Common Location</div>
                <div class="info-card__value">${info.location}</div>
            </div>
            <div class="info-card">
                <div class="info-card__icon">💊</div>
                <div class="info-card__label">Typical Treatment</div>
                <div class="info-card__value">${info.treatment}</div>
            </div>
        `;
    }

    // ─── Utilities ──────────────────────────────────────────────
    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    // ─── Initialize ─────────────────────────────────────────────
    function init() {
        initUploadHandlers();
        loadModel();
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
