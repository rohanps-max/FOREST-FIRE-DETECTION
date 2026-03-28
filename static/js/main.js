document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const logContainer = document.getElementById('log-container');
    const statusBox = document.getElementById('system-status-box');
    const statusText = document.getElementById('system-status-text');
    const logCount = document.getElementById('log-count');
    
    const slider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const captureBtn = document.getElementById('capture-btn');
    const gallery = document.getElementById('snapshot-gallery');
    const galleryGrid = document.getElementById('gallery-grid');
    
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadResult = document.getElementById('upload-result');
    const uploadPreview = document.getElementById('upload-img-preview');
    const uploadStatusText = document.getElementById('upload-status-text');

    // --- Tab Switching ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    // --- Confidence Slider Handler ---
    slider.addEventListener('input', (e) => {
        confVal.textContent = e.target.value;
    });

    slider.addEventListener('change', async (e) => {
        const val = e.target.value / 100.0;
        try {
            await fetch('/set_confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conf: val })
            });
            console.log("Confidence updated to", val);
        } catch(err) {
            console.error("Failed to update confidence", err);
        }
    });

    // --- Capture Picture Handler ---
    captureBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/capture', { method: 'POST' });
            if(res.ok) {
                const data = await res.json();
                if(data.status === 'success') {
                    gallery.classList.remove('hidden');
                    const img = document.createElement('img');
                    img.src = "data:image/jpeg;base64," + data.image;
                    galleryGrid.prepend(img); // Add to beginning
                }
            }
        } catch (err) {
            console.error("Failed to capture image", err);
        }
    });

    // --- Upload File Handler ---
    uploadZone.addEventListener('click', () => fileInput.click());
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    uploadZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files), false);
    
    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        if(files.length === 0) return;
        uploadFile(files[0]);
    }
    
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        uploadZone.innerHTML = '<p>ANALYZING DATA... PLEASE WAIT</p>';
        
        try {
            const res = await fetch('/upload_image', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if(data.status === 'success') {
                uploadZone.innerHTML = '<p>Drag & Drop Fire Suspect Image Here or Click to Browse</p>';
                uploadResult.classList.remove('hidden');
                uploadPreview.src = "data:image/jpeg;base64," + data.image;
                
                if(data.detected) {
                    uploadStatusText.innerHTML = '<span style="color:var(--accent-red)">🚨 HAZARD DETECTED IN MEDIA</span>';
                } else {
                    uploadStatusText.innerHTML = '<span style="color:var(--safe-color)">✅ NO HAZARD FOUND</span>';
                }
            }
        } catch (err) {
            console.error(err);
            uploadZone.innerHTML = '<p>ANALYSIS FAILED. TRY AGAIN.</p>';
        }
    }

    // --- Polling Logs ---
    async function fetchLogs() {
        try {
            const response = await fetch('/get_logs');
            if (response.ok) {
                const logs = await response.json();
                updateLogsUI(logs);
            }
        } catch (error) {
            console.error("Failed to fetch logs:", error);
        }
    }

    function updateLogsUI(logs) {
        logContainer.innerHTML = '';
        if (logs.length === 0) {
            logContainer.innerHTML = '<p class="loading-text" style="color: var(--text-secondary); text-align: center; margin-top: 2rem;">NO HISTORICAL ALERTS.</p>';
            statusBox.className = 'system-status-box';
            statusText.textContent = 'SYSTEM SECURE';
            logCount.textContent = '0';
            return;
        }

        logCount.textContent = logs.length;
        const latestLog = logs[0];
        const logTime = new Date(latestLog.Timestamp).getTime();
        const now = new Date().getTime();
        
        // Alert stays active for 5 mins
        if ((now - logTime) < 300000) {
            statusBox.className = 'system-status-box alert';
            statusText.textContent = 'CRITICAL ALERT';
        } else {
            statusBox.className = 'system-status-box';
            statusText.textContent = 'SYSTEM SECURE';
        }

        logs.forEach((log) => {
            const item = document.createElement('div');
            item.className = 'log-item';
            
            const timeStr = log.Timestamp; // Already formatted as YYYY-MM-DD HH:MM:SS
            const conf = (parseFloat(log.Confidence) * 100).toFixed(1);
            
            item.innerHTML = `
                <div class="time">[${timeStr}] ALERT OCCURRENCE</div>
                <div class="details">
                    <span class="class-name">X_CLASS: ${log.Class.toUpperCase()}</span>
                    <span class="conf">SENS: ${conf}%</span>
                </div>
            `;
            logContainer.appendChild(item);
        });
    }

    setInterval(fetchLogs, 2000);
    fetchLogs();
});
