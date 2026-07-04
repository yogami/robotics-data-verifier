document.getElementById('reportInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            renderDashboard(data);
        } catch (err) {
            alert("Invalid JSON file. Please provide a valid quality_report.json");
        }
    };
    reader.readAsText(file);
});

function renderDashboard(data) {
    // Hide upload section with a slight fade out
    const uploadSection = document.getElementById('uploadSection');
    uploadSection.style.opacity = '0';
    setTimeout(() => {
        uploadSection.style.display = 'none';
        
        // Show dashboard grid and trigger animation
        const grid = document.getElementById('dashboardGrid');
        grid.classList.remove('hidden');
        // A tiny delay to allow display:block to apply before adding visibility class
        setTimeout(() => {
            grid.classList.add('visible');
        }, 50);
    }, 300);

    // Parse and populate KPIs
    const metrics = data.metrics;
    
    // Animate numbers
    animateValue('kpiEntropy', 0, metrics.kinematic_entropy_score, 1000, 4);
    
    const anomalyPercent = metrics.isolation_forest_anomaly_rate * 100;
    animateValue('kpiAnomaly', 0, anomalyPercent, 1000, 2, '%');
    
    animateValue('kpiJitter', 0, metrics.std_timestep_sec, 1000, 4);

    // Dynamic Slashing Logic Visualization
    const actionEl = document.getElementById('kpiAction');
    const actionCard = document.getElementById('actionCard');
    
    // Clear previous classes
    actionCard.classList.remove('slash', 'verify');
    
    let slashAmount = 0;
    if (metrics.kinematic_entropy_score < 1.0) slashAmount += 50;
    if (metrics.isolation_forest_anomaly_rate > 0.02) slashAmount += 50;
    
    if (slashAmount > 0) {
        actionCard.classList.add('slash');
        actionEl.innerText = `SLASH ${slashAmount}%`;
    } else {
        actionCard.classList.add('verify');
        actionEl.innerText = 'VERIFIED';
    }

    // Details Panel (Certificate)
    document.getElementById('valDatasetId').innerText = data.dataset || data.dataset_id;
    
    // Format timestamp nicely
    const date = new Date(data.audit_timestamp);
    document.getElementById('valTimestamp').innerText = date.toLocaleString('en-US', { timeZone: 'UTC' }) + ' UTC';
    
    document.getElementById('valFrames').innerText = data.frames_analyzed.toLocaleString();
    document.getElementById('valHash').innerText = data.solana_attestation.report_hash;
    document.getElementById('valStatus').innerText = data.solana_attestation.status;

    // Flags Panel
    const flagsList = document.getElementById('flagsList');
    flagsList.innerHTML = '';
    
    if (data.flags && data.flags.length > 0) {
        data.flags.forEach((flag) => {
            const li = document.createElement('li');
            li.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> 
                            <span>${flag}</span>`;
            flagsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = 'success';
        li.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        <span>No anomalies detected. Data meets all Verifier Node quality thresholds.</span>`;
        flagsList.appendChild(li);
    }
}

// Utility for animating numbers
function animateValue(id, start, end, duration, decimals = 0, suffix = '') {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        // Easing out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = start + easeProgress * (end - start);
        
        obj.innerHTML = current.toFixed(decimals) + suffix;
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
