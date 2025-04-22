/**
 * Moroccan Road Sign Detection System
 * Main JavaScript functionality
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // File input preview
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    const imagePreview = document.getElementById('image-preview');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    imagePreview.src = event.target.result;
                    preview.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Form submission handling with loading indicator
    const uploadForm = document.querySelector('form[action="/upload"]');
    const submitBtn = uploadForm ? uploadForm.querySelector('button[type="submit"]') : null;
    
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener('submit', function() {
            // Create and show loading spinner
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'text-center mt-3';
            loadingDiv.innerHTML = `
                <div class="loading-spinner"></div>
                <p>Processing image, please wait...</p>
            `;
            
            // Replace the submit button with the loading indicator
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            uploadForm.appendChild(loadingDiv);
        });
    }
    
    // Download report functionality (in results page)
    const downloadReportBtn = document.getElementById('downloadReport');
    
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', function() {
            // Show loading state
            const originalText = downloadReportBtn.textContent;
            downloadReportBtn.textContent = 'Generating PDF...';
            downloadReportBtn.disabled = true;
            
            setTimeout(() => {
                // Reset button state after PDF generation starts
                downloadReportBtn.textContent = originalText;
                downloadReportBtn.disabled = false;
            }, 3000);
        });
    }
    
    // Add image comparison slider if both images are present (in results page)
    const originalImg = document.querySelector('.result-image[alt="Original Image"]');
    const resultImg = document.querySelector('.result-image[alt="Detection Results"]');
    
    if (originalImg && resultImg && window.location.pathname.includes('results')) {
        // Create comparison container after the images section
        const comparisonSection = document.createElement('div');
        comparisonSection.className = 'card mb-4';
        comparisonSection.innerHTML = `
            <div class="card-header bg-warning text-white">
                <h5 class="mb-0">Interactive Comparison</h5>
            </div>
            <div class="card-body">
                <p class="text-center mb-3">Drag the slider to compare original and processed images</p>
                <div class="comparison-container" style="position: relative; max-width: 800px; margin: 0 auto;">
                    <div class="img-comp-img" style="position: absolute; height: 100%; overflow: hidden;">
                        <img src="${originalImg.src}" style="display: block; max-width: 100%;">
                    </div>
                    <div class="img-comp-img" style="position: relative; height: 100%;">
                        <img src="${resultImg.src}" style="display: block; max-width: 100%;">
                    </div>
                    <div class="img-comp-slider" style="position: absolute; z-index: 9; cursor: ew-resize; width: 40px; height: 40px; 
                                                     background: rgba(0, 0, 0, 0.7); border-radius: 50%; top: 50%; left: 50%; 
                                                     transform: translate(-50%, -50%); display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-size: 20px;">⟷</span>
                    </div>
                </div>
            </div>
        `;
        
        // Insert after the images row
        const imagesRow = document.querySelector('.row.mb-4');
        if (imagesRow) {
            imagesRow.parentNode.insertBefore(comparisonSection, imagesRow.nextSibling);
            
            // Initialize comparison functionality
            initComparisons();
        }
    }
    
    // API endpoint testing section for developers (optional, shown only in about page)
    if (window.location.pathname === '/about') {
        const techSection = document.querySelector('h3:contains("Technologies Used")').parentNode;
        if (techSection) {
            const apiTestingSection = document.createElement('div');
            apiTestingSection.className = 'mt-5';
            apiTestingSection.innerHTML = `
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        <h4 class="mb-0">API Testing</h4>
                    </div>
                    <div class="card-body">
                        <p>For developers: Test the road sign detection API endpoint directly.</p>
                        <pre class="bg-light p-3 rounded"><code>POST /api/detect
Content-Type: multipart/form-data

form-data: 
  - image: (your image file)</code></pre>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-secondary" id="showApiDocs">Show API Documentation</button>
                        </div>
                        <div class="mt-3 d-none" id="apiDocs">
                            <h5>API Response Format</h5>
                            <pre class="bg-light p-3 rounded"><code>{
  "status": "success",
  "signs_detected": 2,
  "signs": [
    {
      "id": 1,
      "type": "Stop",
      "confidence": "96.5%",
      "position": {
        "x": 120,
        "y": 50,
        "width": 80,
        "height": 80
      }
    },
    ...
  ]
}</code></pre>
                        </div>
                    </div>
                </div>
            `;
            
            // Add the API section to the page
            techSection.appendChild(apiTestingSection);
            
            // Add event listener for the API docs button
            document.getElementById('showApiDocs').addEventListener('click', function() {
                document.getElementById('apiDocs').classList.toggle('d-none');
                this.textContent = this.textContent === 'Show API Documentation' ? 
                                  'Hide API Documentation' : 'Show API Documentation';
            });
        }
    }
});

// Image comparison slider functionality
function initComparisons() {
    const compContainer = document.querySelector(".comparison-container");
    const slider = document.querySelector(".img-comp-slider");
    const imgComp = document.querySelectorAll(".img-comp-img");
    
    if (!compContainer || !slider || !imgComp.length) return;
    
    let w, h;
    
    // Get the width and height of the container
    function getDimensions() {
        w = compContainer.offsetWidth;
        h = compContainer.offsetHeight;
        
        // Set initial position of the slider to 50%
        slide(50);
    }
    
    // Call once to set initial dimensions
    getDimensions();
    
    // Update when window is resized
    window.addEventListener('resize', getDimensions);
    
    // Set the width of the first image to a percentage
    function slide(x) {
        // Convert percentage to pixels
        let pos = w * x / 100;
        
        // Ensure pos stays within bounds
        if (pos < 0) pos = 0;
        if (pos > w) pos = w;
        
        // Set the width of the first image
        imgComp[0].style.width = pos + "px";
        
        // Position the slider accordingly
        slider.style.left = pos + "px";
    }
    
    // Add mouse and touch event listeners
    slider.addEventListener("mousedown", startSliding);
    slider.addEventListener("touchstart", startSliding);
    
    function startSliding(e) {
        e.preventDefault();
        
        // Add event listeners for dragging
        document.addEventListener("mousemove", sliding);
        document.addEventListener("touchmove", sliding);
        document.addEventListener("mouseup", stopSliding);
        document.addEventListener("touchend", stopSliding);
    }
    
    function sliding(e) {
        e.preventDefault();
        
        // Get cursor position
        let pos;
        if (e.type === "touchmove") {
            const rect = compContainer.getBoundingClientRect();
            pos = e.touches[0].clientX - rect.left;
        } else {
            const rect = compContainer.getBoundingClientRect();
            pos = e.clientX - rect.left;
        }
        
        // Convert to percentage
        const percentage = (pos / w) * 100;
        slide(percentage);
    }
    
    function stopSliding() {
        document.removeEventListener("mousemove", sliding);
        document.removeEventListener("touchmove", sliding);
    }
}