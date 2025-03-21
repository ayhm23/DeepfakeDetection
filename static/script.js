document.getElementById('uploadForm').onsubmit = function(e) {
    e.preventDefault();
    
    // Show loading state
    document.getElementById('results').innerHTML = 'Processing...';
    
    const formData = new FormData(this);
    
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Server error');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = `
            <div class="alert alert-danger">
                Error: ${error.message || 'Failed to process video'}
            </div>
        `;
    });
};

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const facesDiv = document.getElementById('detectedFaces');
    
    if (data.success) {
        // Display overall result
        let overallHtml = `
            <div class="alert ${data.overall_prediction > 0.5 ? 'alert-danger' : 'alert-success'}">
                <h4>Overall Result: ${data.overall_prediction > 0.5 ? 'Likely Fake' : 'Likely Real'}</h4>
                <p>Overall Confidence: ${(data.overall_prediction * 100).toFixed(2)}%</p>
            </div>
        `;
        
        // Display individual face results
        let facesHtml = '<div class="faces-grid">';
        data.faces_data.forEach((face, index) => {
            facesHtml += `
                <div class="face-result">
                    <img src="data:image/jpeg;base64,${face.face}" class="detected-face">
                    <div class="face-prediction ${face.is_fake ? 'fake' : 'real'}">
                        <p>Face ${index + 1}</p>
                        <p>${face.is_fake ? 'FAKE' : 'REAL'}</p>
                        <p>${(face.prediction * 100).toFixed(2)}%</p>
                    </div>
                </div>
            `;
        });
        facesHtml += '</div>';
        
        resultsDiv.innerHTML = overallHtml;
        facesDiv.innerHTML = facesHtml;
    } else {
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                Error: ${data.message}
            </div>
        `;
        facesDiv.innerHTML = '';
    }
} 