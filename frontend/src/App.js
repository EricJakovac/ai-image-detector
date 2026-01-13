import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const API_URL = 'http://127.0.0.1:8000/api/v1';  // Backend URL

  const handleFileChange = (e) => {
    setImages(Array.from(e.target.files));
    setResults([]);  // Reset results
  };

  const handlePredict = async () => {
    if (images.length === 0) return;

    setLoading(true);
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('images', image);
    });

    try {
      const response = await axios.post(
        `${API_URL}/batch-predict-cam`,  // ✅ Grad-CAM endpoint
        formData,
        { 
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000  // 30s timeout za Grad-CAM
        }
      );
      setResults(response.data.results);
      console.log('✅ Prediction success:', response.data);
    } catch (error) {
      console.error('❌ Prediction error:', error.response?.data || error.message);
      alert(`Error: ${error.response?.data?.detail || error.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>🛡️ AI Image Detector + Grad-CAM</h1>
      
      <div className="upload-section">
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileChange}
        />
        <button 
          onClick={handlePredict} 
          disabled={loading || images.length === 0}
          className="predict-btn"
        >
          {loading ? '🎯 Analyzing...' : `🔍 Predict ${images.length} images`}
        </button>
      </div>

      {results.length > 0 && (
        <div className="results">
          <h2>📊 Results ({results.length} images):</h2>
          {results.map((result, i) => (
            <div key={i} className={`result ${result.label.toLowerCase()}`}>
              <div className="result-header">
                <strong>{result.image_filename}</strong>
                <span className="confidence">
                  {(result.ai_probability * 100).toFixed(1)}% {result.label}
                </span>
              </div>
              {result.gradcam_image_b64 && (
                <img 
                  src={`data:image/png;base64,${result.gradcam_image_b64}`} 
                  alt="Grad-CAM Heatmap" 
                  className="gradcam-img"
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
