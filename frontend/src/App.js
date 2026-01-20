import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setImages(Array.from(e.target.files));
    setResults([]);  // Reset results
  };

  const REACT_APP_API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handlePredict = async () => {
    if (images.length === 0) return;

    setLoading(true);
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('images', image);
    });

    try {
      const response = await axios.post(
        `${REACT_APP_API_URL}/api/v1/batch-predict-cam`,  // 🎯 Grad-CAM endpoint
        formData,
        { 
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000  // 30s za Grad-CAM
        }
      );
      setResults(response.data.results);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error: ' + (error.response?.data?.detail || error.message));
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
          className="file-input"
        />
        <button 
          onClick={handlePredict} 
          disabled={loading || images.length === 0}
          className="predict-btn"
        >
          {loading ? '🎯 Generating Heatmaps...' : `🔍 Predict ${images.length} Images`}
        </button>
      </div>

      {results.length > 0 && (
        <div className="results">
          <h2>📊 Results + Grad-CAM Heatmaps:</h2>
          <div className="results-grid">
            {results.map((result, index) => (
              <div key={index} className={`result-card ${result.label.toLowerCase()}`}>
                <div className="result-header">
                  <strong>{result.filename}</strong>
                  <span className={`confidence ${result.label.toLowerCase()}`}>
                    {result.label} ({(result.ai_probability * 100).toFixed(1)}%)
                  </span>
                </div>
                
                {/* 🎯 GRAD-CAM HEATMAP */}
                <div className="gradcam-container">
                  <div className="heatmap-title">What model "saw":</div>
                  <img 
                    src={`data:image/png;base64,${result.gradcam_b64}`} 
                    alt="Grad-CAM Heatmap" 
                    className="gradcam-image"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
