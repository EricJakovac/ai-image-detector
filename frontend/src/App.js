import React, { useState, useCallback } from "react";
import DropzoneArea from "./components/DropzoneArea/DropzoneArea";
import ImagePreview from "./components/ImagePreview/ImagePreview";
import ComparisonGrid from "./components/ComparisonGrid/ComparisonGrid";
import ComparisonSummary from "./components/ComparisonSummary/ComparisonSummary";
import LoadingOverlay from "./components/LoadingOverlay/LoadingOverlay";
import { predictImages } from "./services/api";  // PROMJENA: predictImages umjesto batchPredict
import { getImagePreview } from "./utils/helpers";
import "./App.css";

function App() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [userLabels, setUserLabels] = useState({});
  const MAX_FILES = 3;

  const onDrop = useCallback(
    (acceptedFiles, rejectedFiles) => {
      setDragActive(false);

      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles.map((file) => {
          if (file.errors[0].code === "file-invalid-type") {
            return `${file.file.name}: Invalid format`;
          }
          if (file.errors[0].code === "file-too-large") {
            return `${file.file.name}: File too large (max 20MB)`;
          }
          return `${file.file.name}: ${file.errors[0].message}`;
        });
        alert(errors.join("\n"));
      }

      const newFiles = acceptedFiles.slice(0, MAX_FILES - images.length);
      setImages((prev) => [...prev, ...newFiles].slice(0, MAX_FILES));
      setResults([]);
    },
    [images.length]
  );

  const removeImage = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
    setResults([]);
    
    // Ukloni label kada se ukloni slika
    const newLabels = { ...userLabels };
    Object.keys(newLabels).forEach(key => {
      const keyNum = parseInt(key);
      if (keyNum === index) {
        delete newLabels[key];
      } else if (keyNum > index) {
        newLabels[keyNum - 1] = newLabels[key];
        delete newLabels[key];
      }
    });
    setUserLabels(newLabels);
  };

  const clearAll = () => {
    setImages([]);
    setResults([]);
    setUserLabels({});
  };

  const setImageLabel = (index, label) => {
    setUserLabels(prev => ({
      ...prev,
      [index]: label.toLowerCase()
    }));
  };

  const handlePredict = async () => {
    if (images.length === 0) return;

    setLoading(true);
    
    try {
      // PROMJENA: koristi predictImages za sve slike
      const predictions = await predictImages(images);
      
      // Formatiraj rezultate (isti kao prije)
      const formattedResults = predictions.map((result, index) => ({
        ...result,
        preview: getImagePreview(images[index]),
        userLabel: userLabels[index]
      }));
      
      setResults(formattedResults);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Processing failed: " + error.message);
    }
    
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Image Detector - Model Comparison</h1>
        <p className="subtitle">
          Upload up to {MAX_FILES} images to compare 6 AI models
        </p>
      </header>

      <main className="App-main">
        {/* Upload Section */}
        <section className="upload-section">
          <DropzoneArea
            onDrop={onDrop}
            images={images}
            MAX_FILES={MAX_FILES}
            dragActive={dragActive}
            setDragActive={setDragActive}
          />

          {images.length > 0 && (
            <ImagePreview
              images={images}
              removeImage={removeImage}
              clearAll={clearAll}
              handlePredict={handlePredict}
              loading={loading}
              MAX_FILES={MAX_FILES}
              userLabels={userLabels} 
              setImageLabel={setImageLabel} 
            />
          )}
        </section>

        {/* Results Section */}
        {results.length > 0 && (
          <section className="results-section">
            <h2>📊 Analysis Results</h2>
            <p className="results-count">
              {results.length} image{results.length !== 1 ? "s" : ""} analyzed
              (6 models each)
            </p>

            <div className="results-grid">
              {results.map((result, index) => (
                <div key={index} className="result-card">
                  {/* Original Image */}
                  <div className="original-image-container">
                    <h4 className="image-title">{result.filename || "Image"}</h4>
                    <img
                      src={result.preview}
                      alt="Original"
                      className="original-image"
                    />
                    <div className="image-meta">
                      <span>Type: {result.file_type || "Unknown"}</span>
                      <span>Size: {result.image_size || "Unknown"}</span>
                      <span>Your label: <strong>{result.userLabel?.toUpperCase() || "Not labeled"}</strong></span>
                    </div>
                  </div>

                  {/* Error State */}
                  {result.error && (
                    <div className="error-message">❌ Error: {result.error}</div>
                  )}

                  {/* Models Comparison */}
                  {!result.error && result.models && (
                    <>
                      <ComparisonGrid 
                        models={result.models} 
                        userLabel={result.userLabel} 
                      />
                      
                      <ComparisonSummary 
                        comparison={result.comparison} 
                        userLabel={result.userLabel} 
                        models={result.models}
                      />
                      
                      <div className="processing-info">
                        {result.processing_time && (
                          <div className="info-item">
                            <span className="info-label">Processing time:</span>
                            <span className="info-value">
                              {result.processing_time}s
                            </span>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Loading Overlay */}
        {loading && <LoadingOverlay imageCount={images.length} />}
      </main>

      <footer className="App-footer">
        <p><strong>AI Image Detector</strong> • Compare 6 AI models</p>
        <p className="footer-note">
          CNN • ViT • DeiT • ConvNeXt • All images processed at 224×224 resolution
        </p>
      </footer>
    </div>
  );
}

export default App;