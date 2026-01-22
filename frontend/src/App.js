import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import "./App.css";

function App() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const MAX_FILES = 3;

  const REACT_APP_API_URL =
    process.env.REACT_APP_API_URL || "http://localhost:8000";

  // Podržani formati slika
  const ALLOWED_TYPES = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/webp",
  ];
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB

  // Drag & Drop
  const onDrop = useCallback(
    (acceptedFiles, rejectedFiles) => {
      setDragActive(false);

      // Prikaži greške za odbijene fajlove
      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles.map((file) => {
          if (file.errors[0].code === "file-invalid-type") {
            return `${file.file.name}: Nepodržan format (podržani: JPEG, PNG, BMP, GIF, TIFF, WebP)`;
          }
          if (file.errors[0].code === "file-too-large") {
            return `${file.file.name}: Prevelik fajl (max 20MB)`;
          }
          return `${file.file.name}: ${file.errors[0].message}`;
        });
        alert(errors.join("\n"));
      }

      // Dodaj prihvaćene fajlove
      const newFiles = acceptedFiles.slice(0, MAX_FILES - images.length);
      setImages((prev) => [...prev, ...newFiles].slice(0, MAX_FILES));
      setResults([]);
    },
    [images.length],
  );

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "image/*": ALLOWED_TYPES.map((type) => type.replace("image/", "")),
    },
    maxSize: MAX_FILE_SIZE,
    maxFiles: MAX_FILES,
    multiple: true,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  // Ukloni sliku
  const removeImage = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
    setResults([]);
  };

  // Ukloni sve slike
  const clearAll = () => {
    setImages([]);
    setResults([]);
  };

  // Konvertiraj sliku u base64 za prikaz
  const getImagePreview = (file) => {
    return URL.createObjectURL(file);
  };

  // Predviđanje za jednu sliku
  const predictSingleImage = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `${REACT_APP_API_URL}/predict-dual`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 120000,
        },
      );
      return response.data;
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  };

  // Batch predviđanje
  const handlePredict = async () => {
    if (images.length === 0) return;

    setLoading(true);
    const predictions = [];

    try {
      for (let i = 0; i < images.length; i++) {
        const file = images[i];
        console.log(`Processing ${i + 1}/${images.length}: ${file.name}`);

        try {
          const result = await predictSingleImage(file);
          predictions.push({
            ...result,
            preview: getImagePreview(file),
          });
        } catch (error) {
          predictions.push({
            filename: file.name,
            error:
              error.response?.data?.detail ||
              error.message ||
              "Prediction failed",
            preview: getImagePreview(file),
          });
        }
      }

      setResults(predictions);
    } catch (error) {
      console.error("Batch error:", error);
      alert("Batch processing failed: " + error.message);
    }

    setLoading(false);
  };

  // Formatiraj postotak
  const formatPercentage = (value) => {
    return (value * 100).toFixed(1) + "%";
  };

  // Klasa za confidence
  const getConfidenceClass = (percentage) => {
    if (percentage >= 0.8) return "high-confidence";
    if (percentage >= 0.6) return "medium-confidence";
    return "low-confidence";
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>
          AI Image Detector - EfficientNet-B0 (CNN) vs Vision Transformer
          (ViT-Small/16) Comparison
        </h1>
        <p className="subtitle">
          Upload up to {MAX_FILES} images to compare predictions from both AI
          detection models
        </p>
      </header>

      <main className="App-main">
        {/* UPLOAD SECTION */}
        <section className="upload-section">
          <div
            {...getRootProps()}
            className={`dropzone ${dragActive ? "active" : ""} ${images.length >= MAX_FILES ? "disabled" : ""}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-content">
              <div className="upload-icon">📤</div>
              <p className="dropzone-title">
                {images.length >= MAX_FILES
                  ? "Max files reached"
                  : "Drag & drop images here"}
              </p>
              <p className="dropzone-subtitle">
                or click to browse (max {MAX_FILES} images, max 20MB each)
              </p>
              <p className="file-types">
                Supported: JPEG, PNG, BMP, GIF, TIFF, WebP
              </p>
            </div>
          </div>

          {/* SELECTED IMAGES PREVIEW */}
          {images.length > 0 && (
            <div className="selected-images">
              <div className="selected-header">
                <h3>
                  Selected Images ({images.length}/{MAX_FILES})
                </h3>
                <button onClick={clearAll} className="clear-btn">
                  Clear All
                </button>
              </div>

              <div className="image-previews">
                {images.map((file, index) => (
                  <div key={index} className="image-preview">
                    <img src={getImagePreview(file)} alt={file.name} />
                    <div className="image-info">
                      <span className="filename">{file.name}</span>
                      <span className="filesize">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                    <button
                      onClick={() => removeImage(index)}
                      className="remove-btn"
                      aria-label="Remove image"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>

              <button
                onClick={handlePredict}
                disabled={loading || images.length === 0}
                className={`predict-btn ${loading ? "loading" : ""}`}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing {images.length} image(s)...
                  </>
                ) : (
                  `🔍 Analyze with CNN & ViT Models`
                )}
              </button>
            </div>
          )}
        </section>

        {/* RESULTS SECTION */}
        {results.length > 0 && (
          <section className="results-section">
            <h2>📊 Analysis Results</h2>
            <p className="results-count">
              {results.length} image{results.length !== 1 ? "s" : ""} analyzed
            </p>

            <div className="results-grid">
              {results.map((result, index) => (
                <div key={index} className="result-card">
                  {/* ORIGINAL IMAGE */}
                  <div className="image-container">
                    <h4 className="image-title">
                      {result.filename || result.error}
                    </h4>
                    <img
                      src={result.preview}
                      alt="Original"
                      className="original-image"
                    />

                    {/* ERROR STATE */}
                    {result.error && (
                      <div className="error-message">
                        ❌ Error: {result.error}
                      </div>
                    )}
                  </div>

                  {/* RESULTS IF NO ERROR */}
                  {!result.error && (
                    <>
                      {/* MODEL COMPARISON */}
                      <div className="model-comparison">
                        {/* CNN Model */}
                        <div className="model-result cnn-model">
                          <h5 className="model-title">
                            <span className="model-icon">🖼️</span> CNN
                            (EfficientNet)
                          </h5>
                          {/* Labela je sada dinamička i točna */}
                          <div
                            className={`prediction ${result.cnn.label.toLowerCase()}`}
                          >
                            {result.cnn.label}
                          </div>
                          {/* Confidence se odnosi na ispisanu labelu */}
                          <div
                            className={`confidence ${getConfidenceClass(result.cnn.probability)}`}
                          >
                            Confidence:{" "}
                            {formatPercentage(result.cnn.probability)}
                          </div>
                          {/* Detalji sada vuku točne ključeve iz raw_probabilities */}
                          <div className="probability-details">
                            Real:{" "}
                            {formatPercentage(
                              result.cnn.raw_probabilities?.real,
                            )}{" "}
                            | AI:{" "}
                            {formatPercentage(result.cnn.raw_probabilities?.ai)}
                          </div>
                        </div>

                        {/* ViT Model */}
                        <div className="model-result vit-model">
                          <h5 className="model-title">
                            <span className="model-icon">🧠</span> ViT (Vision
                            Transformer)
                          </h5>
                          <div
                            className={`prediction ${result.vit.label.toLowerCase()}`}
                          >
                            {result.vit.label}
                          </div>
                          <div
                            className={`confidence ${getConfidenceClass(result.vit.probability)}`}
                          >
                            Confidence:{" "}
                            {formatPercentage(result.vit.probability)}
                          </div>
                          <div className="probability-details">
                            Real:{" "}
                            {formatPercentage(
                              result.vit.raw_probabilities?.real,
                            )}{" "}
                            | AI:{" "}
                            {formatPercentage(result.vit.raw_probabilities?.ai)}
                          </div>
                        </div>
                      </div>

                      {/* AGREEMENT INDICATOR */}
                      {result.comparison && (
                        <div
                          className={`agreement-indicator ${result.comparison.models_agree ? "agree" : "disagree"}`}
                        >
                          <div className="agreement-icon">
                            {result.comparison.models_agree ? "✅" : "⚠️"}
                          </div>
                          <div className="agreement-text">
                            <strong>{result.comparison.agreement}</strong>
                            {result.comparison.confidence_difference && (
                              <span>
                                {" "}
                                (Difference:{" "}
                                {result.comparison.confidence_difference}%)
                              </span>
                            )}
                          </div>
                        </div>
                      )}

                      {/* VISUALIZATIONS */}
                      <div className="visualizations">
                        <div className="visualization">
                          <h5 className="viz-title">CNN Grad-CAM Heatmap</h5>
                          <p className="viz-description">
                            Areas that influenced CNN's decision
                          </p>
                          {result.cnn.visualization ? (
                            <img
                              src={`data:image/png;base64,${result.cnn.visualization}`}
                              alt="CNN Grad-CAM"
                              className="heatmap-image"
                            />
                          ) : (
                            <div className="viz-placeholder">
                              No visualization available
                            </div>
                          )}
                        </div>

                        <div className="visualization">
                          <h5 className="viz-title">ViT Attention Map</h5>
                          <p className="viz-description">
                            Areas where ViT focused attention
                          </p>
                          {result.vit.visualization ? (
                            <img
                              src={`data:image/png;base64,${result.vit.visualization}`}
                              alt="ViT Attention Map"
                              className="heatmap-image"
                            />
                          ) : (
                            <div className="viz-placeholder">
                              No visualization available
                            </div>
                          )}
                        </div>
                      </div>

                      {/* SUMMARY STATS */}
                      <div className="summary-stats">
                        <div className="stat">
                          <span className="stat-label">File Type:</span>
                          <span className="stat-value">
                            {result.file_type || "Unknown"}
                          </span>
                        </div>
                        <div className="stat">
                          <span className="stat-label">Original Size:</span>
                          <span className="stat-value">
                            {result.image_size || "Unknown"}
                          </span>
                        </div>
                        <div className="stat">
                          <span className="stat-label">Processed Size:</span>
                          <span className="stat-value">
                            {result.processed_size || "224x224"}
                          </span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* LOADING OVERLAY */}
        {loading && (
          <div className="loading-overlay">
            <div className="loading-content">
              <div className="loading-spinner"></div>
              <h3>Analyzing Images...</h3>
              <p>Running CNN and ViT models with visualizations</p>
              <p className="loading-note">
                This may take 10-30 seconds per image
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>
          <strong>AI Image Detector</strong> • Compares CNN (EfficientNet) vs
          ViT (Vision Transformer) predictions
        </p>
        <p className="footer-note">
          Models analyze images at 224x224 resolution • Heatmaps show model
          attention areas
        </p>
      </footer>
    </div>
  );
}

export default App;
