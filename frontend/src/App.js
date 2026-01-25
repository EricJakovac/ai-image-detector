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

  // Predviđanje za usporedbu RAW vs Fine-tuned
  const predictComparison = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `${REACT_APP_API_URL}/predict-comparison`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 180000, // 3 minute (RAW modeli su sporiji)
        },
      );
      return response.data;
    } catch (error) {
      console.error("Comparison prediction error:", error);
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
          const result = await predictComparison(file);

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
    if (typeof value === "number") {
      return (value * 100).toFixed(1) + "%";
    }
    return "0.0%";
  };

  // Klasa za confidence
  const getConfidenceClass = (percentage) => {
    if (percentage >= 0.8) return "high-confidence";
    if (percentage >= 0.6) return "medium-confidence";
    return "low-confidence";
  };

  // Render model kartice za comparison mode
  const renderModelCard = (modelData, modelKey) => {
    if (!modelData) {
      return (
        <div className={`model-card error`}>
          <h5 className="model-title">
            <span className="model-icon">❌</span> {modelKey.replace("_", " ")}
          </h5>
          <div className="error-message">Model data not available</div>
        </div>
      );
    }

    const isFineTuned = modelKey.includes("fine_tuned");
    const modelName =
      modelData.name || modelKey.replace("_", " ").toUpperCase();

    return (
      <div className={`model-card ${isFineTuned ? "fine-tuned" : "raw"}`}>
        <h5 className="model-title">
          <span className="model-icon">
            {modelKey.includes("cnn") ? "🖼️" : "🧠"}
          </span>{" "}
          {modelName}
        </h5>
        <div
          className={`prediction-label ${modelData.label?.toLowerCase() || "unknown"}`}
        >
          {modelData.label || "Unknown"}
        </div>
        <div
          className={`confidence-badge ${getConfidenceClass(modelData.probability || 0)}`}
        >
          {modelData.confidence_percent
            ? `${modelData.confidence_percent.toFixed(1)}%`
            : modelData.probability
              ? formatPercentage(modelData.probability)
              : "0.0%"}
        </div>
        <div className="probability-details">
          Real:{" "}
          {modelData.raw_probabilities?.real
            ? formatPercentage(modelData.raw_probabilities.real)
            : "0.0%"}
          <br />
          AI:{" "}
          {modelData.raw_probabilities?.ai
            ? formatPercentage(modelData.raw_probabilities.ai)
            : "0.0%"}
        </div>
        {/* VIZUALIZACIJA - samo ako postoji */}
        <div className="model-visualization">
          {modelData.visualization ? (
            <img
              src={`data:image/png;base64,${modelData.visualization}`}
              alt={`${modelName} Visualization`}
              className="model-heatmap"
            />
          ) : (
            <div className="no-visualization">No visualization available</div>
          )}
        </div>
        {/* BADGE za fine-tuned modele */}
        {isFineTuned && <div className="fine-tuned-badge">🚀 Fine-tuned</div>}
          {/* BADGE za raw modele */}
        {!isFineTuned && <div className="raw-badge">🌐 ImageNet</div>}
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Image Detector - Model Comparison</h1>
        <p className="subtitle">
          Upload up to {MAX_FILES} images to compare RAW vs Fine-tuned AI models
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
                  `🔄 Compare RAW vs Fine-tuned Models`
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
              (4 models each)
            </p>

            <div className="results-grid">
              {results.map((result, index) => (
                <div key={index} className="result-card">
                  {/* ORIGINAL IMAGE */}
                  <div className="original-image-container">
                    <h4 className="image-title">
                      {result.filename || "Image"}
                    </h4>
                    <img
                      src={result.preview}
                      alt="Original"
                      className="original-image"
                    />
                    <div className="image-meta">
                      <span>Type: {result.file_type || "Unknown"}</span>
                      <span>Size: {result.image_size || "Unknown"}</span>
                    </div>
                  </div>

                  {/* ERROR STATE */}
                  {result.error && (
                    <div className="error-message">
                      ❌ Error: {result.error}
                    </div>
                  )}

                  {/* 4-MODEL GRID */}
                  {!result.error && result.models && (
                    <>
                      <div className="comparison-grid">
                        {renderModelCard(
                          result.models.cnn_fine_tuned,
                          "cnn_fine_tuned",
                        )}
                        {renderModelCard(result.models.cnn_raw, "cnn_raw")}
                        {renderModelCard(
                          result.models.vit_fine_tuned,
                          "vit_fine_tuned",
                        )}
                        {renderModelCard(result.models.vit_raw, "vit_raw")}
                      </div>

                      {/* COMPARISON SUMMARY */}
                      {result.comparison && (
                        <div className="comparison-summary">
                          <div className="summary-card">
                            <h5 className="summary-title">
                              📈 Fine-tuning Improvement
                            </h5>
                            <div className="improvement-stats">
                              <div className="improvement-stat">
                                <span className="stat-label">CNN:</span>
                                <span className="stat-value improvement-positive">
                                  +
                                  {Math.abs(
                                    result.comparison.cnn_improvement || 0,
                                  ).toFixed(1)}
                                  %
                                </span>
                              </div>
                              <div className="improvement-stat">
                                <span className="stat-label">ViT:</span>
                                <span className="stat-value improvement-positive">
                                  +
                                  {Math.abs(
                                    result.comparison.vit_improvement || 0,
                                  ).toFixed(1)}
                                  %
                                </span>
                              </div>
                            </div>
                            <div className="summary-text">
                              Fine-tuning improves CNN by{" "}
                              {Math.abs(
                                result.comparison.cnn_improvement || 0,
                              ).toFixed(1)}
                              % and ViT by{" "}
                              {Math.abs(
                                result.comparison.vit_improvement || 0,
                              ).toFixed(1)}
                              %
                            </div>

                            <div className="prediction-agreement">
                              <div className="agreement-item">
                                <span className="agreement-label">
                                  CNN agreement:
                                </span>
                                <span
                                  className={`agreement-value ${result.comparison.cnn_same_prediction ? "agree" : "disagree"}`}
                                >
                                  {result.comparison.cnn_same_prediction
                                    ? "✅ Yes"
                                    : "❌ No"}
                                </span>
                              </div>
                              <div className="agreement-item">
                                <span className="agreement-label">
                                  ViT agreement:
                                </span>
                                <span
                                  className={`agreement-value ${result.comparison.vit_same_prediction ? "agree" : "disagree"}`}
                                >
                                  {result.comparison.vit_same_prediction
                                    ? "✅ Yes"
                                    : "❌ No"}
                                </span>
                              </div>
                            </div>

                            <div className="best-model">
                              <span className="best-model-label">
                                Best performing model:
                              </span>
                              <span className="best-model-value">
                                {result.comparison.best_model
                                  ?.replace(/_/g, " ")
                                  .toUpperCase() || "Unknown"}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* PROCESSING INFO */}
                      <div className="processing-info">
                        <div className="info-item">
                          <span className="info-label">Processed at:</span>
                          <span className="info-value">224×224 resolution</span>
                        </div>
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

        {/* LOADING OVERLAY */}
        {loading && (
          <div className="loading-overlay">
            <div className="loading-content">
              <div className="loading-spinner"></div>
              <h3>Analyzing Images...</h3>
              <p>Running 4 models per image (CNN/ViT × RAW/Fine-tuned)</p>
              <p className="loading-note">
                This may take 30-60 seconds per image
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>
          <strong>AI Image Detector</strong> • Compare RAW vs Fine-tuned models
        </p>
        <p className="footer-note">
          RAW models: ImageNet pretrained • Fine-tuned: optimized for AI
          detection • All images processed at 224×224 resolution
        </p>
      </footer>
    </div>
  );
}

export default App;
