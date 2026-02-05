import React from "react";
import { formatPercentage, getConfidenceClass } from "../../utils/helpers";
import "./ModelCard.css";

const ModelCard = ({ modelData, modelKey, userLabel }) => {
  if (!modelData) {
    return (
      <div className="model-card error">
        <h5 className="model-title">
          <span className="model-icon">❌</span> {modelKey.replace("_", " ")}
        </h5>
        <div className="error-message">Model data not available</div>
      </div>
    );
  }

  const isFineTuned = modelKey.includes("fine_tuned");
  const modelName = modelData.name || modelKey.replace("_", " ").toUpperCase();
  const probability = modelData.probability || 0;
  const confidencePercent = modelData.confidence_percent || probability * 100;
  
  // Provjeri je li model točan u odnosu na korisnički label
  const isCorrect = userLabel && modelData?.label && 
    userLabel.toLowerCase() === modelData.label.toLowerCase();
  
  // Dodaj klasu za netočno ako postoji userLabel i model nije točan
  const isIncorrect = userLabel && modelData?.label && !isCorrect;

  return (
    <div className={`model-card ${isFineTuned ? "fine-tuned" : "raw"} ${isCorrect ? 'correct' : ''} ${isIncorrect ? 'incorrect' : ''}`}>
      <h5 className="model-title">
        <span className="model-icon">
          {modelKey.includes("cnn") ? "🖼️" : 
           modelKey.includes("vit") ? "🧠" : 
           modelKey.includes("deit") ? "🔬" : "🏗️"}
        </span>{" "}
        {modelName}
      </h5>
      
      {/* Prikaži accuracy badge samo ako postoji userLabel */}
      {userLabel && (
        <div className={`accuracy-badge ${isCorrect ? 'correct-badge' : 'incorrect-badge'}`}>
          {isCorrect ? '✅ Correct' : '❌ Wrong'}
        </div>
      )}
      
      <div className={`prediction-label ${modelData.label?.toLowerCase() || "unknown"}`}>
        {modelData.label || "Unknown"}
      </div>
      
      <div className={`confidence-badge ${getConfidenceClass(probability)}`}>
        {confidencePercent.toFixed(1)}%
      </div>
      
      <div className="probability-details">
        Real: {formatPercentage(modelData.raw_probabilities?.real || 0)}
        <br />
        AI: {formatPercentage(modelData.raw_probabilities?.ai || 0)}
      </div>
      
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
      
      {isFineTuned ? (
        <div className="fine-tuned-badge">🚀 Fine-tuned</div>
      ) : (
        <div className="raw-badge">🌐 ImageNet</div>
      )}
    </div>
  );
};

export default ModelCard;