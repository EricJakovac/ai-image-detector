import React from "react";
import "./LoadingOverlay.css";

const LoadingOverlay = ({ imageCount }) => {
  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <div className="loading-spinner"></div>
        <h3>Analyzing Images...</h3>
        <p>Running 6 models per image (EfficientNet/ViT/DeiT/ConvNeXt)</p>
        <p className="loading-note">
          Processing {imageCount} image{imageCount !== 1 ? "s" : ""} • 
          This may take 30-90 seconds per image
        </p>
      </div>
    </div>
  );
};

export default LoadingOverlay;