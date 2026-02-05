import React from "react";
import "./ImageLabel.css";

const ImageLabel = ({ index, label, setImageLabel }) => {
  return (
    <div className="image-label">
      <span className="label-text">Is this image AI or Real?</span>
      <div className="label-buttons">
        <button
          className={`label-btn ${label === 'ai' ? 'active' : ''}`}
          onClick={() => setImageLabel(index, 'ai')}
          type="button"
        >
          🤖 AI
        </button>
        <button
          className={`label-btn ${label === 'real' ? 'active' : ''}`}
          onClick={() => setImageLabel(index, 'real')}
          type="button"
        >
          🖼️ Real
        </button>
      </div>
    </div>
  );
};

export default ImageLabel;