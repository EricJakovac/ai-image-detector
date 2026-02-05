import React from "react";
import { getImagePreview, formatFileSize } from "../../utils/helpers";
import ImageLabel from "../ImageLabel/ImageLabel";
import "./ImagePreview.css";

const ImagePreview = ({ 
  images, 
  removeImage, 
  clearAll, 
  handlePredict, 
  loading, 
  MAX_FILES,
  userLabels,
  setImageLabel
}) => {
  return (
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
              <span className="filesize">{formatFileSize(file.size)}</span>
            </div>
            
            <ImageLabel
              index={index}
              label={userLabels[index]}
              setImageLabel={setImageLabel}
            />
            
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
        disabled={loading || images.length === 0 || Object.keys(userLabels).length !== images.length}
        className={`predict-btn ${loading ? "loading" : ""}`}
      >
        {loading ? (
          <>
            <span className="spinner"></span>
            Analyzing {images.length} image(s)...
          </>
        ) : (
          `🔄 Compare 6 AI Models`
        )}
      </button>
      
      {images.length > 0 && Object.keys(userLabels).length !== images.length && (
        <div className="label-warning">
          ⚠️ Please label all images as AI or Real before analyzing
        </div>
      )}
    </div>
  );
};

export default ImagePreview;