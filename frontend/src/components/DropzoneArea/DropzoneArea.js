import React from "react";
import { useDropzone } from "react-dropzone";
import "./DropzoneArea.css";

const DropzoneArea = ({ 
  onDrop, 
  images, 
  MAX_FILES, 
  dragActive, 
  setDragActive 
}) => {
  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "image/*": [
        "jpeg", "jpg", "png", "bmp", "gif", "tiff", "webp"
      ].map((type) => type),
    },
    maxSize: 20 * 1024 * 1024,
    maxFiles: MAX_FILES,
    multiple: true,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  return (
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
  );
};

export default DropzoneArea;