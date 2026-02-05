export const formatPercentage = (value) => {
  if (typeof value === "number") {
    return (value * 100).toFixed(1) + "%";
  }
  return "0.0%";
};

export const getConfidenceClass = (percentage) => {
  if (percentage >= 0.8) return "high-confidence";
  if (percentage >= 0.6) return "medium-confidence";
  return "low-confidence";
};

export const getImagePreview = (file) => {
  return URL.createObjectURL(file);
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

