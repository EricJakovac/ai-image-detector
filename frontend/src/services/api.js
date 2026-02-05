import axios from "axios";

const REACT_APP_API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// JEDINSTVENA OPTIMIZIRANA FUNKCIJA
export const predictImages = async (files) => {
  const filesArray = Array.isArray(files) ? files : [files];
  
  const formData = new FormData();
  filesArray.forEach((file) => {
    formData.append("files", file);
  });

  try {
    // Dinamički timeout ovisno o broju slika
    const timeout = filesArray.length === 1 ? 45000 :  // 45s za 1 sliku
                    filesArray.length === 2 ? 90000 :  // 90s za 2 slike
                    120000;                            // 120s za 3 slike
    
    const response = await axios.post(
      `${REACT_APP_API_URL}/predict-comparison`,
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: timeout,
      }
    );
    
    // Handle response format
    if (filesArray.length === 1) {
      return [response.data];
    } else {
      return response.data.results || [];
    }
    
  } catch (error) {
    console.error("Prediction error:", error);
    
    // Vrati error za svaku sliku
    return filesArray.map(file => ({
      filename: file.name,
      error: error.response?.data?.detail || error.message || "Prediction failed",
    }));
  }
};

// Helper funkcije
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