import React from "react";
import ModelCard from "../ModelCard/ModelCard";
import "./ComparisonGrid.css";

const ComparisonGrid = ({ models, userLabel }) => {
  if (!models) return null;

  const modelOrder = [
    "cnn_fine_tuned",
    "convnext_fine_tuned",
    "vit_fine_tuned",
    "deit_fine_tuned",
    "cnn_raw",
    "vit_raw",
  ];

  return (
    <div className="comparison-grid">
      {modelOrder.map((modelKey) => (
        <ModelCard
          key={modelKey}
          modelData={models[modelKey]}
          modelKey={modelKey}
          userLabel={userLabel}
        />
      ))}
    </div>
  );
};

export default ComparisonGrid;