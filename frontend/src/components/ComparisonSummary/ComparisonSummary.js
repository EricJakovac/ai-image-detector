import React from "react";
import "./ComparisonSummary.css";

const ComparisonSummary = ({ comparison, userLabel, models }) => {
  if (!comparison) return null;

  // Funkcija za pronalaženje najboljeg modela - mora biti TOČAN i imati najveći confidence
  const findBestModels = () => {
    if (!models || !userLabel) return [];
    
    const correctModels = [];
    
    // Prvo pronađi sve točne modele
    Object.entries(models).forEach(([key, model]) => {
      if (model && model.label) {
        const isCorrect = userLabel.toLowerCase() === model.label.toLowerCase();
        if (isCorrect) {
          correctModels.push({
            key,
            name: key.replace(/_/g, " ").toUpperCase(),
            confidence: model.probability || 0,
            label: model.label
          });
        }
      }
    });
    
    // Ako postoje točni modeli, sortiraj ih po confidenceu
    if (correctModels.length > 0) {
      correctModels.sort((a, b) => b.confidence - a.confidence);
      
      // Pronađi najveći confidence
      const maxConfidence = correctModels[0].confidence;
      
      // Vrati sve modele s maksimalnim confidenceom
      return correctModels.filter(model => model.confidence === maxConfidence);
    }
    
    // Ako nema točnih modela, vrati model s najvećim confidenceom (čak i ako je netočan)
    let bestModels = [];
    let maxConfidence = 0;
    
    Object.entries(models).forEach(([key, model]) => {
      if (model && model.probability) {
        if (model.probability > maxConfidence) {
          maxConfidence = model.probability;
          bestModels = [{
            key,
            name: key.replace(/_/g, " ").toUpperCase(),
            confidence: model.probability,
            label: model.label
          }];
        } else if (model.probability === maxConfidence) {
          bestModels.push({
            key,
            name: key.replace(/_/g, " ").toUpperCase(),
            confidence: model.probability,
            label: model.label
          });
        }
      }
    });
    
    return bestModels;
  };

  const bestModels = findBestModels();
  
  // Ako postoji userLabel, izračunaj točnost
  let cnnCorrect = null;
  let vitCorrect = null;
  let deitCorrect = null;
  let convnextCorrect = null;
  
  if (userLabel && models) {
    cnnCorrect = userLabel.toLowerCase() === models.cnn_fine_tuned?.label?.toLowerCase();
    vitCorrect = userLabel.toLowerCase() === models.vit_fine_tuned?.label?.toLowerCase();
    deitCorrect = userLabel.toLowerCase() === models.deit_fine_tuned?.label?.toLowerCase();
    convnextCorrect = userLabel.toLowerCase() === models.convnext_fine_tuned?.label?.toLowerCase();
  }

  // Broj točnih modela
  const correctCount = [cnnCorrect, vitCorrect, deitCorrect, convnextCorrect]
    .filter(Boolean).length;
  const totalModels = 4; // Fine-tuned modeli

  return (
    <div className="comparison-summary">
      <div className="summary-card">
        <h5 className="summary-title">📊 Fine-tuned Models Performance Analysis</h5>
        
        {userLabel ? (
          <>
            {/* STATISTIKA TOČNOSTI */}
            <div className="accuracy-stats">
              <div className="accuracy-meter">
                <div className="accuracy-label">Accuracy:</div>
                <div className="accuracy-value">
                  {(correctCount / totalModels * 100).toFixed(0)}%
                  <span className="accuracy-fraction"> ({correctCount}/{totalModels} correct)</span>
                </div>
              </div>
              <div className="accuracy-visual">
                <div className="accuracy-bar">
                  <div 
                    className="accuracy-fill"
                    style={{ width: `${(correctCount / totalModels * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* MODEL BY MODEL TOČNOST */}
            <div className="model-accuracy-grid">
              <div className={`model-accuracy-item ${cnnCorrect ? 'correct' : 'incorrect'}`}>
                <span className="model-accuracy-name">CNN</span>
                <span className="model-accuracy-status">
                  {cnnCorrect ? '✅' : '❌'}
                </span>
              </div>
              <div className={`model-accuracy-item ${vitCorrect ? 'correct' : 'incorrect'}`}>
                <span className="model-accuracy-name">ViT</span>
                <span className="model-accuracy-status">
                  {vitCorrect ? '✅' : '❌'}
                </span>
              </div>
              <div className={`model-accuracy-item ${deitCorrect ? 'correct' : 'incorrect'}`}>
                <span className="model-accuracy-name">DeiT</span>
                <span className="model-accuracy-status">
                  {deitCorrect ? '✅' : '❌'}
                </span>
              </div>
              <div className={`model-accuracy-item ${convnextCorrect ? 'correct' : 'incorrect'}`}>
                <span className="model-accuracy-name">ConvNeXt</span>
                <span className="model-accuracy-status">
                  {convnextCorrect ? '✅' : '❌'}
                </span>
              </div>
            </div>

            {/* NAJBOLJI MODEL(I) */}
            <div className="best-models-section">
              <h6 className="best-models-title">🏆 Best Model(s):</h6>
              {bestModels.length > 0 ? (
                <div className="best-models-list">
                  {bestModels.map((model, index) => (
                    <div key={index} className="best-model-item">
                      <span className="best-model-name">{model.name}</span>
                      <span className="best-model-confidence">
                        {(model.confidence * 100).toFixed(1)}%
                        {userLabel.toLowerCase() === model.label?.toLowerCase() ? 
                          ' ✅ Correct' : ' ❌ Wrong'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-best-model">No best model identified</div>
              )}
            </div>

            {/* SUMMARIZED FEEDBACK */}
            <div className={`performance-feedback ${correctCount === totalModels ? 'perfect' : correctCount > 0 ? 'partial' : 'poor'}`}>
              {correctCount === totalModels ? (
                '🎉 Perfect! All fine-tuned models  agree with your assessment.'
              ) : correctCount >= totalModels / 2 ? (
                '👍 Good! Most fine-tuned models agree with your assessment.'
              ) : correctCount > 0 ? (
                '⚠️ Some fine-tuned models disagree with your assessment.'
              ) : (
                '❌ All fine-tuned models disagree with your assessment.'
              )}
            </div>
          </>
        ) : (
          /* BEZ USER LABELA - stari prikaz */
          <>
            <div className="improvement-stats">
              <div className="improvement-stat">
                <span className="stat-label">CNN:</span>
                <span className="stat-value improvement-positive">
                  +{Math.abs(comparison.cnn_improvement || 0).toFixed(1)}%
                </span>
              </div>
              <div className="improvement-stat">
                <span className="stat-label">ViT:</span>
                <span className="stat-value improvement-positive">
                  +{Math.abs(comparison.vit_improvement || 0).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="summary-text">
              Fine-tuning improves models by {Math.abs(comparison.cnn_improvement || 0).toFixed(1)}% on average
            </div>

            <div className="prediction-agreement">
              <div className="agreement-item">
                <span className="agreement-label">Best model:</span>
                <span className="agreement-value best-model-name">
                  {models && models.cnn_fine_tuned ? 'EFFICIENTNET FINE TUNED' : 'Unknown'}
                </span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ComparisonSummary;