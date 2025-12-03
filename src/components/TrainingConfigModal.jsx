import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000/api/training';

const TrainingConfigModal = ({ 
  selectedCandle, 
  onClose, 
  isDarkMode = false 
}) => {
  // Training configuration state
  const [trainYears, setTrainYears] = useState(3.0);
  const [testWindowSize, setTestWindowSize] = useState(35);
  
  // Model parameters (defaults from 3.py)
  const [learningRate, setLearningRate] = useState(0.05);
  const [numLeaves, setNumLeaves] = useState(31);
  const [maxDepth, setMaxDepth] = useState(6);
  const [minChildSamples, setMinChildSamples] = useState(20);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [trainingError, setTrainingError] = useState(null);
  const [runId, setRunId] = useState(null);
  const [results, setResults] = useState(null);
  
  // Format candle date for display
  const formatCandleDate = (candle) => {
    if (!candle) return 'N/A';
    const date = new Date(candle.timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  // Calculate train period dates
  const getTrainPeriod = () => {
    if (!selectedCandle) return { start: 'N/A', end: 'N/A' };
    
    const endDate = new Date(selectedCandle.timestamp);
    const startDate = new Date(endDate);
    startDate.setFullYear(startDate.getFullYear() - trainYears);
    
    return {
      start: startDate.toLocaleDateString(),
      end: endDate.toLocaleDateString()
    };
  };
  
  const trainPeriod = getTrainPeriod();
  
  // Start training
  const handleStartTraining = async () => {
    setIsTraining(true);
    setTrainingError(null);
    setTrainingComplete(false);
    
    try {
      const response = await fetch(`${API_BASE_URL}/run-poc`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candle_date: selectedCandle.timestamp,
          train_years: trainYears,
          test_window_size: testWindowSize,
          params: {
            learning_rate: learningRate,
            num_leaves: numLeaves,
            max_depth: maxDepth,
            min_child_samples: minChildSamples
          }
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start training');
      }
      
      const data = await response.json();
      setRunId(data.run_id);
      
      // Start polling for completion
      pollForCompletion(data.run_id);
      
    } catch (error) {
      setTrainingError(error.message);
      setIsTraining(false);
    }
  };
  
  // Poll for training completion
  const pollForCompletion = async (runId) => {
    const maxAttempts = 120; // 10 minutes with 5-second intervals
    let attempts = 0;
    
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/poc-runs/${runId}`);
        if (!response.ok) {
          // Run might not exist yet
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000);
          } else {
            setTrainingError('Training timeout - check PowerShell window');
            setIsTraining(false);
          }
          return;
        }
        
        const data = await response.json();
        
        if (data.status === 'completed') {
          setResults(data);
          setTrainingComplete(true);
          setIsTraining(false);
        } else if (data.status === 'failed') {
          setTrainingError('Training failed - check PowerShell window');
          setIsTraining(false);
        } else {
          // Still running
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 5000);
          } else {
            setTrainingError('Training timeout - check PowerShell window');
            setIsTraining(false);
          }
        }
      } catch (error) {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000);
        } else {
          setTrainingError('Polling error: ' + error.message);
          setIsTraining(false);
        }
      }
    };
    
    // Start polling
    setTimeout(poll, 5000);
  };
  
  // Render metrics
  const renderMetrics = (metrics, label) => {
    if (!metrics) return null;
    
    return (
      <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
        <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
          {label}
        </h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Accuracy:</span>
            <span className="ml-2 font-medium">{(metrics.accuracy * 100).toFixed(2)}%</span>
          </div>
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>F1 Score:</span>
            <span className="ml-2 font-medium">{metrics.f1?.toFixed(4) || 'N/A'}</span>
          </div>
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>AUC:</span>
            <span className="ml-2 font-medium">
              {metrics.auc ? metrics.auc.toFixed(4) : 'N/A'}
            </span>
          </div>
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>MCC:</span>
            <span className="ml-2 font-medium">{metrics.mcc?.toFixed(4) || 'N/A'}</span>
          </div>
          {label === 'Train Metrics' && metrics.best_threshold && (
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Threshold:</span>
              <span className="ml-2 font-medium">{metrics.best_threshold.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>
    );
  };
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className={`w-full max-w-2xl rounded-lg shadow-2xl ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        {/* Header */}
        <div className={`px-6 py-4 border-b flex items-center justify-between ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <h3 className={`text-xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            🎯 Train Custom Model
          </h3>
          <button
            onClick={onClose}
            className={`text-2xl ${isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-600 hover:text-gray-800'}`}
          >
            ×
          </button>
        </div>
        
        {/* Content */}
        <div className="px-6 py-4 max-h-[70vh] overflow-y-auto">
          {!trainingComplete ? (
            <>
              {/* Selected Candle Info */}
              <div className={`mb-4 p-3 rounded ${
                isDarkMode ? 'bg-blue-900/20 border border-blue-800' : 'bg-blue-50 border border-blue-200'
              }`}>
                <div className={`text-sm ${isDarkMode ? 'text-blue-300' : 'text-blue-700'}`}>
                  <strong>Selected Candle:</strong> {formatCandleDate(selectedCandle)}
                </div>
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                  This candle marks the END of the training period. Test window starts immediately after.
                </div>
              </div>
              
              {/* Train Period Configuration */}
              <div className="mb-4">
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Training Period (years before selected candle)
                </label>
                <input
                  type="number"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={trainYears}
                  onChange={(e) => setTrainYears(parseFloat(e.target.value))}
                  disabled={isTraining}
                  className={`w-full px-3 py-2 rounded ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white' 
                      : 'bg-white border-gray-300 text-gray-900'
                  } border`}
                />
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Train on data from {trainPeriod.start} to {trainPeriod.end}
                </div>
              </div>
              
              {/* Test Window Size */}
              <div className="mb-4">
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Test Window Size (candles after selected candle)
                </label>
                <input
                  type="number"
                  min="10"
                  max="100"
                  value={testWindowSize}
                  onChange={(e) => setTestWindowSize(parseInt(e.target.value))}
                  disabled={isTraining}
                  className={`w-full px-3 py-2 rounded ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white' 
                      : 'bg-white border-gray-300 text-gray-900'
                  } border`}
                />
              </div>
              
              {/* Model Parameters */}
              <div className="mb-4">
                <h4 className={`text-sm font-semibold mb-3 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  LightGBM Parameters (4-Param Model)
                </h4>
                
                <div className="space-y-3">
                  {/* Learning Rate */}
                  <div>
                    <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Learning Rate: {learningRate.toFixed(3)}
                    </label>
                    <input
                      type="range"
                      min="0.01"
                      max="0.2"
                      step="0.001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                      disabled={isTraining}
                      className="w-full"
                    />
                    <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      Range: 0.01 - 0.2
                    </div>
                  </div>
                  
                  {/* Num Leaves */}
                  <div>
                    <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Num Leaves: {numLeaves}
                    </label>
                    <input
                      type="range"
                      min="15"
                      max="63"
                      step="1"
                      value={numLeaves}
                      onChange={(e) => setNumLeaves(parseInt(e.target.value))}
                      disabled={isTraining}
                      className="w-full"
                    />
                    <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      Range: 15 - 63
                    </div>
                  </div>
                  
                  {/* Max Depth */}
                  <div>
                    <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Max Depth: {maxDepth}
                    </label>
                    <input
                      type="range"
                      min="3"
                      max="12"
                      step="1"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                      disabled={isTraining}
                      className="w-full"
                    />
                    <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      Range: 3 - 12
                    </div>
                  </div>
                  
                  {/* Min Child Samples */}
                  <div>
                    <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Min Child Samples: {minChildSamples}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="50"
                      step="1"
                      value={minChildSamples}
                      onChange={(e) => setMinChildSamples(parseInt(e.target.value))}
                      disabled={isTraining}
                      className="w-full"
                    />
                    <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      Range: 10 - 50
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Training Status */}
              {isTraining && (
                <div className={`mb-4 p-3 rounded flex items-center ${
                  isDarkMode ? 'bg-blue-900/20 border border-blue-800' : 'bg-blue-50 border border-blue-200'
                }`}>
                  <div className={`animate-spin rounded-full h-5 w-5 border-b-2 mr-3 ${
                    isDarkMode ? 'border-blue-400' : 'border-blue-600'
                  }`}></div>
                  <div>
                    <div className={`text-sm font-medium ${isDarkMode ? 'text-blue-300' : 'text-blue-700'}`}>
                      Training in progress...
                    </div>
                    <div className={`text-xs ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                      Check PowerShell window for progress. This may take a few minutes.
                    </div>
                  </div>
                </div>
              )}
              
              {/* Error Display */}
              {trainingError && (
                <div className={`mb-4 p-3 rounded ${
                  isDarkMode ? 'bg-red-900/20 border border-red-800 text-red-300' : 'bg-red-50 border border-red-200 text-red-700'
                }`}>
                  <div className="font-medium">❌ Error</div>
                  <div className="text-sm mt-1">{trainingError}</div>
                </div>
              )}
            </>
          ) : (
            /* Training Complete - Show Results */
            <div className="space-y-4">
              <div className={`p-4 rounded border ${
                isDarkMode ? 'bg-green-900/20 border-green-800' : 'bg-green-50 border-green-200'
              }`}>
                <div className={`text-lg font-semibold mb-2 ${
                  isDarkMode ? 'text-green-300' : 'text-green-700'
                }`}>
                  ✅ Training Complete!
                </div>
                <div className={`text-sm ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                  Run ID: {runId}
                </div>
              </div>
              
              {/* Training Configuration Summary */}
              <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  Configuration
                </h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Selected Candle:</span>
                    <span className="ml-2 font-medium">{formatCandleDate(selectedCandle)}</span>
                  </div>
                  <div>
                    <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Train Years:</span>
                    <span className="ml-2 font-medium">{trainYears}</span>
                  </div>
                  <div>
                    <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Test Window:</span>
                    <span className="ml-2 font-medium">{testWindowSize} candles</span>
                  </div>
                  <div>
                    <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Learning Rate:</span>
                    <span className="ml-2 font-medium">{learningRate.toFixed(3)}</span>
                  </div>
                </div>
              </div>
              
              {/* Data Splits */}
              {results?.config && (
                <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <h4 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                    Data Splits
                  </h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Train Size:</span>
                      <span className="ml-2 font-medium">{results.config.train_size || 'N/A'} candles</span>
                    </div>
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Test Size:</span>
                      <span className="ml-2 font-medium">{results.config.test_size || 'N/A'} candles</span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Metrics */}
              <div className="space-y-3">
                {renderMetrics(results?.train_metrics, 'Train Metrics')}
                {renderMetrics(results?.test_metrics, 'Test Metrics (Unseen)')}
              </div>
            </div>
          )}
        </div>
        
        {/* Footer Actions */}
        <div className={`px-6 py-4 border-t flex justify-end gap-3 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          {!trainingComplete ? (
            <>
              <button
                onClick={onClose}
                disabled={isTraining}
                className={`px-4 py-2 rounded ${
                  isDarkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Cancel
              </button>
              <button
                onClick={handleStartTraining}
                disabled={isTraining || !selectedCandle}
                className={`px-4 py-2 rounded ${
                  isDarkMode 
                    ? 'bg-blue-600 hover:bg-blue-500 text-white' 
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                } ${(isTraining || !selectedCandle) ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {isTraining ? 'Training...' : 'Start Training'}
              </button>
            </>
          ) : (
            <>
              <button
                onClick={onClose}
                className={`px-4 py-2 rounded ${
                  isDarkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                }`}
              >
                Close
              </button>
              <button
                onClick={() => {
                  // TODO: Implement model loading - for now just close
                  alert(`Model ${runId} trained successfully! \n\nNext step: Integrate model loading into predictions panel.`);
                  onClose();
                }}
                className={`px-4 py-2 rounded ${
                  isDarkMode 
                    ? 'bg-green-600 hover:bg-green-500 text-white' 
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                Load This Model
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingConfigModal;

