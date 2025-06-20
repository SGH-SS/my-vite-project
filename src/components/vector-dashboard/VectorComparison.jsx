import { useState } from 'react';
import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { calculateVectorSimilarity } from '../../utils/vectorCalculations';
import { VECTOR_COMPARISON_INFO, SIMILARITY_SCORE_INFO, SIMILARITY_ANALYSIS_INFO } from '../../utils/tooltipContent.jsx';

/**
 * VectorComparison - Side-by-side comparison of two trading vectors
 */
const VectorComparison = ({ vectorData, selectedVectorType, isDarkMode }) => {
  const [compareIndices, setCompareIndices] = useState([0, 1]);

  if (!vectorData?.data || vectorData.data.length < 2) {
    return (
      <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Need at least 2 vectors for comparison
      </div>
    );
  }

  const vectors = compareIndices.map(i => vectorData.data[i]?.[selectedVectorType]).filter(v => v);
  if (vectors.length < 2) {
    return (
      <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Invalid comparison indices
      </div>
    );
  }

  const maxDimensions = Math.min(vectors[0]?.length || 0, 15);
  const similarity = calculateVectorSimilarity(vectors[0], vectors[1]);

  return (
    <div className="space-y-4">
      {/* Comparison Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <h4 className={`text-sm font-semibold ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>Vector Comparison</h4>
          <InfoTooltip 
            id="vector-comparison" 
            content={VECTOR_COMPARISON_INFO.content} 
            isDarkMode={isDarkMode} 
          />
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Compare:</label>
            <select
              value={compareIndices[0]}
              onChange={(e) => setCompareIndices([parseInt(e.target.value), compareIndices[1]])}
              className={`text-xs px-2 py-1 rounded border ${
                isDarkMode 
                  ? 'bg-gray-700 border-gray-600 text-gray-300' 
                  : 'bg-white border-gray-300'
              }`}
            >
              {vectorData.data.slice(0, 20).map((_, i) => (
                <option key={i} value={i}>Candle #{i + 1}</option>
              ))}
            </select>
            <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>vs</span>
            <select
              value={compareIndices[1]}
              onChange={(e) => setCompareIndices([compareIndices[0], parseInt(e.target.value)])}
              className={`text-xs px-2 py-1 rounded border ${
                isDarkMode 
                  ? 'bg-gray-700 border-gray-600 text-gray-300' 
                  : 'bg-white border-gray-300'
              }`}
            >
              {vectorData.data.slice(0, 20).map((_, i) => (
                <option key={i} value={i}>Candle #{i + 1}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center">
            <div className={`text-sm px-3 py-1 rounded transition-colors duration-200 ${
              similarity > 0.8 
                ? isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-800'
                : similarity > 0.6 
                ? isDarkMode ? 'bg-yellow-900/30 text-yellow-300' : 'bg-yellow-100 text-yellow-800'
                : isDarkMode ? 'bg-red-900/30 text-red-300' : 'bg-red-100 text-red-800'
            }`}>
              Similarity: {(similarity * 100).toFixed(1)}%
            </div>
            <InfoTooltip 
              id="similarity-score" 
              content={SIMILARITY_SCORE_INFO.content} 
              isDarkMode={isDarkMode} 
            />
          </div>
        </div>
      </div>

      {/* Vector Comparison Display */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {vectors.map((vector, vectorIndex) => {
          const candleData = vectorData.data[compareIndices[vectorIndex]];
          return (
            <div key={vectorIndex} className={`p-4 rounded-lg border transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
            }`}>
              <div className={`text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-200' : 'text-gray-800'
              }`}>
                Candle #{compareIndices[vectorIndex] + 1}
              </div>
              <div className={`text-xs mb-3 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {new Date(candleData.timestamp).toLocaleString()}
              </div>
              
              {/* OHLC Info if available */}
              {candleData.open && (
                <div className={`text-xs mb-3 grid grid-cols-4 gap-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  <div>O: {candleData.open.toFixed(2)}</div>
                  <div>H: {candleData.high.toFixed(2)}</div>
                  <div>L: {candleData.low.toFixed(2)}</div>
                  <div>C: {candleData.close.toFixed(2)}</div>
                </div>
              )}
              
              {/* Vector Values */}
              <div className="grid grid-cols-5 gap-1 text-xs">
                {vector.slice(0, maxDimensions).map((value, dimIndex) => {
                  const otherValue = vectors[1 - vectorIndex]?.[dimIndex];
                  const diff = otherValue ? Math.abs(value - otherValue) : 0;
                  const isHighDiff = diff > (Math.abs(value) * 0.1); // 10% difference threshold
                  
                  return (
                    <div key={dimIndex} className={`p-1 rounded text-center font-mono transition-colors duration-200 ${
                      isHighDiff 
                        ? isDarkMode ? 'bg-red-900/30 border border-red-700' : 'bg-red-50 border border-red-200'
                        : isDarkMode ? 'bg-gray-800' : 'bg-gray-50'
                    }`}>
                      <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-xs`}>D{dimIndex}</div>
                      <div className={`font-bold ${
                        value >= 0 ? 'text-green-500' : 'text-red-500'
                      }`}>
                        {value.toFixed(3)}
                      </div>
                      {otherValue && (
                        <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                          Δ{(value - otherValue).toFixed(3)}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              
              {maxDimensions < vector.length && (
                <div className={`text-xs mt-2 text-center ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Showing {maxDimensions} of {vector.length} dimensions
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Similarity Analysis */}
      <div className={`p-3 rounded-lg border transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
      }`}>
        <div className="flex items-center mb-2">
          <h5 className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>Similarity Analysis</h5>
          <InfoTooltip 
            id="similarity-analysis" 
            content={SIMILARITY_ANALYSIS_INFO.content} 
            isDarkMode={isDarkMode} 
          />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Cosine Similarity:</span>
            <div className="font-bold text-blue-500">{(similarity * 100).toFixed(2)}%</div>
          </div>
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Interpretation:</span>
            <div className={`font-medium ${
              similarity > 0.9 ? 'text-green-600' :
              similarity > 0.7 ? 'text-yellow-600' :
              similarity > 0.5 ? 'text-orange-600' : 'text-red-600'
            }`}>
              {similarity > 0.9 ? 'Very Similar' :
               similarity > 0.7 ? 'Similar' :
               similarity > 0.5 ? 'Somewhat Similar' : 'Different'}
            </div>
          </div>
          <div>
            <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Pattern Match:</span>
            <div className={`font-medium ${
              similarity > 0.8 ? 'text-green-600' : 'text-gray-500'
            }`}>
              {similarity > 0.8 ? 'Strong pattern match detected' : 'No strong pattern match'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

VectorComparison.propTypes = {
  vectorData: PropTypes.shape({
    data: PropTypes.arrayOf(PropTypes.object)
  }),
  selectedVectorType: PropTypes.string.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorComparison;
