import { useState } from 'react';
import PropTypes from 'prop-types';
import VectorHeatmap from './VectorHeatmap';
import VectorComparison from './VectorComparison';
import InfoTooltip from '../shared/InfoTooltip';
import { VECTOR_VIEW_MODES } from '../../utils/constants';

/**
 * VectorVisualization - Container for vector visualization modes
 */
const VectorVisualization = ({ 
  vectorData, 
  selectedVectorType, 
  isDarkMode 
}) => {
  const [viewMode, setViewMode] = useState(VECTOR_VIEW_MODES.HEATMAP);

  const viewModeInfo = (
    <div>
      <p className="font-semibold mb-2">👁️ View Modes</p>
      <p className="mb-2">Different ways to visualize your vector data:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Heatmap:</strong> Color-coded matrix view of all vectors</li>
        <li><strong>Comparison:</strong> Side-by-side analysis of two vectors</li>
      </ul>
      <p className="mt-2 text-xs">Switch between modes based on your analysis needs.</p>
    </div>
  );

  return (
    <div className={`p-6 rounded-lg border transition-colors duration-200 ${
      isDarkMode 
        ? 'bg-gray-800 border-gray-700' 
        : 'bg-white border-gray-200'
    }`}>
      {/* Header with view mode toggle */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Vector Visualization</h3>
          <InfoTooltip 
            id="view-mode-info" 
            content={viewModeInfo} 
            isDarkMode={isDarkMode} 
          />
        </div>
        <div className={`flex rounded-lg p-1 transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
        }`}>
          <button
            onClick={() => setViewMode(VECTOR_VIEW_MODES.HEATMAP)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
              viewMode === VECTOR_VIEW_MODES.HEATMAP
                ? isDarkMode
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-white text-blue-600 shadow-sm'
                : isDarkMode
                  ? 'text-gray-300 hover:text-white'
                  : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            🔥 Heatmap
          </button>
          <button
            onClick={() => setViewMode(VECTOR_VIEW_MODES.COMPARISON)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
              viewMode === VECTOR_VIEW_MODES.COMPARISON
                ? isDarkMode
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-white text-blue-600 shadow-sm'
                : isDarkMode
                  ? 'text-gray-300 hover:text-white'
                  : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            ⚖️ Comparison
          </button>
        </div>
      </div>

      {/* Visualization Content */}
      <div className="mt-4">
        {viewMode === VECTOR_VIEW_MODES.HEATMAP ? (
          <VectorHeatmap 
            vectorData={vectorData}
            selectedVectorType={selectedVectorType}
            isDarkMode={isDarkMode}
          />
        ) : (
          <VectorComparison 
            vectorData={vectorData}
            selectedVectorType={selectedVectorType}
            isDarkMode={isDarkMode}
          />
        )}
      </div>
    </div>
  );
};

VectorVisualization.propTypes = {
  vectorData: PropTypes.shape({
    data: PropTypes.arrayOf(PropTypes.object)
  }),
  selectedVectorType: PropTypes.string.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorVisualization;
