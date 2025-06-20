import { useState } from 'react';
import PropTypes from 'prop-types';
import { useTrading } from '../../context/TradingContext';
import { useVectorData } from '../../hooks/useVectorData';
import { DEFAULTS } from '../../utils/constants';

// Import Vector Dashboard Components
import VectorControls from './VectorControls';
import VectorStats from './VectorStats';
import VectorTypeSelector from './VectorTypeSelector';
import VectorVisualization from './VectorVisualization';
import LoadingSpinner from '../shared/LoadingSpinner';
import ErrorDisplay from '../shared/ErrorDisplay';

/**
 * VectorDashboard - Main container for the vector dashboard
 */
const VectorDashboard = ({ isDarkMode }) => {
  // Use context for shared state
  const { selectedSymbol, setSelectedSymbol, selectedTimeframe, setSelectedTimeframe, rowLimit, setRowLimit } = useTrading();
  
  // Local state for vector-specific selections
  const [selectedVectorType, setSelectedVectorType] = useState(DEFAULTS.VECTOR_TYPE);
  
  // Fetch vector data using custom hook
  const { 
    vectorData, 
    loading, 
    error, 
    availableVectors, 
    missingVectors, 
    refetch 
  } = useVectorData(selectedSymbol, selectedTimeframe, rowLimit);

  // If there are available vectors but the selected type is not available, switch to the first available
  if (availableVectors.length > 0 && !availableVectors.find(v => v.key === selectedVectorType)) {
    setSelectedVectorType(availableVectors[0].key);
  }

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-2">Trading Vector Dashboard</h2>
          <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Analyze and visualize trading vectors for pattern recognition and ML applications
          </p>
        </div>
        
        {/* Vector Controls */}
        <VectorControls
          selectedSymbol={selectedSymbol}
          setSelectedSymbol={setSelectedSymbol}
          selectedTimeframe={selectedTimeframe}
          setSelectedTimeframe={setSelectedTimeframe}
          rowLimit={rowLimit}
          setRowLimit={setRowLimit}
          onRefresh={refetch}
          isDarkMode={isDarkMode}
        />
        
        {/* Loading State */}
        {loading && (
          <LoadingSpinner isDarkMode={isDarkMode} message="Loading vector data..." />
        )}
        
        {/* Error State */}
        {error && !loading && (
          <ErrorDisplay isDarkMode={isDarkMode} error={error} onRetry={refetch} />
        )}
        
        {/* Main Content */}
        {!loading && !error && vectorData && (
          <div className="space-y-8">
            {/* Vector Statistics */}
            {vectorData.data && vectorData.data.length > 0 && (
              <VectorStats 
                vectorData={vectorData}
                selectedVectorType={selectedVectorType}
                isDarkMode={isDarkMode}
              />
            )}
            
            {/* Vector Type Selection */}
            <VectorTypeSelector
              selectedVectorType={selectedVectorType}
              setSelectedVectorType={setSelectedVectorType}
              availableVectors={availableVectors}
              missingVectors={missingVectors}
              vectorData={vectorData}
              isDarkMode={isDarkMode}
            />
            
            {/* Vector Visualization */}
            {availableVectors.length > 0 && vectorData.data && vectorData.data.length > 0 && (
              <VectorVisualization
                vectorData={vectorData}
                selectedVectorType={selectedVectorType}
                isDarkMode={isDarkMode}
              />
            )}
            
            {/* No Data Message */}
            {(!vectorData.data || vectorData.data.length === 0) && (
              <div className={`text-center py-12 rounded-lg border transition-colors duration-200 ${
                isDarkMode 
                  ? 'bg-gray-800 border-gray-700 text-gray-400' 
                  : 'bg-white border-gray-200 text-gray-600'
              }`}>
                <p className="text-lg mb-2">No vector data available</p>
                <p className="text-sm">
                  Try selecting a different symbol, timeframe, or run the compute.py script to generate vectors
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

VectorDashboard.propTypes = {
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorDashboard; 