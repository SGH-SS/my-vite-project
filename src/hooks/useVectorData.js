import { useState, useEffect } from 'react';
import { API_BASE_URL, VECTOR_TYPES } from '../utils/constants';

/**
 * Custom hook for fetching and managing vector data
 * @param {string} selectedSymbol - The selected trading symbol
 * @param {string} selectedTimeframe - The selected timeframe
 * @param {number} rowLimit - Number of rows to fetch
 * @returns {Object} Vector data state and methods
 */
export const useVectorData = (selectedSymbol, selectedTimeframe, rowLimit) => {
  const [vectorData, setVectorData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableVectors, setAvailableVectors] = useState([]);
  const [missingVectors, setMissingVectors] = useState([]);

  const fetchVectorData = async () => {
    if (!selectedSymbol || !selectedTimeframe) return;
    
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}&include_vectors=true`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setVectorData(data);
      
      // Check which vector types are available
      if (data.data && data.data.length > 0) {
        const sampleRow = data.data[0];
        const available = VECTOR_TYPES.filter(type => 
          sampleRow.hasOwnProperty(type.key) && sampleRow[type.key] !== null && sampleRow[type.key] !== undefined
        );
        const missing = VECTOR_TYPES.filter(type => 
          !sampleRow.hasOwnProperty(type.key) || sampleRow[type.key] === null || sampleRow[type.key] === undefined
        );
        
        setAvailableVectors(available);
        setMissingVectors(missing);
      }
      
      setError(null);
    } catch (err) {
      setError(`Failed to fetch vector data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchVectorData();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit]);

  return {
    vectorData,
    loading,
    error,
    availableVectors,
    missingVectors,
    allVectorTypes: VECTOR_TYPES,
    refetch: fetchVectorData
  };
}; 