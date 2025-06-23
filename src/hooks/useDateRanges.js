import { useState, useEffect } from 'react';
import { API_BASE_URL } from '../utils/constants';

/**
 * Custom hook to fetch available date ranges for a symbol/timeframe
 * @param {string} symbol - Trading symbol
 * @param {string} timeframe - Timeframe
 * @returns {Object} Hook state with date ranges and loading status
 */
export const useDateRanges = (symbol, timeframe) => {
  const [dateRanges, setDateRanges] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!symbol || !timeframe) {
      setDateRanges(null);
      return;
    }

    const fetchDateRanges = async () => {
      setLoading(true);
      setError(null);

      try {
        // First try to get the summary which includes earliest and latest timestamps
        const summaryResponse = await fetch(`${API_BASE_URL}/summary/${symbol}`);
        
        if (summaryResponse.ok) {
          const summaryData = await summaryResponse.json();
          const timeframeData = summaryData.timeframes?.[timeframe];
          
          if (timeframeData) {
            setDateRanges({
              earliest: timeframeData.earliest_timestamp,
              latest: timeframeData.latest_timestamp,
              count: timeframeData.row_count
            });
          } else {
            // Fallback: try to get a single record to check if table exists
            const testResponse = await fetch(
              `${API_BASE_URL}/data/${symbol}/${timeframe}?limit=1&order=asc`
            );
            
            if (testResponse.ok) {
              const testData = await testResponse.json();
              if (testData.data && testData.data.length > 0) {
                // Get the earliest date
                const earliest = testData.data[0].timestamp;
                
                // Get the latest date
                const latestResponse = await fetch(
                  `${API_BASE_URL}/data/${symbol}/${timeframe}?limit=1&order=desc`
                );
                
                if (latestResponse.ok) {
                  const latestData = await latestResponse.json();
                  const latest = latestData.data[0]?.timestamp;
                  
                  setDateRanges({
                    earliest,
                    latest,
                    count: testData.total_count || 0
                  });
                } else {
                  throw new Error('Failed to fetch latest date');
                }
              } else {
                setDateRanges(null);
              }
            } else {
              throw new Error(`No data available for ${symbol}_${timeframe}`);
            }
          }
        } else {
          throw new Error(`Failed to fetch date ranges for ${symbol}_${timeframe}`);
        }
      } catch (err) {
        console.error('Error fetching date ranges:', err);
        setError(err.message);
        setDateRanges(null);
      } finally {
        setLoading(false);
      }
    };

    fetchDateRanges();
  }, [symbol, timeframe]);

  return {
    dateRanges,
    loading,
    error,
    refetch: () => {
      if (symbol && timeframe) {
        // Re-trigger the effect by updating dependencies
        setDateRanges(null);
        setError(null);
      }
    }
  };
}; 