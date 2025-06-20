import { useState, useEffect } from 'react';
import { API_BASE_URL } from '../utils/constants';

/**
 * Custom hook for fetching and managing trading data
 * @param {string} selectedSymbol - The selected trading symbol
 * @param {string} selectedTimeframe - The selected timeframe
 * @param {number} rowLimit - Number of rows to fetch
 * @param {string} sortOrder - Sort order (asc/desc)
 * @param {string} sortColumn - Column to sort by
 * @param {number} currentPage - Current page number
 * @returns {Object} Trading data state and methods
 */
export const useTradingData = (
  selectedSymbol, 
  selectedTimeframe, 
  rowLimit, 
  sortOrder, 
  sortColumn, 
  currentPage
) => {
  const [tradingData, setTradingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastFetchInfo, setLastFetchInfo] = useState(null);

  const fetchTradingData = async () => {
    if (!selectedSymbol || !selectedTimeframe) return;
    
    setLoading(true);
    console.log(`🚀 fetchTradingData called - Page: ${currentPage}, Symbol: ${selectedSymbol}, Timeframe: ${selectedTimeframe}`);
    
    try {
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}`;
      const offset = (currentPage - 1) * rowLimit;
      
      if (sortOrder === 'asc') {
        url += `&offset=${offset}&order=asc&sort_by=timestamp`;
      } else {
        url += `&offset=${offset}&order=desc&sort_by=timestamp`;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      
      const data = JSON.parse(text);
      
      // Client-side sorting if needed
      if (data.data && Array.isArray(data.data)) {
        const sortedData = [...data.data].sort((a, b) => {
          let aVal = a[sortColumn];
          let bVal = b[sortColumn];
          
          if (sortColumn === 'timestamp') {
            aVal = new Date(aVal);
            bVal = new Date(bVal);
          } else if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
          }
          
          if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
          if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
          return 0;
        });
        
        data.data = sortedData;
      }
      
      // Set fetch info for debugging
      setLastFetchInfo({
        timestamp: new Date().toLocaleTimeString(),
        sortOrder,
        rowLimit,
        currentPage,
        offset,
        apiUrl: url,
        totalRecords: data.total_count || data.count
      });
      
      setTradingData(data);
      setError(null);
    } catch (err) {
      setError(`Failed to fetch trading data: ${err.message}`);
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTradingData();
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, sortColumn, currentPage]);

  return {
    tradingData,
    loading,
    error,
    lastFetchInfo,
    refetch: fetchTradingData
  };
}; 