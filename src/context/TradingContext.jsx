import { createContext, useContext, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { DEFAULTS } from '../utils/constants';

// Create the context
const TradingContext = createContext();

// Storage key for localStorage
const STORAGE_KEY = 'tradingDashboard-state';

/**
 * Trading Context Provider
 * Manages shared state across the Daygent trading platform with localStorage persistence and cross-tab sync
 */
export const TradingProvider = ({ children }) => {
  // Initialize state from localStorage or defaults
  const getInitialState = () => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return {
          selectedSymbol: parsed.selectedSymbol || DEFAULTS.SYMBOL,
          selectedTimeframe: parsed.selectedTimeframe || DEFAULTS.TIMEFRAME,
          rowLimit: parsed.rowLimit || DEFAULTS.ROW_LIMIT,
          sortOrder: parsed.sortOrder || DEFAULTS.SORT_ORDER,
          sortColumn: parsed.sortColumn || DEFAULTS.SORT_COLUMN,
          currentPage: parsed.currentPage || 1,
          searchTerm: parsed.searchTerm || '',
          selectedSearchColumn: parsed.selectedSearchColumn || 'all',
          showColumnFilters: parsed.showColumnFilters || false,
          showDebugInfo: parsed.showDebugInfo || false,
          selectedVectorType: parsed.selectedVectorType || DEFAULTS.VECTOR_TYPE,
          vectorViewMode: parsed.vectorViewMode || 'heatmap',
          selectedCandles: parsed.selectedCandles || [],
          // Date range functionality
          fetchMode: parsed.fetchMode || DEFAULTS.FETCH_MODE,
          dateRangeType: parsed.dateRangeType || DEFAULTS.DATE_RANGE_TYPE,
          startDate: parsed.startDate || DEFAULTS.START_DATE,
          endDate: parsed.endDate || DEFAULTS.END_DATE,
          availableDateRanges: parsed.availableDateRanges || {}
        };
      }
    } catch (error) {
      console.warn('Failed to load state from localStorage:', error);
    }
    
    return {
      selectedSymbol: DEFAULTS.SYMBOL,
      selectedTimeframe: DEFAULTS.TIMEFRAME,
      rowLimit: DEFAULTS.ROW_LIMIT,
      sortOrder: DEFAULTS.SORT_ORDER,
      sortColumn: DEFAULTS.SORT_COLUMN,
      currentPage: 1,
      searchTerm: '',
      selectedSearchColumn: 'all',
      showColumnFilters: false,
      showDebugInfo: false,
      selectedVectorType: DEFAULTS.VECTOR_TYPE,
      vectorViewMode: 'heatmap',
      selectedCandles: [],
      // Date range functionality
      fetchMode: DEFAULTS.FETCH_MODE,
      dateRangeType: DEFAULTS.DATE_RANGE_TYPE,
      startDate: DEFAULTS.START_DATE,
      endDate: DEFAULTS.END_DATE,
      availableDateRanges: {}
    };
  };

  // Shared state
  const [state, setState] = useState(getInitialState);

  // Individual state setters for easier component usage
  const setSelectedSymbol = (value) => {
    setState(prev => ({ ...prev, selectedSymbol: value, currentPage: 1 }));
  };

  const setSelectedTimeframe = (value) => {
    setState(prev => ({ ...prev, selectedTimeframe: value, currentPage: 1 }));
  };

  const setRowLimit = (value) => {
    setState(prev => ({ ...prev, rowLimit: value, currentPage: 1 }));
  };

  const setSortOrder = (value) => {
    setState(prev => ({ ...prev, sortOrder: value, currentPage: 1 }));
  };

  const setSortColumn = (value) => {
    setState(prev => ({ ...prev, sortColumn: value, currentPage: 1 }));
  };

  const setCurrentPage = (value) => {
    setState(prev => ({ ...prev, currentPage: value }));
  };

  const setSearchTerm = (value) => {
    setState(prev => ({ ...prev, searchTerm: value, currentPage: 1 }));
  };

  const setSelectedSearchColumn = (value) => {
    setState(prev => ({ ...prev, selectedSearchColumn: value, currentPage: 1 }));
  };

  const setShowColumnFilters = (value) => {
    setState(prev => ({ ...prev, showColumnFilters: value }));
  };

  const setShowDebugInfo = (value) => {
    setState(prev => ({ ...prev, showDebugInfo: value }));
  };

  const setSelectedVectorType = (value) => {
    setState(prev => ({ ...prev, selectedVectorType: value }));
  };

  const setVectorViewMode = (value) => {
    setState(prev => ({ ...prev, vectorViewMode: value }));
  };

  const setSelectedCandles = (value) => {
    setState(prev => ({ ...prev, selectedCandles: value }));
  };

  const addSelectedCandle = (candle) => {
    setState(prev => ({
      ...prev,
      selectedCandles: [...prev.selectedCandles.filter(c => c.id !== candle.id), candle]
    }));
  };

  const removeSelectedCandle = (candleId) => {
    setState(prev => ({
      ...prev,
      selectedCandles: prev.selectedCandles.filter(c => c.id !== candleId)
    }));
  };

  const clearSelectedCandles = () => {
    setState(prev => ({ ...prev, selectedCandles: [] }));
  };

  // Date range setters
  const setFetchMode = (value) => {
    setState(prev => ({ ...prev, fetchMode: value, currentPage: 1 }));
  };

  const setDateRangeType = (value) => {
    setState(prev => ({ ...prev, dateRangeType: value, currentPage: 1 }));
  };

  const setStartDate = (value) => {
    setState(prev => ({ ...prev, startDate: value, currentPage: 1 }));
  };

  const setEndDate = (value) => {
    setState(prev => ({ ...prev, endDate: value, currentPage: 1 }));
  };

  const setAvailableDateRanges = (value) => {
    setState(prev => ({ ...prev, availableDateRanges: value }));
  };

  // Save to localStorage whenever state changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      
      // Dispatch custom event for cross-tab synchronization
      window.dispatchEvent(new CustomEvent('tradingStateChange', {
        detail: state
      }));
    } catch (error) {
      console.warn('Failed to save state to localStorage:', error);
    }
  }, [state]);

  // Listen for cross-tab state changes
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          const newState = JSON.parse(e.newValue);
          setState(newState);
        } catch (error) {
          console.warn('Failed to parse state from storage event:', error);
        }
      }
    };

    const handleCustomStateChange = (e) => {
      // Update state from other tabs
      setState(e.detail);
    };

    // Listen for localStorage changes (cross-tab)
    window.addEventListener('storage', handleStorageChange);
    
    // Listen for custom events (same tab, different components)
    window.addEventListener('tradingStateChange', handleCustomStateChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('tradingStateChange', handleCustomStateChange);
    };
  }, []);

  const value = {
    // State values
    selectedSymbol: state.selectedSymbol,
    selectedTimeframe: state.selectedTimeframe,
    rowLimit: state.rowLimit,
    sortOrder: state.sortOrder,
    sortColumn: state.sortColumn,
    currentPage: state.currentPage,
    searchTerm: state.searchTerm,
    selectedSearchColumn: state.selectedSearchColumn,
    showColumnFilters: state.showColumnFilters,
    showDebugInfo: state.showDebugInfo,
    selectedVectorType: state.selectedVectorType,
    vectorViewMode: state.vectorViewMode,
    selectedCandles: state.selectedCandles,
    
    // Date range state values
    fetchMode: state.fetchMode,
    dateRangeType: state.dateRangeType,
    startDate: state.startDate,
    endDate: state.endDate,
    availableDateRanges: state.availableDateRanges,
    
    // State setters
    setSelectedSymbol,
    setSelectedTimeframe,
    setRowLimit,
    setSortOrder,
    setSortColumn,
    setCurrentPage,
    setSearchTerm,
    setSelectedSearchColumn,
    setShowColumnFilters,
    setShowDebugInfo,
    setSelectedVectorType,
    setVectorViewMode,
    setSelectedCandles,
    addSelectedCandle,
    removeSelectedCandle,
    clearSelectedCandles,
    
    // Date range setters
    setFetchMode,
    setDateRangeType,
    setStartDate,
    setEndDate,
    setAvailableDateRanges,
    
    // Utility functions
    resetFilters: () => {
      setState(prev => ({
        ...prev,
        searchTerm: '',
        selectedSearchColumn: 'all',
        sortOrder: DEFAULTS.SORT_ORDER,
        showColumnFilters: false,
        currentPage: 1,
        // Reset date range to defaults
        fetchMode: DEFAULTS.FETCH_MODE,
        dateRangeType: DEFAULTS.DATE_RANGE_TYPE,
        startDate: DEFAULTS.START_DATE,
        endDate: DEFAULTS.END_DATE
      }));
    },
    
    resetToDefaults: () => {
      setState({
        selectedSymbol: DEFAULTS.SYMBOL,
        selectedTimeframe: DEFAULTS.TIMEFRAME,
        rowLimit: DEFAULTS.ROW_LIMIT,
        sortOrder: DEFAULTS.SORT_ORDER,
        sortColumn: DEFAULTS.SORT_COLUMN,
        currentPage: 1,
        searchTerm: '',
        selectedSearchColumn: 'all',
        showColumnFilters: false,
        showDebugInfo: false,
        selectedVectorType: DEFAULTS.VECTOR_TYPE,
        vectorViewMode: 'heatmap',
        selectedCandles: [],
        // Date range functionality
        fetchMode: DEFAULTS.FETCH_MODE,
        dateRangeType: DEFAULTS.DATE_RANGE_TYPE,
        startDate: DEFAULTS.START_DATE,
        endDate: DEFAULTS.END_DATE,
        availableDateRanges: {}
      });
    }
  };

  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  );
};

TradingProvider.propTypes = {
  children: PropTypes.node.isRequired
};

/**
 * Custom hook to use the Trading Context
 * @returns {Object} Context value
 */
export const useTrading = () => {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTrading must be used within a TradingProvider');
  }
  return context;
}; 