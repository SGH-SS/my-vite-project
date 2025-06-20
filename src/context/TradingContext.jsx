import { createContext, useContext, useState } from 'react';
import PropTypes from 'prop-types';
import { DEFAULTS } from '../utils/constants';

// Create the context
const TradingContext = createContext();

/**
 * Trading Context Provider
 * Manages shared state across the trading dashboard
 */
export const TradingProvider = ({ children }) => {
  // Shared state
  const [selectedSymbol, setSelectedSymbol] = useState(DEFAULTS.SYMBOL);
  const [selectedTimeframe, setSelectedTimeframe] = useState(DEFAULTS.TIMEFRAME);
  const [rowLimit, setRowLimit] = useState(DEFAULTS.ROW_LIMIT);
  const [dashboardMode, setDashboardMode] = useState(DEFAULTS.DASHBOARD_MODE);

  const value = {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    dashboardMode,
    setDashboardMode
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