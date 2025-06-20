/**
 * Formatting utility functions for the Trading Dashboard
 */

/**
 * Format a price value to 4 decimal places
 * @param {number} price - The price to format
 * @returns {string} Formatted price string
 */
export const formatPrice = (price) => {
  return price ? price.toFixed(4) : 'N/A';
};

/**
 * Format a timestamp to a localized string
 * @param {string|Date} timestamp - The timestamp to format
 * @returns {string} Formatted timestamp string
 */
export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleString();
};

/**
 * Get the color class for a symbol
 * @param {string} symbol - The trading symbol
 * @returns {string} Tailwind color class
 */
export const getSymbolColor = (symbol) => {
  switch (symbol) {
    case 'es': return 'text-blue-600';
    case 'eurusd': return 'text-green-600';
    case 'spy': return 'text-purple-600';
    default: return 'text-gray-600';
  }
};

/**
 * Format a number with thousand separators
 * @param {number} num - The number to format
 * @returns {string} Formatted number string
 */
export const formatNumber = (num) => {
  return num ? num.toLocaleString() : 'N/A';
};

/**
 * Determine candle type based on open and close prices
 * @param {number} open - Opening price
 * @param {number} close - Closing price
 * @returns {Object} Candle type info with type, color, and icon
 */
export const getCandleType = (open, close) => {
  if (close > open) {
    return {
      type: 'BULL',
      color: 'text-green-600',
      icon: '🟢'
    };
  } else if (close < open) {
    return {
      type: 'BEAR',
      color: 'text-red-600',
      icon: '🔴'
    };
  } else {
    return {
      type: 'DOJI',
      color: 'text-gray-500',
      icon: '➖'
    };
  }
};

/**
 * Clean search term by removing commas
 * @param {string} term - The search term to clean
 * @returns {string} Cleaned search term
 */
export const cleanSearchTerm = (term) => {
  return term.replace(/,/g, '');
}; 