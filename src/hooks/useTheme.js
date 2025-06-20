import { useState, useEffect } from 'react';

/**
 * Custom hook for theme management
 * @returns {Object} Theme state and toggle function
 */
export const useTheme = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('tradingDashboard-theme');
    return saved ? saved === 'dark' : false;
  });

  useEffect(() => {
    localStorage.setItem('tradingDashboard-theme', isDarkMode ? 'dark' : 'light');
    // Update document class for global theme
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return { isDarkMode, toggleTheme };
}; 