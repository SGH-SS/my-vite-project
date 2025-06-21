import { useState } from 'react';
import { TradingProvider } from './context/TradingContext';
import { useTheme } from './hooks/useTheme';
import { DASHBOARD_MODES } from './utils/constants';

// Shared Components
import ThemeToggle from './components/shared/ThemeToggle';

// Dashboard Components
import DataDashboard from './components/data-dashboard/DataDashboard';
import VectorDashboard from './components/vector-dashboard/VectorDashboard';

/**
 * Main Application Component
 * Orchestrates the Daygent Trading Platform with modular architecture
 */
function App() {
  const { isDarkMode, toggleTheme } = useTheme();
  const [dashboardMode, setDashboardMode] = useState(DASHBOARD_MODES.DATA);

  return (
    <TradingProvider>
      <div className={`min-h-screen transition-colors duration-200 p-6 ${
        isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
      }`}>
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className={`text-4xl font-bold mb-2 transition-colors duration-200 ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>
                  {dashboardMode === DASHBOARD_MODES.DATA 
                    ? 'Trading Data Dashboard' 
                    : 'Trading Vector Dashboard'}
                </h1>
                <p className={`transition-colors duration-200 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  {dashboardMode === DASHBOARD_MODES.DATA 
                    ? 'Real-time access to your PostgreSQL trading database • pgAdmin but better'
                    : 'ML vector analysis and pattern recognition • AI-powered insights'
                  }
                </p>
              </div>
              
              {/* Controls */}
              <div className="flex items-center space-x-4">
                <ThemeToggle isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
                
                {/* Dashboard Mode Toggle */}
                <DashboardModeToggle 
                  currentMode={dashboardMode}
                  onModeChange={setDashboardMode}
                  isDarkMode={isDarkMode}
                />
              </div>
            </div>
          </div>

          {/* Dashboard Content */}
          {dashboardMode === DASHBOARD_MODES.DATA ? (
            <DataDashboard isDarkMode={isDarkMode} />
          ) : (
            <VectorDashboard isDarkMode={isDarkMode} />
          )}
        </div>
      </div>
    </TradingProvider>
  );
}

/**
 * Dashboard Mode Toggle Component
 * Switches between Data and Vector dashboards
 */
const DashboardModeToggle = ({ currentMode, onModeChange, isDarkMode }) => {
  return (
    <div className={`flex rounded-lg p-1 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-gray-100'
    }`}>
      <button
        onClick={() => onModeChange(DASHBOARD_MODES.DATA)}
        className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
          currentMode === DASHBOARD_MODES.DATA
            ? isDarkMode 
              ? 'bg-gray-700 text-blue-400 shadow-sm ring-1 ring-blue-500/50'
              : 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200'
            : isDarkMode
              ? 'text-gray-300 hover:text-white hover:bg-gray-700'
              : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
        }`}
      >
        <span>📊</span>
        <span>Data Dashboard</span>
      </button>
      <button
        onClick={() => onModeChange(DASHBOARD_MODES.VECTOR)}
        className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
          currentMode === DASHBOARD_MODES.VECTOR
            ? isDarkMode
              ? 'bg-gray-700 text-purple-400 shadow-sm ring-1 ring-purple-500/50'
              : 'bg-white text-purple-600 shadow-sm ring-1 ring-purple-200'
            : isDarkMode
              ? 'text-gray-300 hover:text-white hover:bg-gray-700'
              : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
        }`}
      >
        <span>🧠</span>
        <span>Vector Dashboard</span>
      </button>
    </div>
  );
};

export default App; 