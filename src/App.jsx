import { useState } from 'react';
import { TradingProvider } from './context/TradingContext';
import { useTheme } from './hooks/useTheme';
import { DASHBOARD_MODES } from './utils/constants';

// Shared Components
import ThemeToggle from './components/shared/ThemeToggle';

// Modular Dashboard Components
import DataDashboard from './components/data-dashboard/DataDashboard';
import VectorDashboard from './components/vector-dashboard/VectorDashboard';
import PipelineDashboard from './components/PipelineDashboard';
import ModelTraining from './components/model';

// Original Monolithic Dashboard Component
import OriginalTradingDashboard from './components/TradingDashboard';

/**
 * Main Application Component
 * Supports both modular architecture and original monolithic structure
 */
function App() {
  const { isDarkMode, toggleTheme } = useTheme();
  const [dashboardMode, setDashboardMode] = useState(DASHBOARD_MODES.DATA);
  const [architectureMode, setArchitectureMode] = useState('monolithic'); // 'modular' or 'monolithic'

  return (
    <TradingProvider>
      <div className={`min-h-screen transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        {/* Header */}
        <header className={`transition-colors duration-200 border-b ${
          isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <h1 className="text-xl font-black tracking-tight" style={{ 
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          letterSpacing: '-0.02em'
        }}>
          Day<span className="text-blue-600">gent</span>
        </h1>
                
                {/* Architecture Mode Toggle */}
                <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                }`}>
                  <span className={`text-xs font-medium ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    Architecture:
                  </span>
                  <select
                    value={architectureMode}
                    onChange={(e) => setArchitectureMode(e.target.value)}
                    className={`text-xs font-medium rounded px-2 py-1 border-0 focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                      isDarkMode 
                        ? 'bg-gray-800 text-gray-200' 
                        : 'bg-white text-gray-800'
                    }`}
                  >
                    <option value="modular">🧩 Modular (23 Components)</option>
                    <option value="monolithic">📄 Original (2.8k LOC)</option>
                  </select>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                {/* Dashboard Mode Toggle - Only show for modular architecture */}
                {architectureMode === 'modular' && (
                  <div className={`flex rounded-lg p-1 transition-colors duration-200 ${
                    isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    <button
                      onClick={() => setDashboardMode(DASHBOARD_MODES.DATA)}
                      className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                        dashboardMode === DASHBOARD_MODES.DATA
                          ? isDarkMode
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'bg-white text-blue-600 shadow-sm'
                          : isDarkMode
                            ? 'text-gray-300 hover:text-white'
                            : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      📊 Data
                    </button>
                    <button
                      onClick={() => setDashboardMode(DASHBOARD_MODES.VECTOR)}
                      className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                        dashboardMode === DASHBOARD_MODES.VECTOR
                          ? isDarkMode
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'bg-white text-blue-600 shadow-sm'
                          : isDarkMode
                            ? 'text-gray-300 hover:text-white'
                            : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      🧮 Vectors
                    </button>
                    <button
                      onClick={() => setDashboardMode(DASHBOARD_MODES.PIPELINE)}
                      className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                        dashboardMode === DASHBOARD_MODES.PIPELINE
                          ? isDarkMode
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'bg-white text-blue-600 shadow-sm'
                          : isDarkMode
                            ? 'text-gray-300 hover:text-white'
                            : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      ⚙️ Pipeline
                    </button>
                    <button
                      onClick={() => setDashboardMode(DASHBOARD_MODES.MODEL)}
                      className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                        dashboardMode === DASHBOARD_MODES.MODEL
                          ? isDarkMode
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'bg-white text-blue-600 shadow-sm'
                          : isDarkMode
                            ? 'text-gray-300 hover:text-white'
                            : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      🧪 Model Training
                    </button>
                  </div>
                )}
                
                {/* Architecture Info Badge */}
                <div className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors duration-200 ${
                  architectureMode === 'modular'
                    ? isDarkMode 
                      ? 'bg-green-900/30 text-green-300 border border-green-700'
                      : 'bg-green-100 text-green-800 border border-green-200'
                    : isDarkMode
                      ? 'bg-yellow-900/30 text-yellow-300 border border-yellow-700'
                      : 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                }`}>
                  {architectureMode === 'modular' ? '✨ Modular' : '⚡ Classic'}
                </div>
                
                {/* Theme Toggle */}
                <ThemeToggle isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
              </div>
            </div>
          </div>
        </header>

        {/* Architecture Info Banner */}
        <div className={`transition-colors duration-200 border-b ${
          isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="py-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-4">
                  <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                    {architectureMode === 'modular' ? (
                      <>
                        🧩 <strong>Modular Architecture:</strong> 23 reusable components • Maintainable • Scalable
                      </>
                    ) : (
                      <>
                        📄 <strong>Original Monolithic:</strong> Single 2,845-line component • All functionality preserved
                      </>
                    )}
                  </span>
                </div>
                <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Switch architecture anytime to compare approaches
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main>
          {architectureMode === 'monolithic' ? (
            /* Original Monolithic Dashboard */
            <OriginalTradingDashboard />
          ) : (
            /* Modular Dashboard */
            dashboardMode === DASHBOARD_MODES.DATA
              ? <DataDashboard isDarkMode={isDarkMode} />
              : dashboardMode === DASHBOARD_MODES.VECTOR
                ? <VectorDashboard isDarkMode={isDarkMode} />
                : dashboardMode === DASHBOARD_MODES.PIPELINE
                  ? <PipelineDashboard isDarkMode={isDarkMode} />
                  : <ModelTraining isDarkMode={isDarkMode} />
          )}
        </main>
      </div>
    </TradingProvider>
  );
}

export default App;
