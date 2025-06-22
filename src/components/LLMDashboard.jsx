import React, { useState, useEffect } from 'react';
import { useTrading } from '../context/TradingContext';
import SelectedCandlesPanel from './shared/SelectedCandlesPanel.jsx';

// Reusable InfoTooltip component (shared across dashboards)
const InfoTooltip = ({ id, content, isDarkMode, asSpan = false }) => {
  const [activeTooltip, setActiveTooltip] = useState(null);
  const isActive = activeTooltip === id;
  
  const triggerClasses = `ml-2 w-4 h-4 rounded-full border text-xs font-bold transition-all duration-200 cursor-pointer ${
    isDarkMode 
      ? 'border-gray-500 text-gray-400 hover:border-blue-400 hover:text-blue-400' 
      : 'border-gray-400 text-gray-500 hover:border-blue-500 hover:text-blue-600'
  }`;

  const triggerProps = {
    onClick: () => setActiveTooltip(isActive ? null : id),
    onMouseEnter: () => setActiveTooltip(id),
    onMouseLeave: () => setActiveTooltip(null),
    className: triggerClasses
  };
  
  return (
    <div className="relative">
      {asSpan ? (
        <span {...triggerProps}>i</span>
      ) : (
        <button {...triggerProps}>i</button>
      )}
      {isActive && (
        <div className={`absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-80 p-4 rounded-lg shadow-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-gray-800 border-gray-600 text-gray-200' 
            : 'bg-white border-gray-200 text-gray-800'
        }`}>
          <div className="text-sm leading-relaxed">{content}</div>
          {/* Arrow pointing down */}
          <div className={`absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[8px] border-r-[8px] border-t-[8px] border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`}></div>
        </div>
      )}
    </div>
  );
};

// Reusable Data Selection & Controls component (shared across all dashboards)
const DataSelectionControls = ({ 
  handleRefresh, 
  isDarkMode,
  dashboardType = 'llm'
}) => {
  // Use shared trading context
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder
  } = useTrading();

  return (
    <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>Data Selection & Controls</h3>
        <div className="flex gap-2">
          <button
            onClick={handleRefresh}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
              isDarkMode 
                ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
          >
            🔄 Refresh Analysis
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Symbol
          </label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="es">ES (E-mini S&P 500)</option>
            <option value="eurusd">EURUSD (Euro/US Dollar)</option>
            <option value="spy">SPY (SPDR S&P 500 ETF)</option>
          </select>
        </div>
        
        <div>
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Timeframe
          </label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="30m">30 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Data Limit
          </label>
          <select
            value={rowLimit}
            onChange={(e) => setRowLimit(Number(e.target.value))}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value={25}>25 records</option>
            <option value={50}>50 records</option>
            <option value={100}>100 records</option>
            <option value={250}>250 records</option>
            <option value={500}>500 records</option>
            <option value={1000}>1000 records</option>
            <option value={2000}>2000 records</option>
          </select>
        </div>

        <div>
          <div className="flex items-center mb-2">
            <label className={`block text-sm font-medium ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Sort Order
            </label>
            <InfoTooltip id="sort-order" content={
              <div>
                <p className="font-semibold mb-2">⬇️ Sort Order Options</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Descending (Newest first):</strong> Shows most recent trading data</li>
                  <li><strong>Ascending (Oldest first):</strong> Shows historical data from the beginning</li>
                </ul>
                <p className="mt-2 text-xs"><strong>Note:</strong> Sort order is synchronized across all dashboards. This affects both data tables and charts.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <select
            value={sortOrder || 'desc'}
            onChange={(e) => setSortOrder && setSortOrder(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="desc">⬇ Descending (Newest first)</option>
            <option value="asc">⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>
    </div>
  );
};

const TradingLLMDashboard = ({ 
  tables,
  isDarkMode 
}) => {
  // Use shared trading context
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder
  } = useTrading();

  const [llmData, setLlmData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisMode, setAnalysisMode] = useState('comprehensive'); // comprehensive, signals, sentiment
  const [modelType, setModelType] = useState('gpt4'); // gpt4, claude, local
  const [miniDashboardView, setMiniDashboardView] = useState('data'); // data, vector, chart
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI trading assistant. I can help you analyze market data, explain patterns, and provide trading insights. What would you like to know about the current market?',
      timestamp: new Date().toISOString()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  // Mock LLM analysis data structure
  const [mockAnalysis, setMockAnalysis] = useState({
    marketSentiment: {
      overall: 'Bullish',
      confidence: 0.78,
      reasoning: "Strong momentum patterns and volume confirmation suggest continued upward movement.",
      signals: ['Rising volume', 'Breakout pattern', 'RSI showing strength']
    },
    tradingSignals: {
      primary: 'BUY',
      strength: 'Strong',
      entryPrice: 4250.75,
      stopLoss: 4180.50,
      targets: [4320.00, 4385.25, 4450.00],
      riskReward: 2.8
    },
    riskAssessment: {
      level: 'Medium',
      factors: [
        'Volatility within normal range',
        'Strong support at 4200 level', 
        'Economic calendar shows moderate events'
      ],
      recommendation: 'Position size: 2-3% of portfolio'
    },
    patternRecognition: [
      { pattern: 'Ascending Triangle', confidence: 0.85, timeframe: '4h' },
      { pattern: 'Bull Flag', confidence: 0.72, timeframe: '1h' },
      { pattern: 'Volume Accumulation', confidence: 0.91, timeframe: '1d' }
    ],
    aiInsights: [
      "Current price action suggests institutional accumulation phase",
      "Technical indicators align with fundamental market strength",
      "Options flow shows bullish positioning in near-term strikes",
      "Historical patterns suggest 70% probability of upward continuation"
    ]
  });

  const API_BASE_URL = 'http://localhost:8000/api/trading';

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchLLMAnalysis();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, analysisMode, modelType]);

  // Update mini dashboard data when selection changes
  useEffect(() => {
    if (isInitialLoad) {
      setIsInitialLoad(false);
      return;
    }

    // Add a welcome message when switching symbols or timeframes
    if (selectedSymbol && selectedTimeframe) {
      const contextMessage = {
        role: 'assistant',
        content: `I've updated the mini dashboard view to show ${selectedSymbol.toUpperCase()} ${selectedTimeframe} data. The ${miniDashboardView} dashboard is now displaying current market information. Feel free to ask me about any patterns or trends you notice!`,
        timestamp: new Date().toISOString()
      };
      
      setChatMessages(prev => [...prev, contextMessage]);
    }
  }, [selectedSymbol, selectedTimeframe, miniDashboardView]);

  const fetchLLMAnalysis = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // In a real implementation, this would call your LLM backend
      // const response = await fetch(`${API_BASE_URL}/llm/analyze/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}&model=${modelType}&mode=${analysisMode}`);
      
      // For now, simulate successful analysis with mock data
      setLlmData({
        symbol: selectedSymbol,
        timeframe: selectedTimeframe,
        timestamp: new Date().toISOString(),
        analysis: {
          ...mockAnalysis,
          // Randomize some values to simulate real updates
          marketSentiment: {
            ...mockAnalysis.marketSentiment,
            confidence: 0.65 + Math.random() * 0.3
          },
          tradingSignals: {
            ...mockAnalysis.tradingSignals,
            entryPrice: 4200 + Math.random() * 100,
            riskReward: 2.0 + Math.random() * 2.0
          }
        }
      });
      
    } catch (err) {
      setError(`Failed to fetch LLM analysis: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchLLMAnalysis();
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMessage = {
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date().toISOString()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = {
        role: 'assistant',
        content: generateAIResponse(chatInput.trim()),
        timestamp: new Date().toISOString()
      };
      setChatMessages(prev => [...prev, aiResponse]);
    }, 1000 + Math.random() * 2000);
  };

  const generateAIResponse = (userInput) => {
    const responses = [
      `Based on the current ${selectedSymbol.toUpperCase()} ${selectedTimeframe} data, I can see that the market is showing ${llmData?.analysis?.marketSentiment?.overall || 'mixed'} sentiment. The AI analysis suggests a ${llmData?.analysis?.tradingSignals?.primary || 'HOLD'} signal with ${llmData?.analysis?.tradingSignals?.strength || 'medium'} strength.`,
      
      `Looking at the ${miniDashboardView} dashboard data for ${selectedSymbol.toUpperCase()}, the technical indicators are showing interesting patterns. The recent price action suggests ${Math.random() > 0.5 ? 'bullish' : 'bearish'} momentum in the ${selectedTimeframe} timeframe.`,
      
      `From my analysis of the current market conditions, I recommend paying attention to the ${selectedTimeframe} patterns. The risk assessment shows ${llmData?.analysis?.riskAssessment?.level || 'medium'} risk level, which suggests ${Math.random() > 0.5 ? 'conservative' : 'moderate'} position sizing.`,
      
      `The vector analysis combined with traditional technical indicators shows ${Math.random() > 0.6 ? 'strong correlation' : 'divergence'} in the current ${selectedSymbol.toUpperCase()} setup. This could indicate ${Math.random() > 0.5 ? 'continuation' : 'reversal'} patterns forming.`,
      
      `Based on your question about "${userInput.slice(0, 50)}...", I'd recommend checking the ${miniDashboardView} view above for additional context. The current market structure shows ${Math.random() > 0.5 ? 'support' : 'resistance'} levels that could be significant for your trading decisions.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getSentimentEmoji = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'bullish': return '🐂';
      case 'bearish': return '🐻';
      case 'neutral': return '😐';
      default: return '📊';
    }
  };

  const getSignalEmoji = (signal) => {
    switch (signal.toLowerCase()) {
      case 'buy': return '🟢';
      case 'sell': return '🔴';
      case 'hold': return '🟡';
      default: return '⚪';
    }
  };

  // Mini Dashboard Components
  const MiniDataDashboard = () => (
    <div className={`p-4 rounded-lg border transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Recent Trading Data - {selectedSymbol.toUpperCase()} {selectedTimeframe}
        </h4>
        <span className={`text-xs px-2 py-1 rounded ${
          isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
        }`}>
          Live Data
        </span>
      </div>
      <div className="grid grid-cols-5 gap-2 text-xs">
        <div className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Time</div>
        <div className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Open</div>
        <div className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>High</div>
        <div className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Low</div>
        <div className={`font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Close</div>
        
        {/* Mock recent data */}
        {Array.from({length: 5}, (_, i) => {
          const basePrice = selectedSymbol === 'spy' ? 440 : selectedSymbol === 'es' ? 4400 : 1.08;
          const variation = (Math.random() - 0.5) * (basePrice * 0.02);
          const open = basePrice + variation;
          const high = open + Math.random() * (basePrice * 0.01);
          const low = open - Math.random() * (basePrice * 0.01);
          const close = low + Math.random() * (high - low);
          
          return (
            <React.Fragment key={i}>
              <div className={`${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {new Date(Date.now() - i * 60000).toLocaleTimeString().slice(0, 5)}
              </div>
              <div className={`font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {open.toFixed(selectedSymbol === 'eurusd' ? 4 : 2)}
              </div>
              <div className={`font-mono text-green-500`}>
                {high.toFixed(selectedSymbol === 'eurusd' ? 4 : 2)}
              </div>
              <div className={`font-mono text-red-500`}>
                {low.toFixed(selectedSymbol === 'eurusd' ? 4 : 2)}
              </div>
              <div className={`font-mono font-bold ${close > open ? 'text-green-500' : 'text-red-500'}`}>
                {close.toFixed(selectedSymbol === 'eurusd' ? 4 : 2)}
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );

  const MiniVectorDashboard = () => (
    <div className={`p-4 rounded-lg border transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Vector Analysis - {selectedSymbol.toUpperCase()} Patterns
        </h4>
        <span className={`text-xs px-2 py-1 rounded ${
          isDarkMode ? 'bg-purple-900/30 text-purple-300' : 'bg-purple-100 text-purple-700'
        }`}>
          AI Vectors
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className={`text-xs mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Pattern Similarity Heatmap
          </div>
          <div className="grid grid-cols-8 gap-1">
            {Array.from({length: 32}, (_, i) => {
              const intensity = Math.random();
              return (
                <div
                  key={i}
                  className="w-3 h-3 rounded-sm"
                  style={{
                    backgroundColor: intensity > 0.7 ? '#ef4444' : 
                                   intensity > 0.4 ? '#eab308' : '#22c55e'
                  }}
                  title={`Vector ${i}: ${(intensity * 100).toFixed(1)}%`}
                />
              );
            })}
          </div>
        </div>
        <div>
          <div className={`text-xs mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Vector Statistics
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span>Dimensions:</span>
              <span className="font-mono text-blue-500">384</span>
            </div>
            <div className="flex justify-between">
              <span>Similarity:</span>
              <span className="font-mono text-green-500">87.3%</span>
            </div>
            <div className="flex justify-between">
              <span>Confidence:</span>
              <span className="font-mono text-purple-500">92.1%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const MiniChartDashboard = () => (
    <div className={`p-4 rounded-lg border transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Price Chart Preview - {selectedSymbol.toUpperCase()} {selectedTimeframe}
        </h4>
        <span className={`text-xs px-2 py-1 rounded ${
          isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-700'
        }`}>
          Live Chart
        </span>
      </div>
      <div className="relative">
                  <div className={`h-32 rounded border-2 border-dashed flex items-center justify-center relative ${
            isDarkMode ? 'border-gray-600 bg-gray-800' : 'border-gray-300 bg-white'
          }`}>
            <div className="text-center">
              <div className="text-2xl mb-2">📈</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Interactive Chart Preview
              </div>
              <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                {rowLimit} candles • {selectedTimeframe} timeframe
              </div>
            </div>
            {/* Simple visual chart indicator */}
            <div className="absolute bottom-2 left-2 right-2 flex items-end space-x-1 opacity-30">
              {Array.from({length: 20}, (_, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-t ${Math.random() > 0.5 ? 'bg-green-500' : 'bg-red-500'}`}
                  style={{ height: `${8 + Math.random() * 20}px` }}
                />
              ))}
            </div>
          </div>
        <div className="grid grid-cols-4 gap-2 mt-3 text-xs">
          <div className={`p-2 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Current</div>
            <div className="font-bold text-blue-500">
              ${(Math.random() * 100 + (selectedSymbol === 'spy' ? 440 : 4400)).toFixed(2)}
            </div>
          </div>
          <div className={`p-2 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Change</div>
            <div className={`font-bold ${Math.random() > 0.5 ? 'text-green-500' : 'text-red-500'}`}>
              {Math.random() > 0.5 ? '+' : '-'}${(Math.random() * 10).toFixed(2)}
            </div>
          </div>
          <div className={`p-2 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Volume</div>
            <div className="font-bold text-purple-500">
              {(Math.random() * 1000000).toFixed(0)}
            </div>
          </div>
          <div className={`p-2 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Trend</div>
            <div className={`font-bold ${Math.random() > 0.5 ? 'text-green-500' : 'text-red-500'}`}>
              {Math.random() > 0.5 ? '🐂 Bull' : '🐻 Bear'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderMiniDashboard = () => {
    switch (miniDashboardView) {
      case 'data':
        return <MiniDataDashboard />;
      case 'vector':
        return <MiniVectorDashboard />;
      case 'chart':
        return <MiniChartDashboard />;
      default:
        return <MiniDataDashboard />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
          isDarkMode ? 'border-purple-400' : 'border-purple-600'
        }`}></div>
        <span className={`ml-3 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-600'
        }`}>AI analyzing market data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-lg p-4 border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-red-900/20 border-red-800' 
          : 'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center">
          <div className="text-red-500 text-sm font-medium">Error loading LLM analysis</div>
        </div>
        <div className="text-red-500 text-sm mt-1">{error}</div>
        <button 
          onClick={handleRefresh}
          className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
        >
          Retry Analysis
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
              {/* LLM Dashboard Header */}
        <div className={`p-6 rounded-lg transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-gradient-to-r from-purple-800 via-indigo-700 to-blue-800 text-white' 
            : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
        }`}>
          <div className="flex items-center">
            <h2 className="text-3xl font-black tracking-tight mb-2" style={{ 
              fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
              letterSpacing: '-0.02em',
              textShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }}>
              Day<span className="text-orange-200">gent</span> <span className="text-xl font-semibold text-orange-100">AI Assistant</span>
            </h2>
          <InfoTooltip id="llm-dashboard" content={
            <div>
              <p className="font-semibold mb-2">🤖 AI Trading Intelligence</p>
              <p className="mb-2">Advanced LLM-powered market analysis and trading insights:</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>Market Sentiment:</strong> AI-powered sentiment analysis from multiple sources</li>
                <li><strong>Trading Signals:</strong> Automated buy/sell recommendations with risk management</li>
                <li><strong>Pattern Recognition:</strong> Advanced technical pattern identification</li>
                <li><strong>Risk Assessment:</strong> Comprehensive market risk evaluation</li>
                <li><strong>Natural Language:</strong> Human-readable market explanations and insights</li>
              </ul>
              <p className="mt-2 text-xs">Powered by state-of-the-art language models for institutional-grade analysis.</p>
            </div>
          } isDarkMode={isDarkMode} />
        </div>
                  <p className={isDarkMode ? 'text-purple-200' : 'text-purple-100'}>
            Professional AI-powered trading intelligence • Natural language market analysis
          </p>
        <div className={`mt-4 text-sm rounded p-2 ${
          isDarkMode ? 'bg-black/30' : 'bg-white/20'
        }`}>
          🤖 {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • Model: {modelType.toUpperCase()} • Analyzing {rowLimit} periods
        </div>
      </div>

            {/* Data Selection & Controls */}
      <DataSelectionControls 
        handleRefresh={handleRefresh}
        isDarkMode={isDarkMode}
        dashboardType="llm"
      />

      {/* Selected Candles Panel */}
      <SelectedCandlesPanel 
        isDarkMode={isDarkMode} 
        canSelectCandles={false}
      />

      {/* AI Model Configuration */}
      <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>AI Model Configuration</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Analysis Mode
            </label>
            <select
              value={analysisMode}
              onChange={(e) => setAnalysisMode(e.target.value)}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                isDarkMode 
                  ? 'border-gray-600 bg-gray-700 text-white' 
                  : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              <option value="comprehensive">🔍 Comprehensive Analysis</option>
              <option value="signals">📊 Trading Signals Only</option>
              <option value="sentiment">💭 Sentiment Analysis</option>
              <option value="risk">⚠️ Risk Assessment</option>
            </select>
          </div>

          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              AI Model
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                isDarkMode 
                  ? 'border-gray-600 bg-gray-700 text-white' 
                  : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              <option value="gpt4">🧠 GPT-4 (OpenAI)</option>
              <option value="claude">🎯 Claude (Anthropic)</option>
              <option value="local">🏠 Local Model</option>
              <option value="ensemble">🤝 Ensemble (Multiple Models)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Mini Dashboard View */}
      <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Mini Dashboard View</h3>
            <InfoTooltip id="mini-dashboard" content={
              <div>
                <p className="font-semibold mb-2">📊 Mini Dashboard View</p>
                <p className="mb-2">Contextual data display for better AI conversations:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Data View:</strong> Recent OHLC trading data and price movements</li>
                  <li><strong>Vector View:</strong> Pattern analysis and similarity heatmaps</li>
                  <li><strong>Chart View:</strong> Price chart preview with key metrics</li>
                  <li><strong>Real-time:</strong> Updates with your symbol and timeframe selection</li>
                </ul>
                <p className="mt-2 text-xs">This provides context for your AI chat conversations below.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <select
            value={miniDashboardView}
            onChange={(e) => setMiniDashboardView(e.target.value)}
            className={`px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="data">📊 Data Dashboard</option>
            <option value="vector">🧠 Vector Dashboard</option>
            <option value="chart">📈 Chart Dashboard</option>
          </select>
        </div>

        {renderMiniDashboard()}
      </div>

      {/* LLM Chat Interface */}
      <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>AI Trading Assistant</h3>
            <InfoTooltip id="llm-chat" content={
              <div>
                <p className="font-semibold mb-2">🤖 AI Trading Assistant</p>
                <p className="mb-2">Intelligent conversation about your trading data:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Context Aware:</strong> AI can see the mini dashboard above</li>
                  <li><strong>Market Analysis:</strong> Ask about patterns, trends, and signals</li>
                  <li><strong>Educational:</strong> Learn about trading concepts and strategies</li>
                  <li><strong>Real-time:</strong> Analysis updates with your data selection</li>
                </ul>
                <p className="mt-2 text-xs">Try asking: "What do you see in the current market data?"</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="flex items-center space-x-2">
            <span className={`text-xs px-2 py-1 rounded ${
              isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-700'
            }`}>
              🟢 AI Online
            </span>
            <span className={`text-xs px-2 py-1 rounded ${
              isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
            }`}>
              Model: {modelType.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Chat Messages */}
        <div className={`h-64 overflow-y-auto rounded-lg border p-4 mb-4 transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-900 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="space-y-4">
            {chatMessages.map((message, index) => (
              <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-lg p-3 ${
                  message.role === 'user'
                    ? isDarkMode 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-blue-500 text-white'
                    : isDarkMode
                      ? 'bg-gray-700 text-gray-200 border border-gray-600'
                      : 'bg-white text-gray-800 border border-gray-200'
                }`}>
                  <div className="flex items-start space-x-2">
                    <span className="text-lg">
                      {message.role === 'user' ? '👤' : '🤖'}
                    </span>
                    <div className="flex-1">
                      <div className="text-sm">{message.content}</div>
                      <div className={`text-xs mt-1 opacity-70`}>
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Chat Input */}
        <form onSubmit={handleChatSubmit} className="flex space-x-2">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder={`Ask about ${selectedSymbol.toUpperCase()} ${selectedTimeframe} analysis...`}
            className={`flex-1 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white placeholder-gray-400' 
                : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500'
            }`}
          />
          <button
            type="submit"
            disabled={!chatInput.trim()}
            className={`px-6 py-2 rounded-md font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
              isDarkMode 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            Send 🚀
          </button>
        </form>

        {/* Quick Action Buttons */}
        <div className="flex flex-wrap gap-2 mt-3">
          {[
            "What's the current market sentiment?",
            "Explain the trading signals",
            "Show me the key patterns",
            "What's the risk level?",
            "Compare with historical data"
          ].map((question, index) => (
            <button
              key={index}
              onClick={() => setChatInput(question)}
              className={`text-xs px-3 py-1 rounded-full transition-colors duration-200 ${
                isDarkMode 
                  ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* AI Analysis Results */}
      {llmData && (
        <>
          {/* Market Sentiment Analysis */}
          <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <h3 className={`text-lg font-semibold ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Market Sentiment Analysis</h3>
                <InfoTooltip id="market-sentiment" content={
                  <div>
                    <p className="font-semibold mb-2">💭 AI Market Sentiment</p>
                    <p className="mb-2">Advanced sentiment analysis using multiple data sources:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Price Action:</strong> Technical momentum and trend analysis</li>
                      <li><strong>Volume Patterns:</strong> Institutional flow and retail behavior</li>
                      <li><strong>News Sentiment:</strong> Real-time news and social media analysis</li>
                      <li><strong>Options Flow:</strong> Smart money positioning indicators</li>
                    </ul>
                    <p className="mt-2 text-xs">Confidence scores indicate the AI's certainty in its analysis.</p>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <div className={`text-sm px-3 py-1 rounded-full ${
                isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
              }`}>
                Updated: {new Date(llmData.timestamp).toLocaleTimeString()}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>Overall Sentiment</span>
                  <span className="text-2xl">{getSentimentEmoji(llmData.analysis.marketSentiment.overall)}</span>
                </div>
                <div className={`text-2xl font-bold ${
                  llmData.analysis.marketSentiment.overall.toLowerCase() === 'bullish' ? 'text-green-500' :
                  llmData.analysis.marketSentiment.overall.toLowerCase() === 'bearish' ? 'text-red-500' :
                  'text-yellow-500'
                }`}>
                  {llmData.analysis.marketSentiment.overall}
                </div>
                <div className={`text-sm mt-1 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Confidence: <span className={getConfidenceColor(llmData.analysis.marketSentiment.confidence)}>
                    {formatConfidence(llmData.analysis.marketSentiment.confidence)}
                  </span>
                </div>
              </div>

              <div className={`p-4 rounded-lg border transition-colors duration-200 col-span-2 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <h4 className={`text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>AI Reasoning</h4>
                <p className={`text-sm ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  {llmData.analysis.marketSentiment.reasoning}
                </p>
                <div className="mt-3">
                  <h5 className={`text-xs font-medium mb-1 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>Key Signals</h5>
                  <div className="flex flex-wrap gap-2">
                    {llmData.analysis.marketSentiment.signals.map((signal, index) => (
                      <span key={index} className={`text-xs px-2 py-1 rounded-full ${
                        isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
                      }`}>
                        {signal}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trading Signals */}
          <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className="flex items-center mb-4">
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>AI Trading Signals</h3>
              <InfoTooltip id="trading-signals" content={
                <div>
                  <p className="font-semibold mb-2">📊 AI Trading Signals</p>
                  <p className="mb-2">Automated trading recommendations based on comprehensive analysis:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>Entry/Exit Points:</strong> Optimal timing for position management</li>
                    <li><strong>Risk Management:</strong> Stop loss and take profit levels</li>
                    <li><strong>Position Sizing:</strong> Recommended allocation based on risk</li>
                    <li><strong>Confidence Scoring:</strong> AI certainty in signal quality</li>
                  </ul>
                  <p className="mt-2 text-xs">Always validate signals with your own analysis and risk tolerance.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>Signal</span>
                  <span className="text-2xl">{getSignalEmoji(llmData.analysis.tradingSignals.primary)}</span>
                </div>
                <div className={`text-2xl font-bold ${
                  llmData.analysis.tradingSignals.primary.toLowerCase() === 'buy' ? 'text-green-500' :
                  llmData.analysis.tradingSignals.primary.toLowerCase() === 'sell' ? 'text-red-500' :
                  'text-yellow-500'
                }`}>
                  {llmData.analysis.tradingSignals.primary}
                </div>
                <div className={`text-sm mt-1 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Strength: {llmData.analysis.tradingSignals.strength}
                </div>
              </div>

              <div className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className={`text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Entry Price</div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>
                  ${llmData.analysis.tradingSignals.entryPrice.toFixed(2)}
                </div>
              </div>

              <div className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className={`text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Stop Loss</div>
                <div className="text-2xl font-bold text-red-500">
                  ${llmData.analysis.tradingSignals.stopLoss.toFixed(2)}
                </div>
              </div>

              <div className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className={`text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Risk/Reward</div>
                <div className="text-2xl font-bold text-blue-500">
                  {llmData.analysis.tradingSignals.riskReward.toFixed(1)}:1
                </div>
              </div>
            </div>

            {/* Price Targets */}
            <div className="mt-4">
              <h4 className={`text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>Price Targets</h4>
              <div className="grid grid-cols-3 gap-4">
                {llmData.analysis.tradingSignals.targets.map((target, index) => (
                  <div key={index} className={`p-3 rounded-lg border transition-colors duration-200 ${
                    isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <div className={`text-xs font-medium ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-600'
                    }`}>Target {index + 1}</div>
                    <div className="text-lg font-bold text-green-500">
                      ${target.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Pattern Recognition & AI Insights */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Pattern Recognition */}
            <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="flex items-center mb-4">
                <h3 className={`text-lg font-semibold ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Pattern Recognition</h3>
                <InfoTooltip id="pattern-recognition" content={
                  <div>
                    <p className="font-semibold mb-2">🔍 AI Pattern Recognition</p>
                    <p className="mb-2">Advanced technical pattern identification:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Classical Patterns:</strong> Triangles, flags, head & shoulders</li>
                      <li><strong>Custom Patterns:</strong> Proprietary AI-discovered formations</li>
                      <li><strong>Multi-Timeframe:</strong> Pattern analysis across different periods</li>
                      <li><strong>Confidence Scoring:</strong> Statistical reliability of each pattern</li>
                    </ul>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>

              <div className="space-y-3">
                {llmData.analysis.patternRecognition.map((pattern, index) => (
                  <div key={index} className={`p-3 rounded-lg border transition-colors duration-200 ${
                    isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className={`font-medium ${
                        isDarkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {pattern.pattern}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
                      }`}>
                        {pattern.timeframe}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className={`text-sm ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-600'
                      }`}>
                        Confidence: <span className={getConfidenceColor(pattern.confidence)}>
                          {formatConfidence(pattern.confidence)}
                        </span>
                      </span>
                      <div className={`w-16 bg-gray-200 rounded-full h-2 ${
                        isDarkMode ? 'bg-gray-600' : 'bg-gray-200'
                      }`}>
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${pattern.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Assessment */}
            <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="flex items-center mb-4">
                <h3 className={`text-lg font-semibold ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Risk Assessment</h3>
                <InfoTooltip id="risk-assessment" content={
                  <div>
                    <p className="font-semibold mb-2">⚠️ AI Risk Assessment</p>
                    <p className="mb-2">Comprehensive market risk evaluation:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Volatility Analysis:</strong> Current vs historical volatility</li>
                      <li><strong>Market Structure:</strong> Support/resistance levels</li>
                      <li><strong>Economic Events:</strong> Scheduled news and announcements</li>
                      <li><strong>Position Sizing:</strong> Recommended allocation guidelines</li>
                    </ul>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>

              <div className={`p-4 rounded-lg border mb-4 transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>Risk Level</span>
                  <span className={`text-xl ${
                    llmData.analysis.riskAssessment.level.toLowerCase() === 'low' ? 'text-green-500' :
                    llmData.analysis.riskAssessment.level.toLowerCase() === 'medium' ? 'text-yellow-500' :
                    'text-red-500'
                  }`}>
                    {llmData.analysis.riskAssessment.level.toLowerCase() === 'low' ? '🟢' :
                     llmData.analysis.riskAssessment.level.toLowerCase() === 'medium' ? '🟡' : '🔴'}
                  </span>
                </div>
                <div className={`text-2xl font-bold ${
                  llmData.analysis.riskAssessment.level.toLowerCase() === 'low' ? 'text-green-500' :
                  llmData.analysis.riskAssessment.level.toLowerCase() === 'medium' ? 'text-yellow-500' :
                  'text-red-500'
                }`}>
                  {llmData.analysis.riskAssessment.level}
                </div>
              </div>

              <div className="space-y-2 mb-4">
                <h4 className={`text-sm font-medium ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Risk Factors</h4>
                {llmData.analysis.riskAssessment.factors.map((factor, index) => (
                  <div key={index} className={`text-sm ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    • {factor}
                  </div>
                ))}
              </div>

              <div className={`p-3 rounded-lg ${
                isDarkMode ? 'bg-blue-900/20' : 'bg-blue-50'
              }`}>
                <div className={`text-sm font-medium mb-1 ${
                  isDarkMode ? 'text-blue-300' : 'text-blue-700'
                }`}>
                  Recommendation
                </div>
                <div className={`text-sm ${
                  isDarkMode ? 'text-blue-200' : 'text-blue-600'
                }`}>
                  {llmData.analysis.riskAssessment.recommendation}
                </div>
              </div>
            </div>
          </div>

          {/* AI Insights */}
          <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className="flex items-center mb-4">
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>AI Market Insights</h3>
              <InfoTooltip id="ai-insights" content={
                <div>
                  <p className="font-semibold mb-2">💡 AI Generated Insights</p>
                  <p className="mb-2">Natural language market analysis and observations:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>Market Context:</strong> Current conditions in plain English</li>
                    <li><strong>Historical Parallels:</strong> Similar past market situations</li>
                    <li><strong>Forward Looking:</strong> Potential future scenarios</li>
                    <li><strong>Educational:</strong> Learning opportunities and explanations</li>
                  </ul>
                </div>
              } isDarkMode={isDarkMode} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {llmData.analysis.aiInsights.map((insight, index) => (
                <div key={index} className={`p-4 rounded-lg border transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-start">
                    <span className="text-2xl mr-3">💡</span>
                    <div className={`text-sm ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-700'
                    }`}>
                      {insight}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Model Performance & Settings */}
          <details className={`rounded-lg border transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          }`}>
            <summary className={`p-6 cursor-pointer font-semibold transition-colors duration-200 ${
              isDarkMode 
                ? 'text-gray-300 hover:bg-gray-700' 
                : 'text-gray-700 hover:bg-gray-50'
            }`}>
              🤖 AI Model Performance & Configuration (Click to expand)
            </summary>
            <div className="px-6 pb-6">
              <div className={`rounded p-4 transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
              }`}>
                <h4 className={`font-semibold mb-3 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  🚀 Model Performance Metrics:
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Accuracy (7 days)</div>
                    <div className="text-xl font-bold text-green-500">87.3%</div>
                  </div>
                  <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Response Time</div>
                    <div className="text-xl font-bold text-blue-500">1.2s</div>
                  </div>
                  <div className={`p-3 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Confidence Score</div>
                    <div className="text-xl font-bold text-purple-500">92.1%</div>
                  </div>
                </div>
                
                <h4 className={`font-semibold mt-4 mb-3 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  💡 Backend Integration Status:
                </h4>
                <ul className={`space-y-1 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  <li>✅ <strong>Frontend Interface:</strong> Fully implemented with mock data</li>
                  <li>🟡 <strong>API Endpoints:</strong> Ready for backend integration</li>
                  <li>🟡 <strong>LLM Integration:</strong> Waiting for model deployment</li>
                  <li>🟡 <strong>Real-time Data:</strong> Currently using simulated analysis</li>
                  <li>✅ <strong>Multi-model Support:</strong> GPT-4, Claude, Local models ready</li>
                  <li>✅ <strong>Synchronous Controls:</strong> Integrated with other dashboards</li>
                </ul>
              </div>
            </div>
          </details>
        </>
      )}
    </div>
  );
};

export default TradingLLMDashboard; 