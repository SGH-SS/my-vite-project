import { useState, useEffect, useRef } from 'react';
import { useTrading } from '../context/TradingContext';

// Advanced Charts Component Template
// NOTE: Requires TradingView Advanced Charts library access
// 1. Request access: https://www.tradingview.com/charting-library/
// 2. Download library files to public/charting_library/
// 3. Add script tags to index.html

const AdvancedChart = ({ 
  isDarkMode, 
  chartData,
  onCandleSelect 
}) => {
  // Use shared trading context (same as lightweight charts)
  const {
    selectedSymbol,
    selectedTimeframe,
    selectedCandles,
    addSelectedCandle,
    removeSelectedCandle
  } = useTrading();

  const chartContainerRef = useRef(null);
  const tvWidget = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Custom datafeed adapter to use your existing API data
  const createDatafeed = () => ({
    onReady: (callback) => {
      console.log('📊 Advanced Charts Datafeed Ready');
      setTimeout(() => {
        callback({
          supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D'],
          supports_marks: true,
          supports_timescale_marks: true,
          supports_search: true,
          supports_group_request: false,
        });
      }, 0);
    },

    searchSymbols: (userInput, exchange, symbolType, onResultReadyCallback) => {
      // Use your existing symbols
      const symbols = [
        {
          symbol: 'ES',
          full_name: 'E-mini S&P 500 Futures',
          description: 'ES',
          exchange: 'CME',
          ticker: 'ES',
          type: 'futures'
        },
        {
          symbol: 'EURUSD',
          full_name: 'Euro vs US Dollar',
          description: 'EURUSD',
          exchange: 'FX',
          ticker: 'EURUSD',
          type: 'forex'
        },
        {
          symbol: 'SPY',
          full_name: 'SPDR S&P 500 ETF Trust',
          description: 'SPY',
          exchange: 'NYSE',
          ticker: 'SPY',
          type: 'stock'
        }
      ];

      const filteredSymbols = symbols.filter(s => 
        s.symbol.toLowerCase().includes(userInput.toLowerCase())
      );
      
      onResultReadyCallback(filteredSymbols);
    },

    resolveSymbol: (symbolName, onSymbolResolvedCallback, onResolveErrorCallback) => {
      console.log('🔍 Resolving symbol:', symbolName);
      
      const symbolInfo = {
        name: symbolName,
        ticker: symbolName,
        description: symbolName,
        type: 'stock',
        session: '24x7',
        timezone: 'Etc/UTC',
        exchange: '',
        minmov: 1,
        pricescale: 10000,
        has_intraday: true,
        has_no_volume: false,
        has_weekly_and_monthly: true,
        supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D'],
        volume_precision: 0,
        data_status: 'streaming',
      };

      setTimeout(() => {
        onSymbolResolvedCallback(symbolInfo);
      }, 0);
    },

    getBars: (symbolInfo, resolution, from, to, onHistoryCallback, onErrorCallback, firstDataRequest) => {
      console.log('📊 Getting bars:', symbolInfo.name, resolution, new Date(from * 1000), new Date(to * 1000));
      
      if (!chartData?.data || chartData.data.length === 0) {
        onHistoryCallback([], { noData: true });
        return;
      }

      try {
        // Convert your API data format to TradingView format
        const bars = chartData.data.map(candle => ({
          time: Math.floor(new Date(candle.timestamp).getTime() / 1000) * 1000, // Convert to seconds
          open: parseFloat(candle.open),
          high: parseFloat(candle.high),
          low: parseFloat(candle.low),
          close: parseFloat(candle.close),
          volume: candle.volume || 0
        }));

        // Filter bars by time range
        const filteredBars = bars.filter(bar => {
          const barTime = bar.time / 1000;
          return barTime >= from && barTime <= to;
        });

        // Sort by time (ascending)
        filteredBars.sort((a, b) => a.time - b.time);

        console.log('📈 Returning', filteredBars.length, 'bars');
        onHistoryCallback(filteredBars, { noData: filteredBars.length === 0 });
      } catch (error) {
        console.error('❌ Error in getBars:', error);
        onErrorCallback(error);
      }
    },

    subscribeBars: (symbolInfo, resolution, onRealtimeCallback, subscribeUID, onResetCacheNeededCallback) => {
      console.log('📡 Subscribing to real-time data:', symbolInfo.name, resolution);
      // Implement real-time data subscription using your existing WebSocket/API
      // When new data arrives, call: onRealtimeCallback(newBar);
    },

    unsubscribeBars: (subscribeUID) => {
      console.log('📡 Unsubscribing from real-time data:', subscribeUID);
      // Clean up real-time subscription
    }
  });

  // Initialize Advanced Charts widget
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    // Check if TradingView library is loaded
    if (!window.TradingView || !window.TradingView.widget) {
      setError('TradingView Advanced Charts library not loaded. Please add the script tags to your HTML.');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Clean up existing widget
      if (tvWidget.current) {
        tvWidget.current.remove();
        tvWidget.current = null;
      }

      // Create new TradingView widget
      tvWidget.current = new window.TradingView.widget({
        container: chartContainerRef.current,
        locale: 'en',
        library_path: '/charting_library/',
        
        // Data connection
        datafeed: createDatafeed(),
        symbol: selectedSymbol || 'ES',
        interval: selectedTimeframe || '5',
        
        // Theme and styling to match your current design
        theme: isDarkMode ? 'Dark' : 'Light',
        
        // Enabled features (professional trading tools)
        enabled_features: [
          'study_templates',
          'compare_symbol',
          'left_toolbar',
          'header_symbol_search',
          'header_interval_dialog_button',
          'show_interval_dialog_on_key_press',
          'header_chart_type',
          'header_settings',
          'header_indicators',
          'header_compare',
          'header_undo_redo',
          'header_screenshot',
          'header_fullscreen_button',
          'use_localstorage_for_settings',
          'volume_force_overlay',
          'left_toolbar',
          'control_bar',
          'timeframes_toolbar'
        ],

        // Disabled features (customize as needed)
        disabled_features: [
          'popup_hints',
          'header_saveload'
        ],

        // Color scheme matching your current charts
        overrides: {
          // Candlestick colors
          "mainSeriesProperties.candleStyle.upColor": "#22c55e",
          "mainSeriesProperties.candleStyle.downColor": "#ef4444",
          "mainSeriesProperties.candleStyle.borderUpColor": "#22c55e",
          "mainSeriesProperties.candleStyle.borderDownColor": "#ef4444",
          "mainSeriesProperties.candleStyle.wickUpColor": "#22c55e",
          "mainSeriesProperties.candleStyle.wickDownColor": "#ef4444",
          
          // Background colors
          "paneProperties.background": isDarkMode ? "#1f2937" : "#ffffff",
          "paneProperties.vertGridProperties.color": isDarkMode ? "#374151" : "#f3f4f6",
          "paneProperties.horzGridProperties.color": isDarkMode ? "#374151" : "#f3f4f6",
          
          // Volume colors
          "volume.volume.color.0": "#ef4444",
          "volume.volume.color.1": "#22c55e",
        },

        // Studies (technical indicators) - automatically add some basic ones
        studies_overrides: {
          "volume.volume.color.0": "#ef444480",
          "volume.volume.color.1": "#22c55e80",
        },

        // Auto-save settings
        auto_save_delay: 5,

        // Widget dimensions
        fullscreen: false,
        autosize: true,
        
        // Debugging
        debug: false,

        // Custom CSS
        custom_css_url: '/charting_library/custom.css',

        // Callback functions
        onChartReady: () => {
          console.log('🎉 Advanced Charts Ready!');
          setLoading(false);
          
          // Access the chart widget for custom functionality
          const widget = tvWidget.current;
          
          // Example: Add custom button
          widget.headerReady().then(() => {
            // widget.createButton()
            //   .attr('title', 'Custom Action')
            //   .text('Custom')
            //   .on('click', () => {
            //     console.log('Custom button clicked');
            //   });
          });

          // Example: Listen to symbol changes
          widget.subscribe('onSymbolChanged', (symbolData) => {
            console.log('📊 Symbol changed:', symbolData);
            // Update your context if needed
          });

          // Example: Listen to interval changes  
          widget.subscribe('onIntervalChanged', (interval) => {
            console.log('⏱️ Interval changed:', interval);
            // Update your context if needed
          });
        }
      });

    } catch (error) {
      console.error('❌ Error initializing Advanced Charts:', error);
      setError(`Failed to initialize Advanced Charts: ${error.message}`);
      setLoading(false);
    }

    // Cleanup on unmount
    return () => {
      if (tvWidget.current) {
        tvWidget.current.remove();
        tvWidget.current = null;
      }
    };
  }, [selectedSymbol, selectedTimeframe, isDarkMode, chartData]);

  // Loading state
  if (loading) {
    return (
      <div className={`flex items-center justify-center h-96 rounded border transition-colors duration-200 ${
        isDarkMode 
          ? 'border-gray-600 bg-gray-900' 
          : 'border-gray-200 bg-gray-50'
      }`}>
        <div className="text-center">
          <div className={`animate-spin rounded-full h-12 w-12 border-b-2 mx-auto mb-4 ${
            isDarkMode ? 'border-blue-400' : 'border-blue-600'
          }`}></div>
          <div className={`text-lg font-semibold mb-2 ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            Loading Advanced Charts
          </div>
          <div className={`text-sm ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Initializing TradingView widget...
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`rounded-lg p-6 border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-red-900/20 border-red-800' 
          : 'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center mb-4">
          <div className="text-red-500 text-xl mr-3">⚠️</div>
          <div className="text-red-500 font-medium">Advanced Charts Error</div>
        </div>
        <div className="text-red-500 text-sm mb-4">{error}</div>
        
        <div className={`p-4 rounded border ${
          isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'
        }`}>
          <div className={`text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Setup Instructions:
          </div>
          <ol className={`text-sm space-y-1 list-decimal list-inside ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            <li>Request access at: <code className="bg-gray-200 px-1 rounded">https://www.tradingview.com/charting-library/</code></li>
            <li>Download the library ZIP file</li>
            <li>Extract contents to <code className="bg-gray-200 px-1 rounded">public/charting_library/</code></li>
            <li>Add script tags to your HTML file</li>
            <li>Refresh this page</li>
          </ol>
        </div>
        
        <button 
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
        >
          Retry
        </button>
      </div>
    );
  }

  // Chart container
  return (
    <div className="space-y-4">
      {/* Advanced Charts Info Banner */}
      <div className={`p-4 rounded-lg border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-gradient-to-r from-green-800 to-blue-800 border-green-700 text-white' 
          : 'bg-gradient-to-r from-green-600 to-blue-600 border-green-500 text-white'
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold mb-1">🚀 TradingView Advanced Charts Active</h4>
            <p className="text-sm opacity-90">
              Professional trading tools • 100+ indicators • 70+ drawing tools • Enterprise features
            </p>
          </div>
          <div className="text-xs bg-white/20 px-2 py-1 rounded">
            {selectedSymbol} • {selectedTimeframe}
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div 
        ref={chartContainerRef}
        className={`w-full rounded border transition-colors duration-200 ${
          isDarkMode 
            ? 'border-gray-600 bg-gray-900' 
            : 'border-gray-200 bg-gray-50'
        }`}
        style={{ 
          height: '500px', 
          minHeight: '400px'
        }}
      />

      {/* Feature Comparison Note */}
      <details className={`rounded-lg border transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <summary className={`p-4 cursor-pointer font-semibold transition-colors duration-200 ${
          isDarkMode 
            ? 'text-gray-300 hover:bg-gray-700' 
            : 'text-gray-700 hover:bg-gray-50'
        }`}>
          📊 Advanced Charts vs Lightweight Charts Comparison
        </summary>
        <div className="px-4 pb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className={`p-4 rounded ${isDarkMode ? 'bg-blue-900/20' : 'bg-blue-50'}`}>
              <h5 className={`font-semibold mb-2 ${isDarkMode ? 'text-blue-300' : 'text-blue-700'}`}>
                🚀 Advanced Charts (Current)
              </h5>
              <ul className={`text-sm space-y-1 ${isDarkMode ? 'text-blue-200' : 'text-blue-600'}`}>
                <li>✅ 12+ chart types (Renko, P&F, Kagi)</li>
                <li>✅ 100+ technical indicators</li>
                <li>✅ 70+ drawing tools</li>
                <li>✅ Volume profile & market depth</li>
                <li>✅ Symbol comparison</li>
                <li>✅ Advanced time zones</li>
                <li>✅ Chart snapshots</li>
                <li>✅ Professional UI matching TradingView</li>
              </ul>
            </div>
            
            <div className={`p-4 rounded ${isDarkMode ? 'bg-gray-700/50' : 'bg-gray-100'}`}>
              <h5 className={`font-semibold mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                📊 Lightweight Charts (Previous)
              </h5>
              <ul className={`text-sm space-y-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                <li>✅ 3 chart types (candlestick, line, area)</li>
                <li>✅ Lightweight (~200KB)</li>
                <li>✅ Fast performance</li>
                <li>✅ Custom selection system (your code)</li>
                <li>✅ Basic theme support</li>
                <li>⚠️ No built-in indicators</li>
                <li>⚠️ No drawing tools</li>
                <li>⚠️ Limited customization</li>
              </ul>
            </div>
          </div>
        </div>
      </details>
    </div>
  );
};

export default AdvancedChart; 