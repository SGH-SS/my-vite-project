import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { createChart, CandlestickSeries, ColorType } from 'lightweight-charts';
import InfoTooltip from './shared/InfoTooltip.jsx';
import { useTrading } from '../context/TradingContext';
import { useDateRanges } from '../hooks/useDateRanges';
import { API_BASE_URL, FETCH_MODES, DATE_RANGE_TYPES } from '../utils/constants';

// Backtest Settings panel
const BacktestSettings = ({
  isDarkMode,
  fullData,
  currentIndex,
  setCurrentIndex,
  isPlaying,
  setIsPlaying,
  speedMs,
  setSpeedMs,
  stepSize,
  setStepSize,
  loopPlayback,
  setLoopPlayback
}) => {
  const total = fullData?.length || 0;
  const current = Math.min(Math.max(currentIndex, 0), Math.max(total - 1, 0));
  const currentCandle = total > 0 ? fullData[current] : null;

  const formatTs = (ts) => (ts ? new Date(ts).toLocaleString() : 'N/A');

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            Backtest Settings
          </h3>
          <InfoTooltip
            id="backtest-settings"
            isDarkMode={isDarkMode}
            content={(
              <div>
                <p className="font-semibold mb-2">🧪 Backtest Playback</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li><strong>Happened:</strong> Candles up to the current index are rendered.</li>
                  <li><strong>Coming:</strong> Remaining candles are hidden until playback advances.</li>
                  <li><strong>Speed:</strong> Controls delay between candles during autoplay.</li>
                  <li><strong>Step:</strong> Move forward/back by N candles.</li>
                </ul>
              </div>
            )}
          />
        </div>
        <div className="text-xs opacity-75">
          {total > 0 ? (
            <span>
              Loaded {total} candles • Range {formatTs(fullData[0]?.timestamp)} → {formatTs(fullData[total - 1]?.timestamp)}
            </span>
          ) : (
            <span>No data loaded</span>
          )}
        </div>
      </div>

      {/* Playback controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Position / progress */}
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            Progress
          </label>
          <input
            type="range"
            min={0}
            max={Math.max(total - 1, 0)}
            value={current}
            onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
            className="w-full"
          />
          <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {current + 1} / {total} candles happened
            {currentCandle && (
              <span className="ml-2">
                • Current: {formatTs(currentCandle.timestamp)}
              </span>
            )}
          </div>
        </div>

        {/* Speed / step */}
        <div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Speed</label>
              <select
                value={speedMs}
                onChange={(e) => setSpeedMs(parseInt(e.target.value))}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={1500}>0.5x (1500ms)</option>
                <option value={1000}>1x (1000ms)</option>
                <option value={500}>2x (500ms)</option>
                <option value={250}>4x (250ms)</option>
                <option value={100}>10x (100ms)</option>
              </select>
            </div>
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Step Size</label>
              <select
                value={stepSize}
                onChange={(e) => setStepSize(parseInt(e.target.value))}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={1}>1 candle</option>
                <option value={5}>5 candles</option>
                <option value={10}>10 candles</option>
                <option value={25}>25 candles</option>
              </select>
            </div>
          </div>
        </div>

        {/* Buttons / toggles */}
        <div className="flex items-end gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              isDarkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => setCurrentIndex(Math.max(current - stepSize, 0))}
            className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
          >
            ◀ Step Back
          </button>
          <button
            onClick={() => setCurrentIndex(Math.min(current + stepSize, Math.max(total - 1, 0)))}
            className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
          >
            Step Fwd ▶
          </button>
          <label className="ml-auto inline-flex items-center gap-2 text-sm">
            <input type="checkbox" checked={loopPlayback} onChange={(e) => setLoopPlayback(e.target.checked)} />
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Loop</span>
          </label>
        </div>
      </div>
    </div>
  );
};

// Compact, visually clear prediction card displayed above the chart
const CurrentPredictionCard = ({ isDarkMode, prediction, modelName }) => {
  const direction = prediction?.pred_label === 1 ? 'UP' : 'DOWN';
  const probPct = ((prediction?.pred_prob_up || 0) * 100).toFixed(1);
  const threshold = (prediction?.threshold_used || 0).toFixed(2);
  const marginPct = (Math.abs(prediction?.decision_margin || 0) * 100).toFixed(1);
  const truth = prediction ? (prediction.true_label === 1 ? 'UP' : 'DOWN') : 'N/A';
  const isCorrect = !!prediction?.correct;

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Current Candle Prediction</h3>
          {modelName && (
            <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>Model: {modelName}</span>
          )}
        </div>
        {prediction && (
          <span className={`text-xs px-2 py-1 rounded ${
            isCorrect ? (isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-700') : (isDarkMode ? 'bg-red-900/30 text-red-300' : 'bg-red-100 text-red-700')
          }`}>{isCorrect ? 'Correct' : 'Incorrect'}</span>
        )}
      </div>

      {!prediction ? (
        <div className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>No prediction for this candle.</div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <div className={`col-span-2 md:col-span-2 flex items-center justify-center rounded-lg py-4 font-bold text-xl ${
            direction === 'UP' ? 'bg-green-600/10 text-green-400' : 'bg-red-600/10 text-red-400'
          }`}>
            {direction}
          </div>
          <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Probability (p_up)</div>
            <div className="font-semibold text-lg">{probPct}%</div>
          </div>
          <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Threshold</div>
            <div className="font-semibold text-lg">{threshold}</div>
          </div>
          <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Confidence</div>
            <div className="font-semibold text-lg">{marginPct}%</div>
          </div>
          <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Truth</div>
            <div className="font-semibold text-lg">{truth}</div>
          </div>
        </div>
      )}

      {prediction && (
        <div className={`mt-3 font-mono text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          pred={direction} p_up={(prediction.pred_prob_up || 0).toFixed(4)} thr={threshold} margin={(prediction.decision_margin || 0).toFixed(4)} truth={truth} → {isCorrect ? '✅ CORRECT' : '❌ WRONG'}
        </div>
      )}
    </div>
  );
};

// Live Model Prediction Box - uses real-time model inference API
const LivePredictionBox = ({
  isDarkMode,
  selectedSymbol,
  selectedTimeframe,
  currentCandle,
  currentIndex
}) => {
  const [livePrediction, setLivePrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Only support SPY 1D with GB1D model for now
  const isSupported = selectedSymbol === 'spy' && selectedTimeframe === '1d';
  
  // Training cutoff - only show predictions for test period (2024-12-17 and after)
  const TRAIN_CUTOFF = new Date('2024-12-16T23:59:59.999Z');
  const isTestPeriod = currentCandle && new Date(currentCandle.timestamp) > TRAIN_CUTOFF;
  
  const shouldShowPrediction = isSupported && isTestPeriod && currentCandle;

  // Fetch live prediction for current candle
  useEffect(() => {
    if (!shouldShowPrediction) {
      setLivePrediction(null);
      setError(null);
      return;
    }

    const fetchLivePrediction = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Format the current candle's date for API call
        const candleDate = new Date(currentCandle.timestamp);
        const dateStr = candleDate.toISOString().split('T')[0]; // YYYY-MM-DD format
        
        const response = await fetch(
          `${API_BASE_URL}/live-model-predictions/${selectedSymbol}/${selectedTimeframe}?start_date=${dateStr}&end_date=${dateStr}`
        );
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Find prediction that matches current candle timestamp
        const matchingPrediction = data.predictions?.find(pred => {
          const predTime = new Date(pred.timestamp).getTime();
          const candleTime = new Date(currentCandle.timestamp).getTime();
          return Math.abs(predTime - candleTime) <= 60000; // 1 minute tolerance
        });
        
        setLivePrediction(matchingPrediction || null);
        
      } catch (err) {
        console.error('Error fetching live prediction:', err);
        setError(err.message);
        setLivePrediction(null);
      } finally {
        setLoading(false);
      }
    };

    fetchLivePrediction();
  }, [shouldShowPrediction, currentCandle, selectedSymbol, selectedTimeframe]);

  if (!isSupported) {
    return (
      <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Live Model Predictions</h3>
          <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
            Real-time AI
          </span>
        </div>
        <div className={`rounded-md p-4 border ${isDarkMode ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' : 'bg-yellow-50 border-yellow-200 text-yellow-800'}`}>
          <div className="text-sm font-medium">Live Predictions Not Available</div>
          <div className="text-xs mt-1">
            Live model predictions are currently only available for SPY 1D timeframe.
            Current selection: {selectedSymbol?.toUpperCase()} {selectedTimeframe?.toUpperCase()}
          </div>
        </div>
      </div>
    );
  }

  if (!isTestPeriod && currentCandle) {
    return (
      <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Live Model Predictions</h3>
          <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
            Real-time AI
          </span>
        </div>
        <div className={`rounded-md p-4 border ${isDarkMode ? 'bg-blue-900/20 border-blue-800 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-800'}`}>
          <div className="text-sm font-medium">Training Period Candle</div>
          <div className="text-xs mt-1">
            This candle ({new Date(currentCandle.timestamp).toLocaleDateString()}) is from the training period. 
            Live predictions are only shown for test period candles (2024-12-17 and after).
          </div>
        </div>
      </div>
    );
  }

  if (!currentCandle) {
    return (
      <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Live Model Predictions</h3>
          <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
            Real-time AI
          </span>
        </div>
        <div className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>No candle selected for prediction.</div>
      </div>
    );
  }

  const direction = livePrediction?.prediction === 1 ? 'UP' : 'DOWN';
  const probPct = ((livePrediction?.probability || 0) * 100).toFixed(1);
  const confidencePct = ((livePrediction?.confidence || 0) * 100).toFixed(1);
  const threshold = 0.57; // GB1D threshold

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Live Model Predictions</h3>
          <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-700'}`}>
            Real-time AI
          </span>
        </div>
        <div className="text-right">
          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Model: Gradient Boosting 1D
          </div>
          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Candle: {currentIndex + 1} • {new Date(currentCandle.timestamp).toLocaleDateString()}
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`}></div>
          <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Running AI model...</span>
        </div>
      )}

      {error && (
        <div className={`rounded-md p-4 border ${isDarkMode ? 'bg-red-900/20 border-red-800 text-red-300' : 'bg-red-50 border-red-200 text-red-700'}`}>
          <div className="text-sm font-medium">Prediction Error</div>
          <div className="text-xs mt-1">{error}</div>
        </div>
      )}

      {!loading && !error && !livePrediction && (
        <div className={`rounded-md p-4 border ${isDarkMode ? 'bg-gray-900/20 border-gray-600 text-gray-300' : 'bg-gray-50 border-gray-200 text-gray-700'}`}>
          <div className="text-sm font-medium">No Prediction Available</div>
          <div className="text-xs mt-1">Unable to generate prediction for this candle.</div>
        </div>
      )}

      {!loading && !error && livePrediction && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
            <div className={`col-span-2 md:col-span-2 flex items-center justify-center rounded-lg py-6 font-bold text-2xl ${
              direction === 'UP' ? 'bg-green-600/10 text-green-400' : 'bg-red-600/10 text-red-400'
            }`}>
              🤖 {direction}
            </div>
            <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Probability</div>
              <div className="font-semibold text-lg">{probPct}%</div>
            </div>
            <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Confidence</div>
              <div className="font-semibold text-lg">{confidencePct}%</div>
            </div>
            <div className={`rounded-lg p-3 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Threshold</div>
              <div className="font-semibold text-lg">{threshold}</div>
            </div>
          </div>

          <div className={`font-mono text-xs p-3 rounded ${isDarkMode ? 'bg-gray-900 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
            🤖 Live: pred={direction} p_up={livePrediction.probability.toFixed(4)} thr={threshold} conf={livePrediction.confidence.toFixed(4)} • {new Date(livePrediction.timestamp).toLocaleString()}
          </div>

          <div className={`mt-3 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            ⚡ This prediction was generated in real-time using the live GB1D model inference engine.
          </div>
        </>
      )}
    </div>
  );
};

// Enhanced Model prediction panel with DB (v1_models) integration + CSV fallback
const ModelPredictionPanel = ({
  isDarkMode,
  selectedSymbol,
  selectedTimeframe,
  currentIndex,
  setCurrentIndex,
  fullData,
  onCurrentPredictionChange
}) => {
  const [selectedModel, setSelectedModel] = useState('gb1d'); // Default to GB 1D (normalized)
  const [availableModels, setAvailableModels] = useState([]);
  const [allPredictions, setAllPredictions] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model metadata and key normalization between DB tables (gb1d, gb4h, lgbm4h)
  // and CSV keys (gb_1d, gb_4h, lgbm_4h)
  const MODEL_META = {
    gb1d: { key: 'gb1d', name: 'Gradient Boosting 1D', threshold: 0.57 },
    gb4h: { key: 'gb4h', name: 'Gradient Boosting 4H', threshold: 0.50 },
    lgbm4h: { key: 'lgbm4h', name: 'LightGBM Financial 4H', threshold: 0.30 },
  };
  const normalizeKey = (k) => (k ? k.replace(/_/g, '').toLowerCase() : '');

  const canPredict = selectedSymbol === 'spy' && (selectedTimeframe === '1d' || selectedTimeframe === '4h');
  const currentCandle = fullData && fullData.length > 0 && currentIndex >= 0 ? fullData[currentIndex] : null;

  // Helper to build available models array from DB response
  const buildAvailableFromDB = (resp) => {
    if (!resp || !resp.models) return [];
    
    // Filter models based on current symbol and timeframe
    const validModels = [];
    if (selectedSymbol === 'spy' && selectedTimeframe === '1d') {
      // Only gb1d for SPY 1D
      if (resp.models.gb1d && (resp.models.gb1d?.count || 0) > 0) {
        validModels.push('gb1d');
      }
    } else if (selectedSymbol === 'spy' && selectedTimeframe === '4h') {
      // gb4h and lgbm4h for SPY 4H  
      if (resp.models.gb4h && (resp.models.gb4h?.count || 0) > 0) {
        validModels.push('gb4h');
      }
      if (resp.models.lgbm4h && (resp.models.lgbm4h?.count || 0) > 0) {
        validModels.push('lgbm4h');
      }
    }
    
    return validModels.map((k) => {
      const nk = normalizeKey(k);
      const meta = MODEL_META[nk] || { key: nk, name: nk, threshold: null };
      return {
        key: nk,
        name: meta.name,
        threshold: meta.threshold,
        record_count: resp.models[k]?.count || 0,
        available: (resp.models[k]?.count || 0) > 0,
      };
    });
  };

  // Load predictions when symbol/timeframe changes (Database only)
  useEffect(() => {
    if (!canPredict) {
      setAllPredictions({});
      setError(null);
      return;
    }

    const loadPredictions = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Load model results from database
        const dbResp = await fetch(`${API_BASE_URL}/model-results/${selectedSymbol}/${selectedTimeframe}`);
        if (!dbResp.ok) {
          throw new Error(`Database error: HTTP ${dbResp.status}`);
        }
        
        const dbData = await dbResp.json();
        const byModel = {};
        Object.entries(dbData.models || {}).forEach(([table, payload]) => {
          const nk = normalizeKey(table);
          if (!byModel[nk]) byModel[nk] = {};
          (payload?.predictions || []).forEach((p) => {
            if (p?.timestamp) byModel[nk][p.timestamp] = p;
          });
        });
        
        setAllPredictions(byModel);
        const avail = buildAvailableFromDB(dbData);
        setAvailableModels(avail);
        
        if (avail.length > 0) {
          const prefer = selectedTimeframe === '1d' ? 'gb1d' : 'lgbm4h';
          const pick = avail.find(a => a.key === prefer) || avail[0];
          setSelectedModel(pick.key);
        }
      } catch (e) {
        console.error('Error loading predictions:', e);
        setError(`Failed to load predictions from database: ${e.message}`);
        setAllPredictions({});
        setAvailableModels([]);
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
  }, [selectedSymbol, selectedTimeframe, canPredict]);

  // Get current candle prediction
  const getCurrentPrediction = () => {
    if (!currentCandle || !allPredictions[selectedModel]) {
      return null;
    }

    // Try to match by timestamp - normalize both timestamps for comparison
    const candleTimestamp = currentCandle.timestamp;
    const predictions = allPredictions[selectedModel];
    
    // First try exact match
    if (predictions[candleTimestamp]) {
      return predictions[candleTimestamp];
    }
    
    // If no exact match, try to find by converting timestamps to Date objects and comparing
    const candleDate = new Date(candleTimestamp);
    const candleTimeMs = candleDate.getTime();
    
    // Look for predictions within a small time window (1 minute tolerance)
    for (const [predTimestamp, prediction] of Object.entries(predictions)) {
      const predDate = new Date(predTimestamp);
      const predTimeMs = predDate.getTime();
      const timeDiff = Math.abs(candleTimeMs - predTimeMs);
      
      // If within 1 minute, consider it a match
      if (timeDiff <= 60000) {
        return prediction;
      }
    }
    
    return null;
  };

  const currentPrediction = getCurrentPrediction();
  
  // Get compatible models for current symbol/timeframe
  const compatibleModels = availableModels.filter(m => m.available);

  // Format prediction string as requested
  const formatPredictionString = (prediction) => {
    if (!prediction) return "No prediction available for this candle";
    
    try {
      const pred_label = prediction.pred_label === 1 ? "UP" : "DOWN";
      const truth_label = prediction.true_label === 1 ? "UP" : "DOWN";
      const prob = (prediction.pred_prob_up || 0).toFixed(4);
      const threshold = (prediction.threshold_used || 0).toFixed(2);
      const margin = (prediction.decision_margin || 0).toFixed(4);
      const correct_symbol = prediction.correct ? "✅ CORRECT" : "❌ WRONG";
      
      return `pred=${pred_label} p_up=${prob} thr=${threshold} margin=${margin} truth=${truth_label} → ${correct_symbol}`;
    } catch (e) {
      console.error('Error formatting prediction:', e, prediction);
      return "Error formatting prediction";
    }
  };

  // Notify parent of the current prediction and model name for the standalone card
  useEffect(() => {
    if (!onCurrentPredictionChange) return;
    const modelName = availableModels.find(m => m.key === selectedModel)?.name || selectedModel;
    onCurrentPredictionChange({ prediction: currentPrediction, modelName });
  }, [currentPrediction, selectedModel, availableModels, onCurrentPredictionChange]);

  return (
    <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            Database Model Predictions
          </h3>
          <InfoTooltip
            id="database-model-predictions"
            isDarkMode={isDarkMode}
            content={(
              <div>
                <p className="font-semibold mb-2">🤖 Database Model Predictions</p>
                <p className="mb-2">Shows predictions from pre-trained models stored in PostgreSQL v1_models schema:</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li><strong>GB 1D:</strong> Gradient Boosting for SPY daily (threshold: 0.57)</li>
                  <li><strong>GB 4H:</strong> Gradient Boosting for SPY 4-hour (threshold: 0.50)</li>
                  <li><strong>LightGBM 4H:</strong> LightGBM Financial for SPY 4-hour (threshold: 0.30)</li>
                  <li><strong>Format:</strong> pred=UP/DOWN p_up=probability thr=threshold margin=decision_margin truth=actual → ✅/❌</li>
                </ul>
              </div>
            )}
          />
        </div>
        <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          {canPredict ? `SPY ${selectedTimeframe.toUpperCase()} • ${availableModels.length} models available` : 'Database predictions available for SPY 1D/4H only'}
        </span>
      </div>

      {!canPredict ? (
        <div className={`rounded-md p-4 border ${isDarkMode ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' : 'bg-yellow-50 border-yellow-200 text-yellow-800'}`}>
          <div className="text-sm font-medium">Model Predictions Not Available</div>
          <div className="text-xs mt-1">
            Database model predictions are only available for SPY symbol with 1d or 4h timeframes.
            Current selection: {selectedSymbol?.toUpperCase()} {selectedTimeframe?.toUpperCase()}
          </div>
        </div>
      ) : (
        <>
          {/* Model Selection */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className={`w-full rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                {availableModels.map(model => (
                  <option key={model.key} value={model.key}>
                    {model.name} (thr: {model.threshold})
                  </option>
                ))}
              </select>
              {availableModels.length === 0 && (
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                  No compatible models found for {selectedSymbol?.toUpperCase()} {selectedTimeframe?.toUpperCase()}
                </div>
              )}
            </div>
            
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Current Candle Info
              </label>
              <div className={`rounded-md p-3 text-sm ${isDarkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-900'}`}>
                {currentCandle ? (
                  <>
                    <div>Index: {currentIndex + 1} / {fullData.length}</div>
                    <div>Time: {new Date(currentCandle.timestamp).toLocaleString()}</div>
                    <div>Close: ${currentCandle.close?.toFixed(2)}</div>
                  </>
                ) : (
                  <div>No candle selected</div>
                )}
              </div>
            </div>
          </div>

          {/* Standalone prediction card is rendered above the chart; here we only manage selection/history */}
          {loading && (
            <div className={isDarkMode ? 'text-blue-300' : 'text-blue-700'}>Loading predictions...</div>
          )}
          {error && (
            <div className={isDarkMode ? 'text-red-300' : 'text-red-700'}>Error: {error}</div>
          )}

          {/* Model Statistics */}
          {availableModels.length > 0 && (
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
              {availableModels.map(model => {
                const modelPredictions = Object.values(allPredictions[model.key] || {});
                const correctCount = modelPredictions.filter(p => p.correct).length;
                const accuracy = modelPredictions.length > 0 ? (correctCount / modelPredictions.length * 100).toFixed(1) : 'N/A';
                
                return (
                  <div key={model.key} className={`rounded-md p-3 border ${
                    selectedModel === model.key 
                      ? isDarkMode ? 'border-blue-500 bg-blue-900/20' : 'border-blue-500 bg-blue-50'
                      : isDarkMode ? 'border-gray-600 bg-gray-700' : 'border-gray-200 bg-gray-50'
                  }`}>
                    <div className={`text-sm font-semibold mb-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      {model.name}
                    </div>
                    <div className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      <div>Threshold: {model.threshold}</div>
                      <div>Records: {model.record_count}</div>
                      <div>Accuracy: {accuracy}%</div>
                      {model.date_range && (
                        <div>Range: {new Date(model.date_range.start).toLocaleDateString()} - {new Date(model.date_range.end).toLocaleDateString()}</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Navigation through predictions */}
          {currentPrediction && (
            <div className="mt-4">
              <div className={`text-xs mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Prediction {currentPrediction.candle_index || currentIndex + 1} • Model: {availableModels.find(m => m.key === selectedModel)?.name}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const newIndex = Math.max(0, currentIndex - 1);
                    setCurrentIndex(newIndex);
                  }}
                  disabled={currentIndex <= 0}
                  className={`px-3 py-1 text-xs rounded ${
                    currentIndex <= 0
                      ? isDarkMode ? 'bg-gray-600 text-gray-500 cursor-not-allowed' : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : isDarkMode ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  ← Prev Candle
                </button>
                <button
                  onClick={() => {
                    const newIndex = Math.min(fullData.length - 1, currentIndex + 1);
                    setCurrentIndex(newIndex);
                  }}
                  disabled={currentIndex >= fullData.length - 1}
                  className={`px-3 py-1 text-xs rounded ${
                    currentIndex >= fullData.length - 1
                      ? isDarkMode ? 'bg-gray-600 text-gray-500 cursor-not-allowed' : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : isDarkMode ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Next Candle →
                </button>
              </div>
            </div>
          )}

          {/* Prediction History/Context (show nearby predictions) */}
          {allPredictions[selectedModel] && Object.keys(allPredictions[selectedModel]).length > 0 && (
            <details className="mt-4">
              <summary className={`cursor-pointer text-sm font-medium ${isDarkMode ? 'text-gray-300 hover:text-white' : 'text-gray-700 hover:text-gray-900'}`}>
                📊 View All Predictions for {availableModels.find(m => m.key === selectedModel)?.name} ({Object.keys(allPredictions[selectedModel]).length} total)
              </summary>
              <div className={`mt-3 max-h-64 overflow-y-auto rounded border ${isDarkMode ? 'border-gray-600 bg-gray-800' : 'border-gray-200 bg-gray-50'}`}>
                <div className="space-y-1 p-3">
                  {Object.values(allPredictions[selectedModel])
                    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
                    .map((pred, idx) => {
                      const isCurrentCandle = currentCandle && pred.timestamp === currentCandle.timestamp;
                      return (
                        <div 
                          key={idx}
                          className={`p-2 rounded text-xs font-mono cursor-pointer transition-colors ${
                            isCurrentCandle 
                              ? isDarkMode ? 'bg-blue-900/40 border-l-4 border-blue-400' : 'bg-blue-50 border-l-4 border-blue-500'
                              : isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-white hover:bg-gray-50'
                          }`}
                          onClick={() => {
                            // Find the index of this candle in fullData and navigate to it
                            const candleIndex = fullData.findIndex(c => c.timestamp === pred.timestamp);
                            if (candleIndex >= 0) {
                              setCurrentIndex(candleIndex);
                            }
                          }}
                          title="Click to navigate to this candle"
                        >
                          <div className={`font-semibold mb-1 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                            {new Date(pred.timestamp).toLocaleDateString()} {new Date(pred.timestamp).toLocaleTimeString()}
                          </div>
                          <div className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                            {formatPredictionString(pred)}
                          </div>
                        </div>
                      );
                    })
                  }
                </div>
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
};

// Data Selection & Controls (mirrors the one inside Chart.jsx)
const DataSelectionControls = ({ handleRefresh, isDarkMode }) => {
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    fetchMode,
    setFetchMode,
    dateRangeType,
    setDateRangeType,
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useTrading();

  const { dateRanges } = useDateRanges(selectedSymbol, selectedTimeframe);

  const [pendingStartDate, setPendingStartDate] = useState(startDate);
  const [pendingEndDate, setPendingEndDate] = useState(endDate);
  const [pendingDateRangeType, setPendingDateRangeType] = useState(dateRangeType);

  useEffect(() => {
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
    setPendingDateRangeType(dateRangeType);
  }, [startDate, endDate, dateRangeType]);

  useEffect(() => {
    if (!dateRanges) return;
    switch (pendingDateRangeType) {
      case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
        setPendingStartDate(dateRanges.earliest);
        break;
      case DATE_RANGE_TYPES.DATE_TO_LATEST:
        setPendingEndDate(dateRanges.latest);
        break;
      case DATE_RANGE_TYPES.DATE_TO_DATE:
        if (!pendingStartDate) setPendingStartDate(dateRanges.earliest);
        if (!pendingEndDate) setPendingEndDate(dateRanges.latest);
        break;
    }
  }, [pendingDateRangeType, dateRanges]);

  const formatDatePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(0, 10);
  };

  const formatTimePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(11, 16);
  };

  const combineDateAndTime = (dateValue, timeValue) => {
    if (!dateValue) return null;
    const timeToUse = timeValue || '00:00';
    return new Date(`${dateValue}T${timeToUse}:00.000Z`).toISOString();
  };

  const hasPendingChanges = () => (
    pendingDateRangeType !== dateRangeType ||
    pendingStartDate !== startDate ||
    pendingEndDate !== endDate
  );

  const applyDateChanges = () => {
    if (setDateRangeType) setDateRangeType(pendingDateRangeType);
    if (setStartDate) setStartDate(pendingStartDate);
    if (setEndDate) setEndDate(pendingEndDate);
  };

  const resetDateChanges = () => {
    setPendingDateRangeType(dateRangeType);
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
  };

  const isStartDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE;
  const isEndDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST;

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Data Selection & Controls
        </h3>
        <button
          onClick={handleRefresh}
          className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
            isDarkMode ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
          }`}
        >
          🔄 Refresh Chart
        </button>
      </div>

      {/* Top controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Symbol</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            <option value="es">ES (E-mini S&P 500)</option>
            <option value="eurusd">EURUSD</option>
            <option value="spy">SPY</option>
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Timeframe</label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
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
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Data Limit</label>
          <select
            value={rowLimit}
            onChange={(e) => setRowLimit(parseInt(e.target.value))}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            {[25, 50, 100, 250, 500, 1000, 2000].map((n) => (
              <option key={n} value={n}>{n} records</option>
            ))}
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Sort Order</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            <option value="desc">⬇ Descending (Newest first)</option>
            <option value="asc">⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>

      {/* Date range section */}
      <div className="mt-6">
        <div className="flex items-center gap-4 mb-3">
          <span className={isDarkMode ? 'text-gray-300 text-sm' : 'text-gray-700 text-sm'}>Fetch Mode:</span>
          <div className={`flex rounded-lg p-1 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
            <button
              onClick={() => setFetchMode(FETCH_MODES.LIMIT)}
              className={`px-3 py-1 text-sm rounded-md ${
                fetchMode === FETCH_MODES.LIMIT
                  ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                  : isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              📊 Record Limit
            </button>
            <button
              onClick={() => setFetchMode(FETCH_MODES.DATE_RANGE)}
              className={`ml-1 px-3 py-1 text-sm rounded-md ${
                fetchMode === FETCH_MODES.DATE_RANGE
                  ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                  : isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              📅 Date Range
            </button>
          </div>
        </div>

        {fetchMode === FETCH_MODES.DATE_RANGE && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Range Type</label>
              <select
                value={pendingDateRangeType}
                onChange={(e) => setPendingDateRangeType(e.target.value)}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={DATE_RANGE_TYPES.EARLIEST_TO_DATE}>Earliest → Date</option>
                <option value={DATE_RANGE_TYPES.DATE_TO_DATE}>Date → Date</option>
                <option value={DATE_RANGE_TYPES.DATE_TO_LATEST}>Date → Latest</option>
              </select>
            </div>

            {/* Start */}
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Start</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={formatDatePart(pendingStartDate)}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(e.target.value, formatTimePart(pendingStartDate)))}
                  disabled={isStartDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
                <input
                  type="time"
                  value={formatTimePart(pendingStartDate)}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(formatDatePart(pendingStartDate), e.target.value))}
                  disabled={isStartDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
              </div>
            </div>

            {/* End */}
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>End</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={formatDatePart(pendingEndDate)}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(e.target.value, formatTimePart(pendingEndDate)))}
                  disabled={isEndDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
                <input
                  type="time"
                  value={formatTimePart(pendingEndDate)}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(formatDatePart(pendingEndDate), e.target.value))}
                  disabled={isEndDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
              </div>
            </div>
          </div>
        )}

        {/* Apply/reset */}
        {fetchMode === FETCH_MODES.DATE_RANGE && (
          <div className="mt-3 flex gap-2">
            <button
              onClick={applyDateChanges}
              disabled={!hasPendingChanges()}
              className={`px-3 py-2 rounded-md text-sm ${
                hasPendingChanges()
                  ? isDarkMode ? 'bg-green-700 hover:bg-green-600 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
                  : isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-500'
              }`}
            >
              Apply Range
            </button>
            <button
              onClick={resetDateChanges}
              className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
            >
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// Main Backtest Dashboard component
const BacktestDashboard = ({ isDarkMode = false }) => {
  const {
    selectedSymbol,
    selectedTimeframe,
    rowLimit,
    sortOrder,
    fetchMode,
    dateRangeType,
    startDate,
    endDate
  } = useTrading();

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const priceSeriesRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null); // full payload from API
  const fullData = chartData?.data || [];

  // Backtest state
  const [currentIndex, setCurrentIndex] = useState(0); // inclusive index of latest happened candle
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(1000);
  const [stepSize, setStepSize] = useState(1);
  const [loopPlayback, setLoopPlayback] = useState(false);
  const [currentPredictionInfo, setCurrentPredictionInfo] = useState({ prediction: null, modelName: null });

  // Get current candle for live predictions
  const currentCandle = fullData && fullData.length > 0 && currentIndex >= 0 ? fullData[currentIndex] : null;

  // Build display data whenever currentIndex or fullData changes
  const visibleCandles = useMemo(() => {
    if (!fullData || fullData.length === 0) return [];
    const slice = fullData.slice(0, Math.min(currentIndex + 1, fullData.length));
    return slice.map(c => ({
      time: Math.floor(new Date(c.timestamp).getTime() / 1000),
      open: parseFloat(c.open),
      high: parseFloat(c.high),
      low: parseFloat(c.low),
      close: parseFloat(c.close)
    }));
  }, [fullData, currentIndex]);

  const initializeChart = useCallback(() => {
    if (!chartContainerRef.current || visibleCandles.length === 0) return;

    // Cleanup previous chart
    if (chartRef.current) {
      try { chartRef.current.remove(); } catch {}
      chartRef.current = null;
      priceSeriesRef.current = null;
    }

    const width = chartContainerRef.current.clientWidth || 800;
    const height = 420;

    const chart = createChart(chartContainerRef.current, {
      width,
      height,
      layout: {
        background: { type: ColorType.Solid, color: isDarkMode ? '#1f2937' : '#ffffff' },
        textColor: isDarkMode ? '#e5e7eb' : '#374151'
      },
      grid: {
        vertLines: { color: isDarkMode ? '#374151' : '#f3f4f6' },
        horzLines: { color: isDarkMode ? '#374151' : '#f3f4f6' }
      },
      rightPriceScale: {
        borderColor: isDarkMode ? '#4b5563' : '#d1d5db'
      },
      timeScale: {
        borderColor: isDarkMode ? '#4b5563' : '#d1d5db',
        timeVisible: true,
        secondsVisible: false
      }
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444'
    });
    series.setData(visibleCandles);

    chartRef.current = chart;
    priceSeriesRef.current = series;

    chart.timeScale().fitContent();

    const onResize = () => {
      const newWidth = chartContainerRef.current?.clientWidth || 800;
      chart.applyOptions({ width: newWidth });
    };
    window.addEventListener('resize', onResize);
    chart._onResize = onResize;
  }, [visibleCandles, isDarkMode]);

  // Update chart data when visibleCandles changes
  useEffect(() => {
    if (!priceSeriesRef.current) {
      initializeChart();
      return;
    }
    priceSeriesRef.current.setData(visibleCandles);
  }, [visibleCandles, initializeChart]);

  // Autoplay timer
  useEffect(() => {
    if (!isPlaying || fullData.length === 0) return;
    const atEnd = currentIndex >= fullData.length - 1;
    if (atEnd) {
      if (loopPlayback) {
        setCurrentIndex(0);
      } else {
        setIsPlaying(false);
      }
      return;
    }
    const id = setTimeout(() => {
      setCurrentIndex((i) => Math.min(i + 1, fullData.length - 1));
    }, speedMs);
    return () => clearTimeout(id);
  }, [isPlaying, speedMs, currentIndex, fullData.length, loopPlayback]);

  // Fetch chart data (mirrors Chart.jsx logic)
  const fetchChartData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setChartData(null);
    try {
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}`;
      const queryParams = new URLSearchParams();

      if (fetchMode === FETCH_MODES.DATE_RANGE) {
        switch (dateRangeType) {
          case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
            if (endDate) queryParams.append('end_date', endDate);
            break;
          case DATE_RANGE_TYPES.DATE_TO_DATE:
            if (startDate) queryParams.append('start_date', startDate);
            if (endDate) queryParams.append('end_date', endDate);
            break;
          case DATE_RANGE_TYPES.DATE_TO_LATEST:
            if (startDate) queryParams.append('start_date', startDate);
            break;
        }
        queryParams.append('limit', '10000');
      } else {
        queryParams.append('limit', String(rowLimit));
      }

      queryParams.append('order', sortOrder || 'desc');
      queryParams.append('sort_by', 'timestamp');

      const response = await fetch(`${url}?${queryParams.toString()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.data && Array.isArray(data.data)) {
        const sorted = [...data.data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        data.data = sorted;
      }

      setChartData(data);
      // Always start at the first candle (index 0)
      setCurrentIndex(0);
    } catch (err) {
      setError(`Failed to fetch chart data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate]);

  // Fetch on inputs change
  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) fetchChartData();
    // Cleanup chart resize listener on unmount
    return () => {
      if (chartRef.current?._onResize) window.removeEventListener('resize', chartRef.current._onResize);
      if (chartRef.current) try { chartRef.current.remove(); } catch {}
    };
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate, fetchChartData]);

  // Header info
  const headerBadge = (
    <div className={`mt-4 text-sm rounded p-2 ${isDarkMode ? 'bg-black/30' : 'bg-white/20'}`}>
      🧪 Backtest • {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • {fullData.length} candles loaded
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-lg transition-colors duration-200 ${
        isDarkMode ? 'bg-gradient-to-r from-purple-800 via-indigo-700 to-blue-800 text-white' : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
      }`}>
        <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            letterSpacing: '-0.02em',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Day<span className="text-green-200">gent</span> <span className="text-xl font-semibold text-green-100">Backtest Dashboard</span>
          </h2>
          <InfoTooltip
            id="backtest-dashboard"
            isDarkMode={isDarkMode}
            content={(
              <div>
                <p className="font-semibold mb-2">🧪 Backtest Dashboard Overview</p>
                <p className="mb-2">Replay historical candles with controllable speed and stepping.</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li>Use the Data Selection box to load symbol/timeframe data.</li>
                  <li>Adjust playback speed and step size below.</li>
                  <li>Move the slider to set what has "happened" so far.</li>
                </ul>
              </div>
            )}
          />
        </div>
        {headerBadge}
      </div>

      {/* Data selection & controls */}
      <DataSelectionControls handleRefresh={fetchChartData} isDarkMode={isDarkMode} />

      {/* Backtest settings */}
      <BacktestSettings
        isDarkMode={isDarkMode}
        fullData={fullData}
        currentIndex={currentIndex}
        setCurrentIndex={setCurrentIndex}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        speedMs={speedMs}
        setSpeedMs={setSpeedMs}
        stepSize={stepSize}
        setStepSize={setStepSize}
        loopPlayback={loopPlayback}
        setLoopPlayback={setLoopPlayback}
      />

      {/* Live Model Prediction Box */}
      <LivePredictionBox
        isDarkMode={isDarkMode}
        selectedSymbol={selectedSymbol}
        selectedTimeframe={selectedTimeframe}
        currentCandle={currentCandle}
        currentIndex={currentIndex}
      />

      {/* Standalone current prediction card */}
      <CurrentPredictionCard
        isDarkMode={isDarkMode}
        prediction={currentPredictionInfo.prediction}
        modelName={currentPredictionInfo.modelName}
      />

      {/* Chart */}
      <div className={`rounded-lg shadow-md transition-colors duration-200 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Price Chart</h3>
          </div>
          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {currentIndex + 1} / {fullData.length} candles shown
          </div>
        </div>

        <div className="p-6">
          {loading && (
            <div className="flex items-center justify-center h-96">
              <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`}></div>
            </div>
          )}
          {error && (
            <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-red-900/20 border-red-800 text-red-300' : 'bg-red-50 border-red-200 text-red-700'}`}>
              {error}
            </div>
          )}
          {!loading && !error && (
            <div
              ref={chartContainerRef}
              className={`w-full h-96 rounded border transition-colors duration-200 ${
                isDarkMode ? 'border-gray-600 bg-gray-900' : 'border-gray-200 bg-gray-50'
              }`}
              style={{ minHeight: '400px' }}
            />
          )}
        </div>
      </div>

      {/* Predictions & confidence */}
      <ModelPredictionPanel
        isDarkMode={isDarkMode}
        selectedSymbol={selectedSymbol}
        selectedTimeframe={selectedTimeframe}
        currentIndex={currentIndex}
        setCurrentIndex={setCurrentIndex}
        fullData={fullData}
        onCurrentPredictionChange={setCurrentPredictionInfo}
      />
    </div>
  );
};

export default BacktestDashboard;


