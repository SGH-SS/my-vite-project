// New file implementing game mode logic
import React, { useState, useEffect, useRef } from 'react';

/*
  GameModeController
  -------------------
  Props:
    chartRef      – ref to the active lightweight-charts chart instance (IChartApi)
    containerRef  – ref to the DOM element that contains the chart
    isDarkMode    – boolean flag to switch button colours

  Behaviour:
    • Displays a small "🎮" logo-button. Clicking it toggles full-screen game mode.
    • In game mode the chart container is stretched to fill the viewport and
      the underlying chart is resized to match.
    • Keyboard controls while active:
        – A / D : pan left / right (timeScale.scrollToPosition)
        – W / S : small pan up / down (same horizontal pan for now)
        – Shift  : zoom-in step (reduce barSpacing)
        – Space  : zoom-out step (increase barSpacing)
        – Esc    : exit game mode
*/

const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

export const GameModeController = ({ chartRef, containerRef, isDarkMode = false }) => {
  const [active, setActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Sensitivity settings (stateful)
  const [panStepBars, setPanStepBars] = useState(20); // horizontal pan in bars
  const [zoomSensitivity, setZoomSensitivity] = useState(0.1); // 0.1 => ±10%
  const [verticalSensitivity, setVerticalSensitivity] = useState(0.05); // fraction of visible range

  const prevStyleRef = useRef({});

  /* ----------------------- Full-screen management ---------------------- */
  const enter = () => {
    if (active || !containerRef?.current) return;
    const el = containerRef.current;

    prevStyleRef.current = {
      position: el.style.position,
      left: el.style.left,
      top: el.style.top,
      width: el.style.width,
      height: el.style.height,
      zIndex: el.style.zIndex,
    };

    Object.assign(el.style, {
      position: 'fixed',
      left: '0',
      top: '0',
      width: '100vw',
      height: '100vh',
      zIndex: 10000,
    });

    if (chartRef?.current) {
      chartRef.current.resize(window.innerWidth, window.innerHeight);
    }

    setActive(true);
  };

  const exit = () => {
    if (!active || !containerRef?.current) return;
    const el = containerRef.current;
    Object.assign(el.style, prevStyleRef.current);

    if (chartRef?.current) {
      const w = el.clientWidth || 800;
      const h = el.clientHeight || 400;
      chartRef.current.resize(w, h);
    }

    setActive(false);
  };

  /* ------------------------- Keyboard controls ------------------------- */
  useEffect(() => {
    if (!active) return;
    if (!chartRef?.current) return;

    const chart = chartRef.current;
    const ts = chart.timeScale();
    const priceScale = chart.priceScale('right');

    const zoom = (factor) => {
      const current = ts.options().barSpacing || 6;
      const next = clamp(current * factor, 0.5, 200);
      ts.applyOptions({ barSpacing: next });
    };

    const panTime = (bars) => {
      ts.scrollToPosition(ts.scrollPosition() + bars, false);
    };

    const panPrice = (direction) => {
      if (!priceScale.getVisibleRange || !priceScale.setVisibleRange) return;
      const range = priceScale.getVisibleRange();
      if (!range) return;

      const delta = (range.to - range.from) * verticalSensitivity * direction;

      if (priceScale.setAutoScale) priceScale.setAutoScale(false);

      // Ensure new range stays valid
      const newFrom = range.from + delta;
      const newTo = range.to + delta;
      if (newTo <= newFrom) return;

      priceScale.setVisibleRange({ from: newFrom, to: newTo });
    };

    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'a':
        case 'A':
          // Move chart RIGHT (older data to the right edge) – negative scroll
          panTime(-panStepBars);
          break;
        case 'd':
        case 'D':
          // Move chart LEFT – positive scroll
          panTime(panStepBars);
          break;
        case 'w':
        case 'W':
          // Simulate drag down (move range up)
          panPrice(1);
          break;
        case 's':
        case 'S':
          // Simulate drag up (move range down)
          panPrice(-1);
          break;
        case 'Shift':
          // Zoom OUT
          zoom(1 + zoomSensitivity);
          break;
        case ' ': // Space bar => zoom IN
          e.preventDefault();
          zoom(1 - zoomSensitivity);
          break;
        case 'Escape':
          exit();
          break;
        default:
          break;
      }
    };

    const handleResize = () => {
      if (chartRef?.current) {
        chartRef.current.resize(window.innerWidth, window.innerHeight);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', handleResize);
    };
  }, [active, chartRef, panStepBars, zoomSensitivity, verticalSensitivity]);

  /* --------------------- Sensitivity Settings Modal -------------------- */
  const SettingsModal = () => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className={`rounded-lg p-6 w-80 ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
        <h3 className="text-lg font-semibold mb-4">Game Mode Settings</h3>

        <label className="block text-sm mb-2">Horizontal Pan (bars)</label>
        <input
          type="number"
          value={panStepBars}
          onChange={(e) => setPanStepBars(parseInt(e.target.value, 10) || 1)}
          className="w-full mb-3 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <label className="block text-sm mb-2">Zoom Sensitivity (0.01 - 0.5)</label>
        <input
          type="number"
          min="0.01"
          max="0.5"
          step="0.01"
          value={zoomSensitivity}
          onChange={(e) => setZoomSensitivity(parseFloat(e.target.value) || 0.1)}
          className="w-full mb-3 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={() => setSettingsOpen(false)}
            className="px-3 py-1 rounded bg-gray-400 text-sm text-white"
          >Cancel</button>
          <button
            onClick={() => {
              setSettingsOpen(false);
              enter();
            }}
            className="px-3 py-1 rounded bg-blue-600 text-sm text-white"
          >Start</button>
        </div>
      </div>
    </div>
  );

  /* --------------------------- Icon click --------------------------- */
  const handleIconClick = () => {
    if (active) {
      exit();
    } else {
      setSettingsOpen(true);
    }
  };

  /* ----------------------------- Render ----------------------------- */
  return (
    <>
      <button
        onClick={handleIconClick}
        title={active ? 'Exit Game Mode' : 'Enter Game Mode'}
        className={`${isDarkMode ? 'text-white' : 'text-gray-800'} ml-2 text-xl hover:opacity-80 focus:outline-none`}
        style={{ cursor: 'pointer' }}
        disabled={!containerRef?.current}
      >
        🎮
      </button>

      {settingsOpen && <SettingsModal />}
    </>
  );
};

export default GameModeController; 