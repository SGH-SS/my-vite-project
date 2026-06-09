/**
 * GameModeController - Full-screen interactive chart navigation
 * 
 * Features:
 * - Full-screen chart mode with keyboard controls
 * - WASD pan navigation (pixel-normalized)
 * - Zoom controls (Shift/Space)
 * - X/Y axis scaling (Q/E and Z/C)
 * - Configurable sensitivity settings
 * - Robust key handling without stuck keys
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ============================================================================
// CONSTANTS
// ============================================================================

const DEFAULT_SETTINGS = {
  wasdSpeed: 6,
  zoomSensitivity: 0.1,
  xScaleSensitivity: 0.03,
  yScaleSensitivity: 0.03,
};

const KEY_GROUPS = {
  movement: ['a', 'A', 'd', 'D', 'w', 'W', 's', 'S'],
  zoom: ['Shift', ' '],
  scale: ['q', 'Q', 'e', 'E', 'z', 'Z', 'c', 'C'],
};

// Combine all interactive keys
const ALL_INTERACTIVE_KEYS = [
  ...KEY_GROUPS.movement,
  ...KEY_GROUPS.zoom,
  ...KEY_GROUPS.scale
];

const MODIFIER_KEYS = ['Shift', 'Control', 'Alt'];

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

const KeyBindsDisplay = ({ isDarkMode }) => (
  <div 
    className="fixed top-4 left-4 z-[10001] p-3 rounded-lg text-xs font-mono"
    style={{
      backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      color: isDarkMode ? '#e5e7eb' : '#374151',
      backdropFilter: 'blur(4px)',
      border: isDarkMode ? '1px solid rgba(75, 85, 99, 0.5)' : '1px solid rgba(209, 213, 219, 0.5)'
    }}
  >
    <div className="font-bold mb-2 text-center text-sm">🎮 Game Controls</div>
    <div className="grid grid-cols-2 gap-x-6 gap-y-1">
      <div><kbd className="px-1 bg-gray-700 rounded text-white">W/A/S/D</kbd> Pan</div>
      <div><kbd className="px-1 bg-gray-700 rounded text-white">Q/E</kbd> X-Scale</div>
      <div><kbd className="px-1 bg-gray-700 rounded text-white">Shift</kbd> Zoom In</div>
      <div><kbd className="px-1 bg-gray-700 rounded text-white">Z/C</kbd> Y-Scale</div>
      <div><kbd className="px-1 bg-gray-700 rounded text-white">Space</kbd> Zoom Out</div>
      <div><kbd className="px-1 bg-gray-700 rounded text-white">Esc</kbd> Exit</div>
    </div>
  </div>
);

const SettingsModal = ({ isDarkMode, settings, setSettings, onCancel, onStart }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
    <div className={`rounded-lg p-6 w-80 ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
      <h3 className="text-lg font-semibold mb-4">🎮 Game Mode Settings</h3>
      <p className="text-xs mb-4 opacity-75">
        Configure controls before entering full-screen mode.
      </p>

      <div className="space-y-4">
        <div>
          <label className="block text-sm mb-1">WASD Movement Speed (2-15)</label>
          <input
            type="number"
            min="2"
            max="15"
            step="1"
            value={settings.wasdSpeed}
            onChange={(e) => setSettings(s => ({ ...s, wasdSpeed: parseInt(e.target.value) || DEFAULT_SETTINGS.wasdSpeed }))}
            className={`w-full px-2 py-1 rounded border text-sm ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}
          />
        </div>

        <div>
          <label className="block text-sm mb-1">Zoom Sensitivity (0.05-0.3)</label>
          <input
            type="number"
            min="0.05"
            max="0.3"
            step="0.01"
            value={settings.zoomSensitivity}
            onChange={(e) => setSettings(s => ({ ...s, zoomSensitivity: parseFloat(e.target.value) || DEFAULT_SETTINGS.zoomSensitivity }))}
            className={`w-full px-2 py-1 rounded border text-sm ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}
          />
        </div>

        <div>
          <label className="block text-sm mb-1">X-Axis Scale (Q/E) (0.01-0.15)</label>
          <input
            type="number"
            min="0.01"
            max="0.15"
            step="0.01"
            value={settings.xScaleSensitivity}
            onChange={(e) => setSettings(s => ({ ...s, xScaleSensitivity: parseFloat(e.target.value) || DEFAULT_SETTINGS.xScaleSensitivity }))}
            className={`w-full px-2 py-1 rounded border text-sm ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}
          />
        </div>

        <div>
          <label className="block text-sm mb-1">Y-Axis Scale (Z/C) (0.01-0.15)</label>
          <input
            type="number"
            min="0.01"
            max="0.15"
            step="0.01"
            value={settings.yScaleSensitivity}
            onChange={(e) => setSettings(s => ({ ...s, yScaleSensitivity: parseFloat(e.target.value) || DEFAULT_SETTINGS.yScaleSensitivity }))}
            className={`w-full px-2 py-1 rounded border text-sm ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}
          />
        </div>
      </div>

      <div className="flex justify-end gap-2 mt-6">
        <button
          onClick={onCancel}
          className="px-4 py-2 rounded bg-gray-500 text-sm text-white hover:bg-gray-600"
        >
          Cancel
        </button>
        <button
          onClick={onStart}
          className="px-4 py-2 rounded bg-blue-600 text-sm text-white hover:bg-blue-700"
        >
          Start Game Mode
        </button>
      </div>
    </div>
  </div>
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const GameModeController = ({ chartRef, containerRef, isDarkMode = false }) => {
  const [active, setActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

  const prevStyleRef = useRef({});
  const keysPressed = useRef(new Set());
  const animationFrameRef = useRef(null);

  // ========================================================================
  // FULL-SCREEN MANAGEMENT
  // ========================================================================
  
  const enterFullscreen = useCallback(() => {
    if (active || !containerRef?.current) return;
    
    const el = containerRef.current;

    // Store previous styles for restoration
    prevStyleRef.current = {
      position: el.style.position,
      left: el.style.left,
      top: el.style.top,
      width: el.style.width,
      height: el.style.height,
      zIndex: el.style.zIndex,
    };

    // Apply fullscreen styles
    Object.assign(el.style, {
      position: 'fixed',
      left: '0',
      top: '0',
      width: '100vw',
      height: '100vh',
      zIndex: '10000',
    });

    // Resize chart to fill viewport
    if (chartRef?.current) {
      chartRef.current.resize(window.innerWidth, window.innerHeight);
    }

    setActive(true);
  }, [active, containerRef, chartRef]);

  const exitFullscreen = useCallback(() => {
    if (!active || !containerRef?.current) return;
    
    const el = containerRef.current;

    // Restore previous styles
    Object.assign(el.style, prevStyleRef.current);

    // Resize chart back to container size
    if (chartRef?.current) {
      const w = el.clientWidth || 800;
      const h = el.clientHeight || 400;
      chartRef.current.resize(w, h);
    }

    // Clear all pressed keys to prevent stuck state
    keysPressed.current.clear();
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    setActive(false);
  }, [active, containerRef, chartRef]);

  // ========================================================================
  // KEYBOARD CONTROLS
  // ========================================================================
  
  useEffect(() => {
    if (!active || !chartRef?.current) return;

    const chart = chartRef.current;
    const timeScale = chart.timeScale();
    const priceScale = chart.priceScale('right');

    // Clear all pressed keys (called on focus loss)
    const clearAllKeys = () => {
      keysPressed.current.clear();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    // Zoom function
    const zoom = (factor) => {
      const current = timeScale.options().barSpacing || 6;
      const next = clamp(current * factor, 0.5, 200);
      timeScale.applyOptions({ barSpacing: next });
    };

    // Calculate horizontal movement (in bars)
    const getHorizontalMovement = () => {
      const barSpacing = timeScale.options().barSpacing || 6;
      return settings.wasdSpeed / barSpacing;
    };

    // Calculate vertical movement (in price units)
    const getVerticalMovement = () => {
      const range = priceScale.getVisibleRange?.();
      if (!range) return 0.001;
      
      const chartHeight = containerRef?.current?.clientHeight || 400;
      const pixelsToFraction = settings.wasdSpeed / chartHeight;
      const priceRangeSize = range.to - range.from;
      
      return pixelsToFraction * priceRangeSize;
    };

    // Pan time (horizontal)
    const panTime = (bars) => {
      timeScale.scrollToPosition(timeScale.scrollPosition() + bars, false);
    };

    // Pan price (vertical)
    const panPrice = (priceMovement) => {
      if (!priceScale.getVisibleRange || !priceScale.setVisibleRange) return;
      
      const range = priceScale.getVisibleRange();
      if (!range) return;

      // Disable auto-scale to allow manual positioning
      if (priceScale.setAutoScale) priceScale.setAutoScale(false);

      const newFrom = range.from + priceMovement;
      const newTo = range.to + priceMovement;
      if (newTo <= newFrom) return;

      priceScale.setVisibleRange({ from: newFrom, to: newTo });
    };

    // Scale X-axis
    const scaleXAxis = (direction) => {
      const current = timeScale.options().barSpacing || 6;
      const factor = direction > 0 ? (1 + settings.xScaleSensitivity) : (1 - settings.xScaleSensitivity);
      const next = clamp(current * factor, 0.5, 200);
      timeScale.applyOptions({ barSpacing: next });
    };

    // Scale Y-axis
    const scaleYAxis = (direction) => {
      if (!priceScale.getVisibleRange || !priceScale.setVisibleRange) return;
      
      const range = priceScale.getVisibleRange();
      if (!range) return;

      if (priceScale.setAutoScale) priceScale.setAutoScale(false);

      const center = (range.from + range.to) / 2;
      const currentHeight = range.to - range.from;
      const factor = direction > 0 ? (1 + settings.yScaleSensitivity) : (1 - settings.yScaleSensitivity);
      const newHeight = currentHeight * factor;

      const newFrom = center - newHeight / 2;
      const newTo = center + newHeight / 2;
      if (newTo <= newFrom) return;

      priceScale.setVisibleRange({ from: newFrom, to: newTo });
    };

    // Animation loop for smooth continuous input
    const animate = () => {
      // Safety: stop if focus lost
      if (!document.hasFocus()) {
        clearAllKeys();
        return;
      }

      const keys = keysPressed.current;
      const speedMultiplier = 0.8;

      // Movement (WASD)
      if (keys.has('a') || keys.has('A')) panTime(-getHorizontalMovement());
      if (keys.has('d') || keys.has('D')) panTime(getHorizontalMovement());
      if (keys.has('w') || keys.has('W')) panPrice(getVerticalMovement());
      if (keys.has('s') || keys.has('S')) panPrice(-getVerticalMovement());

      // Zoom (Shift/Space)
      if (keys.has('Shift')) zoom(1 + settings.zoomSensitivity * speedMultiplier);
      if (keys.has(' ')) zoom(1 - settings.zoomSensitivity * speedMultiplier);

      // X-axis scaling (Q/E)
      if (keys.has('q') || keys.has('Q')) scaleXAxis(-speedMultiplier);
      if (keys.has('e') || keys.has('E')) scaleXAxis(speedMultiplier);

      // Y-axis scaling (Z/C)
      if (keys.has('z') || keys.has('Z')) scaleYAxis(speedMultiplier);
      if (keys.has('c') || keys.has('C')) scaleYAxis(-speedMultiplier);

      // Continue animation if keys are pressed
      if (keys.size > 0) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    // Keydown handler
    const handleKeyDown = (e) => {
      // Check if this is an interactive key
      if (ALL_INTERACTIVE_KEYS.includes(e.key)) {
        if (e.key === ' ') e.preventDefault(); // Prevent page scroll
        
        if (!keysPressed.current.has(e.key)) {
          keysPressed.current.add(e.key);
          
          // Start animation loop if first key
          if (keysPressed.current.size === 1 && !animationFrameRef.current) {
            animate();
          }
        }
        return;
      }

      // Handle exit
      if (e.key === 'Escape') {
        exitFullscreen();
      }
    };

    // Keyup handler
    const handleKeyUp = (e) => {
      if (ALL_INTERACTIVE_KEYS.includes(e.key)) {
        keysPressed.current.delete(e.key);
        
        // Special handling: also clear opposite case
        const opposite = e.key === e.key.toLowerCase() ? e.key.toUpperCase() : e.key.toLowerCase();
        keysPressed.current.delete(opposite);

        // Stop animation if no keys pressed
        if (keysPressed.current.size === 0 && animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      }
    };

    // Focus handlers
    const handleWindowBlur = () => clearAllKeys();
    const handleWindowFocus = () => clearAllKeys();
    const handleVisibilityChange = () => {
      if (document.hidden) clearAllKeys();
    };

    // Resize handler
    const handleResize = () => {
      if (chartRef?.current) {
        chartRef.current.resize(window.innerWidth, window.innerHeight);
      }
    };

    // Add listeners
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleWindowBlur);
    window.addEventListener('focus', handleWindowFocus);
    window.addEventListener('resize', handleResize);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleWindowBlur);
      window.removeEventListener('focus', handleWindowFocus);
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      clearAllKeys();
    };
  }, [active, chartRef, containerRef, settings, exitFullscreen]);

  // ========================================================================
  // HANDLERS
  // ========================================================================
  
  const handleIconClick = () => {
    if (active) {
      exitFullscreen();
    } else {
      setSettingsOpen(true);
    }
  };

  const handleStartGame = () => {
    setSettingsOpen(false);
    enterFullscreen();
  };

  // ========================================================================
  // RENDER
  // ========================================================================
  
  return (
    <>
      <button
        onClick={handleIconClick}
        title={active ? 'Exit Game Mode' : 'Enter Game Mode'}
        disabled={!containerRef?.current}
        className={`ml-2 text-xl hover:opacity-80 focus:outline-none transition-transform hover:scale-110 ${
          isDarkMode ? 'text-white' : 'text-gray-800'
        }`}
        style={{ cursor: containerRef?.current ? 'pointer' : 'not-allowed' }}
      >
        🎮
      </button>

      {settingsOpen && (
        <SettingsModal
          isDarkMode={isDarkMode}
          settings={settings}
          setSettings={setSettings}
          onCancel={() => setSettingsOpen(false)}
          onStart={handleStartGame}
        />
      )}

      {active && <KeyBindsDisplay isDarkMode={isDarkMode} />}
    </>
  );
};

export default GameModeController;
