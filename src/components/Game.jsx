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
        – Q      : X-axis zoom out (decrease barSpacing)
        – E      : X-axis zoom in (increase barSpacing)
        – Z      : Y-axis zoom in (contract price range)
        – C      : Y-axis zoom out (expand price range)
        – Esc    : exit game mode
*/

const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

export const GameModeController = ({ chartRef, containerRef, isDarkMode = false }) => {
  const [active, setActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Simplified settings (stateful)
  const [wasdSpeed, setWasdSpeed] = useState(6); // pixels per frame for WASD movement (increased from 4)
  const [zoomSensitivity, setZoomSensitivity] = useState(0.1); // 0.1 => ±10%
  const [xScaleSensitivity, setXScaleSensitivity] = useState(0.03); // Q/E X-axis scaling sensitivity (lowered from 0.1)
  const [yScaleSensitivity, setYScaleSensitivity] = useState(0.03); // Z/C Y-axis scaling sensitivity (lowered from 0.1)

  const prevStyleRef = useRef({});
  const keysPressed = useRef(new Set());
  const animationFrameRef = useRef(null);

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

  /* ------------------------- Key Binds Display ------------------------- */
  const KeyBindsDisplay = () => (
    <div 
      className="fixed top-4 left-4 z-[10001] p-3 rounded-lg text-xs font-mono"
      style={{
        backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        color: isDarkMode ? '#e5e7eb' : '#374151',
        backdropFilter: 'blur(4px)',
        border: isDarkMode ? '1px solid rgba(75, 85, 99, 0.3)' : '1px solid rgba(209, 213, 219, 0.3)'
      }}
    >
      <div className="font-bold mb-2 text-center">🎮 Game Controls</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div><span className="font-bold">W/A/S/D</span> - Pan</div>
        <div><span className="font-bold">Q/E</span> - X-Scale</div>
        <div><span className="font-bold">Shift</span> - Zoom In</div>
        <div><span className="font-bold">Z/C</span> - Y-Scale</div>
        <div><span className="font-bold">Space</span> - Zoom Out</div>
        <div><span className="font-bold">Esc</span> - Exit</div>
      </div>
    </div>
  );

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

    // Normalized movement functions with zoom-aware speed scaling
    const getMovementSpeedMultiplier = () => {
      // Base multiplier that creates smooth, consistent movement
      return 0.8;
    };

    // Calculate movement amounts directly in chart units
    const getHorizontalMovement = () => {
      try {
        const barSpacing = ts.options().barSpacing || 6;
        // Convert WASD speed (pixels) to bars based on current zoom
        return wasdSpeed / barSpacing;
      } catch (e) {
        return 0.5; // fallback
      }
    };

    const getVerticalMovement = () => {
      try {
        const priceRange = priceScale.getVisibleRange?.();
        if (!priceRange) return 0.001;
        
        const chartHeight = containerRef?.current?.clientHeight || 400;
        const pixelsToFraction = wasdSpeed / chartHeight;
        const priceRangeSize = priceRange.to - priceRange.from;
        
        // Convert pixel movement to price units
        return pixelsToFraction * priceRangeSize;
      } catch (e) {
        return 0.001; // fallback
      }
    };

    const panTime = (bars) => {
      ts.scrollToPosition(ts.scrollPosition() + bars, false);
    };

    const panPrice = (priceMovement) => {
      if (!priceScale.getVisibleRange || !priceScale.setVisibleRange) return;
      const range = priceScale.getVisibleRange();
      if (!range) return;

      // Use the direct price movement value (already calculated in price units)
      const delta = priceMovement;

      if (priceScale.setAutoScale) priceScale.setAutoScale(false);

      // Ensure new range stays valid
      const newFrom = range.from + delta;
      const newTo = range.to + delta;
      if (newTo <= newFrom) return;

      priceScale.setVisibleRange({ from: newFrom, to: newTo });
    };

    // Clear all pressed keys - used when focus is lost or other interruptions
    const clearAllKeys = () => {
      keysPressed.current.clear();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };

    // Periodic cleanup counter to check for stuck keys
    let cleanupCounter = 0;

        // Smooth movement and action animation loop
    const animate = () => {
      // Safety check: if we somehow lost focus but animation is still running, stop it
      if (!document.hasFocus()) {
        clearAllKeys();
        return;
      }

      // Periodic cleanup check (every ~1 second at 60fps)
      cleanupCounter++;
      if (cleanupCounter >= 60) {
        cleanupCounter = 0;
        
        // Aggressive cleanup for modifier keys that commonly get stuck
        // Especially Shift when combined with movement keys
        const suspiciousKeys = ['Shift', 'Control', 'Alt'];
        let hadStuckKeys = false;
        
        suspiciousKeys.forEach(key => {
          if (keysPressed.current.has(key)) {
            // Remove stuck modifier keys - they're most likely to cause issues
            keysPressed.current.delete(key);
            hadStuckKeys = true;
          }
        });
        
        // If we cleared stuck keys and no regular keys remain, stop animation
        if (hadStuckKeys && keysPressed.current.size === 0 && animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        
        // Also run the general suspicious state cleanup
        cleanupSuspiciousKeyStates();
      }

      // Process each key group independently to prevent conflicts

      // Movement keys (WASD) - using pixel-normalized movement
      const hasMovementKeys = keysPressed.current.has('a') || keysPressed.current.has('A') || 
                              keysPressed.current.has('d') || keysPressed.current.has('D') ||
                              keysPressed.current.has('w') || keysPressed.current.has('W') ||
                              keysPressed.current.has('s') || keysPressed.current.has('S');
      
      if (hasMovementKeys) {
        const horizontalMove = getHorizontalMovement();
        const verticalMove = getVerticalMovement();
        
        if (keysPressed.current.has('a') || keysPressed.current.has('A')) {
          panTime(-horizontalMove); // Direct movement in bars
        }
        if (keysPressed.current.has('d') || keysPressed.current.has('D')) {
          panTime(horizontalMove);
        }
        if (keysPressed.current.has('w') || keysPressed.current.has('W')) {
          panPrice(verticalMove); // Direct movement in price units
        }
        if (keysPressed.current.has('s') || keysPressed.current.has('S')) {
          panPrice(-verticalMove); // Direct movement in price units
        }
      }

      // Zoom keys (Shift/Space) - processed independently
      const hasZoomKeys = keysPressed.current.has('Shift') || keysPressed.current.has(' ');
      
      if (hasZoomKeys) {
        const baseSpeed = getMovementSpeedMultiplier();
        if (keysPressed.current.has('Shift')) {
          zoom(1 + zoomSensitivity * baseSpeed); // Consistent speed scaling
        }
        if (keysPressed.current.has(' ')) {
          zoom(1 - zoomSensitivity * baseSpeed);
        }
      }

      // Scaling keys (Q/E/Z/C) - processed independently
      const hasScaleKeys = keysPressed.current.has('q') || keysPressed.current.has('Q') ||
                           keysPressed.current.has('e') || keysPressed.current.has('E') ||
                           keysPressed.current.has('z') || keysPressed.current.has('Z') ||
                           keysPressed.current.has('c') || keysPressed.current.has('C');
      
      if (hasScaleKeys) {
        const baseSpeed = getMovementSpeedMultiplier();
        
        // X-axis scaling keys
        if (keysPressed.current.has('q') || keysPressed.current.has('Q')) {
          scaleXAxis(-baseSpeed); // Q zooms out X-axis
        }
        if (keysPressed.current.has('e') || keysPressed.current.has('E')) {
          scaleXAxis(baseSpeed); // E zooms in X-axis
        }

        // Y-axis scaling keys
        if (keysPressed.current.has('z') || keysPressed.current.has('Z')) {
          scaleYAxis(baseSpeed); // Z zooms in Y-axis
        }
        if (keysPressed.current.has('c') || keysPressed.current.has('C')) {
          scaleYAxis(-baseSpeed); // C zooms out Y-axis
        }
      }

      if (keysPressed.current.size > 0) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    const scaleXAxis = (direction) => {
      // Scale X-axis by adjusting barSpacing
      const current = ts.options().barSpacing || 6;
      const factor = direction > 0 ? (1 + xScaleSensitivity) : (1 - xScaleSensitivity);
      const next = clamp(current * factor, 0.5, 200);
      ts.applyOptions({ barSpacing: next });
    };

    const scaleYAxis = (direction) => {
      // Scale Y-axis by adjusting the visible price range
      if (!priceScale.getVisibleRange || !priceScale.setVisibleRange) return;
      const range = priceScale.getVisibleRange();
      if (!range) return;

      if (priceScale.setAutoScale) priceScale.setAutoScale(false);

      const center = (range.from + range.to) / 2;
      const currentHeight = range.to - range.from;
      const factor = direction > 0 ? (1 + yScaleSensitivity) : (1 - yScaleSensitivity);
      const newHeight = currentHeight * factor;

      // Keep the center point the same while scaling
      const newFrom = center - newHeight / 2;
      const newTo = center + newHeight / 2;

      if (newTo <= newFrom) return;

      priceScale.setVisibleRange({ from: newFrom, to: newTo });
    };

    const handleKeyDown = (e) => {
      // Define key groups for independent processing
      const movementKeys = ['a', 'A', 'd', 'D', 'w', 'W', 's', 'S'];
      const zoomKeys = ['Shift', ' '];
      const scaleKeys = ['q', 'Q', 'e', 'E', 'z', 'Z', 'c', 'C'];
      const allInteractiveKeys = [...movementKeys, ...zoomKeys, ...scaleKeys];
      
      if (allInteractiveKeys.includes(e.key)) {
        if (e.key === ' ') e.preventDefault(); // Prevent space from scrolling page
        
        if (!keysPressed.current.has(e.key)) {
          keysPressed.current.add(e.key);
          
          // Start animation loop if this is the first interactive key
          if (keysPressed.current.size === 1) {
            animate();
          }
        }
        return;
      }

      // Handle non-interactive keys (single actions)
      switch (e.key) {
        case 'Escape':
          exit();
          break;
        default:
          break;
      }
    };

    const handleKeyUp = (e) => {
      // Define key groups for independent processing
      const movementKeys = ['a', 'A', 'd', 'D', 'w', 'W', 's', 'S'];
      const zoomKeys = ['Shift', ' '];
      const scaleKeys = ['q', 'Q', 'e', 'E', 'z', 'Z', 'c', 'C'];
      const allInteractiveKeys = [...movementKeys, ...zoomKeys, ...scaleKeys];
      
      if (allInteractiveKeys.includes(e.key)) {
        keysPressed.current.delete(e.key);
        
        // Special handling for modifier keys - also clear related variants
        if (e.key === 'Shift') {
          // Clear both 'Shift' and any shift variants that might be stuck
          keysPressed.current.delete('Shift');
        }
        
        // Check for suspicious key states after any key release
        cleanupSuspiciousKeyStates();
        
        if (keysPressed.current.size === 0 && animationFrameRef.current) {
          // Stop animation loop when no interactive keys are pressed
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      }
    };

    // Additional cleanup function for problematic key combinations
    const cleanupSuspiciousKeyStates = () => {
      // Define key groups for independent processing
      const movementKeys = ['a', 'A', 'd', 'D', 'w', 'W', 's', 'S'];
      const zoomKeys = ['Shift', ' '];
      const scaleKeys = ['q', 'Q', 'e', 'E', 'z', 'Z', 'c', 'C'];
      const modifierKeys = ['Shift', 'Control', 'Alt'];
      
      const hasMovement = movementKeys.some(key => keysPressed.current.has(key));
      const hasZoom = zoomKeys.some(key => keysPressed.current.has(key));
      const hasScale = scaleKeys.some(key => keysPressed.current.has(key));
      const hasModifiers = modifierKeys.some(key => keysPressed.current.has(key));
      const hasRegularKeys = hasMovement || hasScale || keysPressed.current.has(' ');
      
      // If we only have modifier keys (especially Shift) with no regular keys,
      // it's likely a stuck state - clear the modifiers
      if (hasModifiers && !hasRegularKeys) {
        modifierKeys.forEach(key => keysPressed.current.delete(key));
        
        // Stop animation if no keys remain
        if (keysPressed.current.size === 0 && animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      }
    };

    // Handle window focus loss/gain
    const handleWindowBlur = () => {
      clearAllKeys();
    };

    const handleWindowFocus = () => {
      clearAllKeys(); // Clear any stuck keys when regaining focus
    };

    // Handle tab visibility changes
    const handleVisibilityChange = () => {
      if (document.hidden) {
        clearAllKeys();
      }
    };

    const handleResize = () => {
      if (chartRef?.current) {
        chartRef.current.resize(window.innerWidth, window.innerHeight);
      }
    };

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
      // Clean up animation frame and clear keys
      clearAllKeys();
    };
  }, [active, chartRef, wasdSpeed, zoomSensitivity, xScaleSensitivity, yScaleSensitivity]);

  /* --------------------- Sensitivity Settings Modal -------------------- */
  const SettingsModal = () => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className={`rounded-lg p-6 w-80 ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
        <h3 className="text-lg font-semibold mb-4">Game Mode Settings</h3>
        <p className="text-xs mb-4 opacity-75">
          🎮 Independent key processing - mix WASD + Shift + Q/E/Z/C without conflicts!
        </p>

        <label className="block text-sm mb-2">WASD Movement Speed (2 - 15 pixels)</label>
        <input
          type="number"
          min="2"
          max="15"
          step="1"
          value={wasdSpeed}
          onChange={(e) => setWasdSpeed(parseInt(e.target.value) || 6)}
          className="w-full mb-4 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <label className="block text-sm mb-2">Zoom Sensitivity (0.05 - 0.3)</label>
        <input
          type="number"
          min="0.05"
          max="0.3"
          step="0.01"
          value={zoomSensitivity}
          onChange={(e) => setZoomSensitivity(parseFloat(e.target.value) || 0.1)}
          className="w-full mb-4 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <label className="block text-sm mb-2">X-Axis Scale (Q/E) Sensitivity (0.01 - 0.15)</label>
        <input
          type="number"
          min="0.01"
          max="0.15"
          step="0.01"
          value={xScaleSensitivity}
          onChange={(e) => setXScaleSensitivity(parseFloat(e.target.value) || 0.03)}
          className="w-full mb-4 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <label className="block text-sm mb-2">Y-Axis Scale (Z/C) Sensitivity (0.01 - 0.15)</label>
        <input
          type="number"
          min="0.01"
          max="0.15"
          step="0.01"
          value={yScaleSensitivity}
          onChange={(e) => setYScaleSensitivity(parseFloat(e.target.value) || 0.03)}
          className="w-full mb-4 px-2 py-1 rounded border border-gray-300 text-sm bg-transparent"
        />

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={() => setSettingsOpen(false)}
            className="px-4 py-2 rounded bg-gray-400 text-sm text-white hover:bg-gray-500"
          >Cancel</button>
          <button
            onClick={() => {
              setSettingsOpen(false);
              enter();
            }}
            className="px-4 py-2 rounded bg-blue-600 text-sm text-white hover:bg-blue-700"
          >Start Game Mode</button>
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
      {active && <KeyBindsDisplay />}
    </>
  );
};

export default GameModeController; 