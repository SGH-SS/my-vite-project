import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { BxtBreakoutPage, breakoutSpecFromUrl } from './components/BxtVisualizer.jsx'
import { useTheme } from './hooks/useTheme'

// A BXT Visualizer breakout opens in its own browser tab: the URL carries the
// selection (?bxtviz=1&schema=…&tf=…&start=…&end=…) and this tab renders just
// that chart instead of the full dashboard.
function BreakoutRoot({ spec }) {
  const { isDarkMode } = useTheme();
  return <BxtBreakoutPage isDarkMode={isDarkMode} spec={spec} />;
}

const breakoutSpec = breakoutSpecFromUrl();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {breakoutSpec ? <BreakoutRoot spec={breakoutSpec} /> : <App />}
  </StrictMode>,
)
