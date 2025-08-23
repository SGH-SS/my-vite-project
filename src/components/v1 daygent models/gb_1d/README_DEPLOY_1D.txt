============================================
GradientBoosting 1D Inference — Deployment Notes
============================================

Artifacts:
- Model:       gb_1d_final.joblib
- Scaler:      scaler_1d.joblib
- Config:      deployment_config.json
- Feature schema: feature_schema.json
- Threshold:   0.57
- Predictions: test_predictions_1d.csv
- Report:      gb_1d_day_by_day.txt

Feature order (must match EXACTLY):
['raw_o', 'raw_h', 'raw_l', 'raw_c', 'raw_v', 'iso_0', 'iso_1', 'iso_2', 'iso_3', 'tf_1d', 'tf_4h', 'hl_range', 'price_change', 'upper_shadow', 'lower_shadow', 'volume_m']

Inference pipeline for your site:
1) Parse raw input row:
   - Parse 'raw_ohlcv_vec' -> [o,h,l,c,v]
   - Parse 'iso_ohlc'      -> [iso_0..iso_3]
   - Add one-hot: tf_1d=1, tf_4h=0
   - Compute engineered features as in feature_schema.json
   - Concatenate into a single 16-length vector in the listed order.

2) Load scaler with joblib and call scaler.transform([vector]).
3) Load model with joblib and call model.predict_proba(scaled)[0,1].
4) If prob >= 0.57 => predict UP (1); else DOWN (0).

Notes:
- Trained with class_weight via compute_sample_weight('balanced').
- Scaler fit only on pre-test train split (first 80%).
- Keep feature order and scaling identical for consistent results.