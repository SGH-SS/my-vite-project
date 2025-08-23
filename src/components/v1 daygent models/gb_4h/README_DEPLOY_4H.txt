============================================
GradientBoosting 4H Inference — Deployment Notes
============================================

Artifacts:
- Model:       gb_4h_w2_style.joblib
- Scaler:      scaler_4h_w2_style.joblib
- Config:      deployment_config_4h.json
- Feature schema: feature_schema_4h.json
- Predictions: test_predictions_4h.csv
- Report:      gb_4h_day_by_day.txt

Pipeline (w2-style):
• Train on first 80% slice of pre-test 4h data (after scaling with that same slice).
• No threshold calibration; default 0.50 is used.
• No refit on train+val.
• Inference uses scaler.transform then model.predict_proba(...)[1] and 0.50 cutoff.

Feature order (must match EXACTLY):
['raw_o', 'raw_h', 'raw_l', 'raw_c', 'raw_v', 'iso_0', 'iso_1', 'iso_2', 'iso_3', 'tf_1d', 'tf_4h', 'hl_range', 'price_change', 'upper_shadow', 'lower_shadow', 'volume_m']

Inference steps for your site:
1) Parse inputs:
   - 'raw_ohlcv_vec' -> [o,h,l,c,v]
   - 'iso_ohlc'      -> [iso_0..iso_3]
   - one-hot: tf_1d=0, tf_4h=1
   - engineered: hl_range, price_change, upper_shadow, lower_shadow, volume_m
   - concatenate into a 16-length vector in the EXACT order above
2) Load scaler (joblib) and transform the vector.
3) Load model (joblib) and compute P(up) = predict_proba(...)[0,1].
4) Predict UP if P(up) >= 0.50 else DOWN.

Keep the feature order + scaling identical for consistent results.