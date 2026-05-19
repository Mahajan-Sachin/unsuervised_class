import io
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.predict import predict_anomaly

app = Flask(__name__)

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # 1. Check file was sent
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    # 2. Read CSV
    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    # Remove label column if present (we don't use it)
    df = df.drop(columns=['Class'], errors='ignore')

    # 3. Validate columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400
    # 4. Limit rows — model runs one-by-one, large files will hang
    MAX_ROWS = 500
    if len(df) > MAX_ROWS:
        return jsonify({"error": f"Too many rows ({len(df):,}). Max allowed: {MAX_ROWS}. Use test_normal/fraud/mixed.csv instead."}), 400

    # 5. Run prediction on each row
    results = []
    for idx, row in df[FEATURE_COLS].iterrows():
        pred = predict_anomaly(row.values.tolist())
        results.append({
            "row":              idx + 1,
            "time":             round(float(row['Time']), 2),
            "amount":           round(float(row['Amount']), 2),
            "is_anomaly":       pred['is_anomaly'],
            "isolation_forest": pred['isolation_forest'],
            "autoencoder":      pred['autoencoder'],
        })

    # 5. Summary stats
    fraud_count = sum(1 for r in results if r['is_anomaly'])
    return jsonify({
        "total":        len(results),
        "fraud_count":  fraud_count,
        "normal_count": len(results) - fraud_count,
        "fraud_pct":    round(fraud_count / len(results) * 100, 1),
        "transactions": results,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
