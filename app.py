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
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    # Drop Class column if present (label — not needed for prediction)
    df = df.drop(columns=['Class'], errors='ignore')

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    results = []
    for idx, row in df[FEATURE_COLS].iterrows():
        try:
            pred = predict_anomaly(row.values.tolist())
            results.append({
                "row": idx + 1,
                "time": round(float(row['Time']), 2),
                "amount": round(float(row['Amount']), 2),
                "result": pred['result'],
                "is_anomaly": pred['is_anomaly'],
                "isolation_forest": pred['isolation_forest'],
                "lof": pred['lof'],
                "autoencoder": pred['autoencoder'],
                "recon_error": pred['recon_error'],
            })
        except Exception as e:
            results.append({"row": idx + 1, "error": str(e)})

    fraud_count  = sum(1 for r in results if r.get('is_anomaly'))
    normal_count = len(results) - fraud_count

    return jsonify({
        "total":        len(results),
        "fraud_count":  fraud_count,
        "normal_count": normal_count,
        "fraud_pct":    round(fraud_count / len(results) * 100, 1) if results else 0,
        "transactions": results,
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
