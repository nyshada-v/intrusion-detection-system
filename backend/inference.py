"""
inference.py - Hybrid ML Inference Pipeline
============================================
Mirrors the exact preprocessing + feature engineering + hybrid scoring
pipeline from the training notebook (anomaly_detection_v3).

Stage 1: IsolationForest + Autoencoder - XGBoost binary detector
Stage 2: XGBoost attack type classifier (on detected attacks)

Usage:
    from inference import IDSInferenceEngine
    engine = IDSInferenceEngine(model_dir="models")
    result = engine.predict(flow_dict)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

warnings.filterwarnings("ignore")

# - Constants (must match training notebook exactly) -

DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Timestamp"
]

# Remediation advice per attack type
ATTACK_ADVICE = {
    "DoS Hulk": {
        "description": "HTTP Flood (DoS Hulk) - Massive volume of HTTP requests overwhelming the server.",
        "advice": [
            "Enable rate limiting on your web server or firewall.",
            "Use a Web Application Firewall (WAF) or DDoS protection service (e.g., Cloudflare).",
            "Consider blocking the source IP temporarily.",
            "Contact your ISP if the attack volume exceeds your infrastructure capacity.",
        ],
    },
    "DoS GoldenEye": {
        "description": "DoS GoldenEye - Keeps HTTP connections open to exhaust server resources.",
        "advice": [
            "Enable connection timeout settings on your web server.",
            "Limit the maximum number of connections per IP.",
            "Use a reverse proxy (e.g., Nginx) to absorb the connection pressure.",
        ],
    },
    "DoS slowloris": {
        "description": "Slowloris - Slowly sends HTTP headers to hold connections open, starving the server.",
        "advice": [
            "Set aggressive connection timeout values on your server.",
            "Use Nginx instead of Apache (Nginx is not vulnerable to Slowloris by design).",
            "Enable the mod_reqtimeout module if using Apache.",
        ],
    },
    "DoS Slowhttptest": {
        "description": "Slow HTTP attack - Sends data at an extremely slow rate to keep connections alive.",
        "advice": [
            "Configure minimum data rate thresholds on your web server.",
            "Deploy a WAF to detect and block slow HTTP attacks.",
            "Reduce the maximum request body size and timeout values.",
        ],
    },
    "FTP-Patator": {
        "description": "FTP Brute Force (Patator) - Automated password guessing on your FTP service.",
        "advice": [
            "Disable FTP if not needed; use SFTP instead.",
            "Enable account lockout after failed login attempts.",
            "Use strong, unique passwords and consider key-based authentication.",
            "Restrict FTP access to whitelisted IPs only.",
        ],
    },
    "SSH-Patator": {
        "description": "SSH Brute Force (Patator) - Automated password guessing on your SSH service.",
        "advice": [
            "Disable password-based SSH login; use SSH key pairs.",
            "Change the default SSH port (22) to a non-standard port.",
            "Install fail2ban to automatically block repeated failed attempts.",
            "Restrict SSH access to specific trusted IP addresses.",
        ],
    },
    "Bot": {
        "description": "Botnet Activity - Your system may be communicating with a botnet C&C server.",
        "advice": [
            "Run a full malware scan immediately (Malwarebytes, Windows Defender).",
            "Isolate the affected machine from the network.",
            "Check for unusual scheduled tasks, startup programs, or processes.",
            "Change all passwords from a clean, unaffected device.",
        ],
    },
    "Web Brute Force": {
        "description": "Web Application Brute Force - Login page is being attacked with credential stuffing.",
        "advice": [
            "Enable CAPTCHA on your login pages.",
            "Implement account lockout after N failed attempts.",
            "Enable Multi-Factor Authentication (MFA).",
            "Block the attacking IP range via your firewall or WAF.",
        ],
    },
    "Web XSS": {
        "description": "Cross-Site Scripting (XSS) - Malicious scripts are being injected into web pages.",
        "advice": [
            "Sanitise and escape all user input on your web application.",
            "Implement a strict Content Security Policy (CSP) header.",
            "Use a WAF to filter XSS payloads.",
            "Update all web frameworks and libraries to the latest versions.",
        ],
    },
    "Web SQL Injection": {
        "description": "SQL Injection - Attacker is attempting to manipulate your database queries.",
        "advice": [
            "Use parameterised queries / prepared statements in your code.",
            "Deploy a WAF with SQL injection filtering rules.",
            "Limit database user privileges to the minimum required.",
            "Regularly audit and patch your web application code.",
        ],
    },
    "Heartbleed": {
        "description": "Heartbleed (CVE-2014-0160) - Critical OpenSSL vulnerability being exploited.",
        "advice": [
            "Immediately patch OpenSSL to a non-vulnerable version.",
            "Revoke and reissue all SSL/TLS certificates.",
            "Change all passwords and session tokens.",
            "Audit your server for signs of data exfiltration.",
        ],
    },
    "Infiltration": {
        "description": "Network Infiltration - An attacker has gained access to your internal network.",
        "advice": [
            "Isolate all affected systems immediately.",
            "Conduct a full forensic investigation of network logs.",
            "Change all internal credentials and revoke active sessions.",
            "Engage your incident response team or a cybersecurity professional.",
        ],
    },
    "PortScan": {
        "description": "Port Scan - Someone is probing your open ports to identify attack surfaces.",
        "advice": [
            "Close all unnecessary open ports via your firewall.",
            "Enable stealth mode on your firewall to not respond to unsolicited probes.",
            "Monitor for follow-up exploitation attempts.",
            "Ensure no sensitive services are exposed to the internet.",
        ],
    },
    "DDoS": {
        "description": "Distributed Denial of Service (DDoS) - Multiple sources are flooding your network.",
        "advice": [
            "Enable DDoS protection through your ISP or a cloud service (Cloudflare, AWS Shield).",
            "Implement traffic scrubbing / filtering at the network edge.",
            "Contact your ISP immediately to assist with upstream filtering.",
            "Temporarily increase your bandwidth capacity if available.",
        ],
    },
    "UNKNOWN": {
        "description": "Unknown Anomaly - Unusual network behaviour detected that doesn't match known patterns.",
        "advice": [
            "Monitor your network traffic closely for the next few minutes.",
            "Check running processes on your device for anything suspicious.",
            "Consider disconnecting from the network temporarily if behaviour continues.",
            "Update your antivirus and run a full system scan.",
        ],
    },
}


# -

class IDSInferenceEngine:
    """
    Loads all saved models and runs the full 2-stage inference pipeline.
    Mirrors the preprocessing from the training notebook exactly.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._load_models()

    def _load_models(self):
        """Load all model artifacts from disk."""
        print("[IDS] Loading models...")

        d = self.model_dir

        # Preprocessing artifacts
        self.scaler        = joblib.load(os.path.join(d, "scaler.pkl"))
        self.nzv_cols      = joblib.load(os.path.join(d, "nzv_cols.pkl"))
        self.feature_cols  = joblib.load(os.path.join(d, "feature_cols.pkl"))

        # Score normalisation stats (combined dict: if_median, if_iqr, ae_median, ae_iqr)
        score_stats        = joblib.load(os.path.join(d, "score_stats.pkl"))
        self.if_median     = score_stats["if_median"]
        self.if_iqr        = score_stats["if_iqr"]
        self.ae_median     = score_stats["ae_median"]
        self.ae_iqr        = score_stats["ae_iqr"]

        # Stage 1 models
        self.if_model      = joblib.load(os.path.join(d, "isolation_forest.pkl"))
        # Load autoencoder as ONNX session (replaces TensorFlow)
        self.autoencoder   = ort.InferenceSession(
            os.path.join(d, "autoencoder.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self._ae_input_name  = self.autoencoder.get_inputs()[0].name
        self._ae_output_name = self.autoencoder.get_outputs()[0].name
        self.detector      = joblib.load(os.path.join(d, "xgb_detector.pkl"))
        self.threshold     = joblib.load(os.path.join(d, "best_threshold.pkl"))

        # Stage 2 model
        self.atk_classifier = joblib.load(os.path.join(d, "attack_label_encoder.pkl"))
        # Note: attack_label_encoder.pkl is the LabelEncoder.
        # The XGBoost attack classifier is also saved - check for both names.
        atk_clf_path = os.path.join(d, "attack_classifier_xgb.pkl")
        if not os.path.exists(atk_clf_path):
            atk_clf_path = os.path.join(d, "attack_classifier.pkl")
        if os.path.exists(atk_clf_path):
            self.atk_model = joblib.load(atk_clf_path)
        else:
            self.atk_model = None
            print("[IDS] Warning: Attack classifier model not found. Stage 2 will return UNKNOWN.")

        self.label_encoder = self.atk_classifier  # alias for clarity

        print("[IDS] - All models loaded successfully.")
        print(f"[IDS]    Decision threshold : {self.threshold:.4f}")
        print(f"[IDS]    NZV cols to drop   : {len(self.nzv_cols)}")
        print(f"[IDS]    Feature columns    : {len(self.feature_cols)}")

    # - Preprocessing (mirrors clean_chunk + add_engineered_features) -

    @staticmethod
    def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Exactly mirrors add_engineered_features() from the training notebook.
        Must produce the same columns in the same order.
        """
        eps = 1e-8

        if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
            df["fwd_bwd_pkt_ratio"] = (
                df["Total Fwd Packets"] / (df["Total Backward Packets"] + eps)
            )

        if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
            df["fwd_bwd_byte_ratio"] = (
                df["Total Length of Fwd Packets"] / (df["Total Length of Bwd Packets"] + eps)
            )

        for col in [
            "Flow Duration",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
        ]:
            if col in df.columns:
                safe_col = col.replace("/", "_per_")
                df[f"log1p_{safe_col}"] = np.log1p(np.clip(df[col].values, 0, None))

        if "Packet Length Mean" in df.columns and "Packet Length Std" in df.columns:
            df["pkt_len_cv"] = df["Packet Length Std"] / (df["Packet Length Mean"] + eps)

        if "Active Mean" in df.columns and "Idle Mean" in df.columns:
            df["active_idle_ratio"] = df["Active Mean"] / (df["Idle Mean"] + eps)

        if "PSH Flag Count" in df.columns and "Total Fwd Packets" in df.columns:
            df["psh_urg_density"] = (
                (df.get("PSH Flag Count", 0) + df.get("URG Flag Count", 0))
                / (df["Total Fwd Packets"] + eps)
            )

        return df

    def _preprocess(self, flow: dict) -> np.ndarray:
        """
        Convert a raw flow dict - scaled numpy array ready for IF/AE.
        Mirrors clean_chunk() from the training notebook.
        """
        df = pd.DataFrame([flow])
        df.columns = df.columns.str.strip()

        # Drop identifier columns if present
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

        # Remove label column if accidentally included
        df.drop(columns=["Label"], errors="ignore", inplace=True)

        # Handle infinity
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Ensure all numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Drop near-zero-variance columns (identified during training)
        df.drop(columns=[c for c in self.nzv_cols if c in df.columns], inplace=True)

        # Feature engineering
        df = self._add_engineered_features(df)

        # Fill NaNs introduced by engineering
        df.fillna(0, inplace=True)

        # Align columns to training feature set
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0  # missing feature - zero
        df = df[self.feature_cols]  # enforce exact column order

        # Scale
        X_scaled = self.scaler.transform(df)
        X_scaled = np.clip(X_scaled, -10, 10).astype(np.float32)

        return X_scaled

    # - Hybrid feature builder (mirrors build_hybrid_features()) -

    def _run_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Run ONNX autoencoder inference."""
        return self.autoencoder.run(
            [self._ae_output_name],
            {self._ae_input_name: X.astype(np.float32)}
        )[0]

    def _build_hybrid_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Appends IF anomaly score + AE reconstruction error to scaled features.
        Mirrors build_hybrid_features() from the training notebook exactly.
        """
        # IsolationForest score (higher = more anomalous)
        if_score = -self.if_model.score_samples(X_scaled)
        if_norm  = (if_score - self.if_median) / self.if_iqr

        # Autoencoder reconstruction error via ONNX
        ae_pred  = self._run_autoencoder(X_scaled)
        ae_error = np.mean((X_scaled - ae_pred) ** 2, axis=1)
        ae_norm  = (ae_error - self.ae_median) / self.ae_iqr
        ae_log   = np.log1p(ae_error)

        return np.hstack([
            X_scaled,
            if_norm.reshape(-1, 1),
            ae_norm.reshape(-1, 1),
            ae_log.reshape(-1, 1),
        ]).astype(np.float32)

    # - Main prediction entry point -

    def predict(self, flow: dict) -> dict:
        """
        Run the full 2-stage inference pipeline on a single flow.

        Args:
            flow: dict with CICIDS flow feature names as keys.

        Returns:
            {
                "is_anomaly"   : bool,
                "confidence"   : float (0-1, probability of attack),
                "attack_type"  : str or None,
                "description"  : str,
                "advice"       : list[str],
                "if_score"     : float,
                "ae_error"     : float,
            }
        """
        # Step 1: Preprocess
        X_scaled = self._preprocess(flow)

        # Step 2: Build hybrid features
        X_hybrid = self._build_hybrid_features(X_scaled)

        # Step 3: Stage 1 - binary detection
        prob        = self.detector.predict_proba(X_hybrid)[0, 1]
        is_anomaly  = bool(prob >= self.threshold)

        # Raw IF and AE scores for UI display
        if_score_raw = float(-self.if_model.score_samples(X_scaled)[0])
        ae_pred_raw  = self._run_autoencoder(X_scaled)
        ae_error_raw = float(np.mean((X_scaled - ae_pred_raw) ** 2))

        if not is_anomaly:
            return {
                "is_anomaly"  : False,
                "confidence"  : float(prob),
                "attack_type" : None,
                "description" : "Traffic appears normal. No anomaly detected.",
                "advice"      : [],
                "if_score"    : if_score_raw,
                "ae_error"    : ae_error_raw,
            }

        # Step 4: Stage 2 - attack classification
        attack_type = "UNKNOWN"
        if self.atk_model is not None:
            try:
                atk_pred    = self.atk_model.predict(X_hybrid)[0]
                attack_type = self.label_encoder.inverse_transform([atk_pred])[0]
            except Exception as e:
                print(f"[IDS] Stage 2 classification failed: {e}")
                attack_type = "UNKNOWN"

        # Fetch remediation info
        info = ATTACK_ADVICE.get(attack_type, ATTACK_ADVICE["UNKNOWN"])

        return {
            "is_anomaly"  : True,
            "confidence"  : float(prob),
            "attack_type" : attack_type,
            "description" : info["description"],
            "advice"      : info["advice"],
            "if_score"    : if_score_raw,
            "ae_error"    : ae_error_raw,
        }

    def predict_batch(self, flows: list[dict]) -> list[dict]:
        """Run prediction on a batch of flows (more efficient than looping predict())."""
        if not flows:
            return []

        # Preprocess all
        X_scaled_list = [self._preprocess(f) for f in flows]
        X_scaled_all  = np.vstack(X_scaled_list)

        # Hybrid features
        X_hybrid_all  = self._build_hybrid_features(X_scaled_all)

        # Stage 1
        probs    = self.detector.predict_proba(X_hybrid_all)[:, 1]
        is_atk   = probs >= self.threshold

        # Stage 2 - only on detected attacks
        atk_indices = np.where(is_atk)[0]
        atk_types   = ["UNKNOWN"] * len(flows)

        if len(atk_indices) > 0 and self.atk_model is not None:
            try:
                X_atk       = X_hybrid_all[atk_indices]
                atk_preds   = self.atk_model.predict(X_atk)
                atk_labels  = self.label_encoder.inverse_transform(atk_preds)
                for idx, label in zip(atk_indices, atk_labels):
                    atk_types[idx] = label
            except Exception as e:
                print(f"[IDS] Batch Stage 2 failed: {e}")

        results = []
        for i, (flow, prob, anomaly) in enumerate(zip(flows, probs, is_atk)):
            if not anomaly:
                results.append({
                    "is_anomaly"  : False,
                    "confidence"  : float(prob),
                    "attack_type" : None,
                    "description" : "Traffic appears normal.",
                    "advice"      : [],
                })
            else:
                atype = atk_types[i]
                info  = ATTACK_ADVICE.get(atype, ATTACK_ADVICE["UNKNOWN"])
                results.append({
                    "is_anomaly"  : True,
                    "confidence"  : float(prob),
                    "attack_type" : atype,
                    "description" : info["description"],
                    "advice"      : info["advice"],
                })

        return results


# - Quick sanity-check (run this file directly to test model loading) -

if __name__ == "__main__":
    import json

    engine = IDSInferenceEngine(model_dir="models")

    # Create a dummy all-zero flow to test the pipeline end-to-end
    dummy_flow = {col: 0.0 for col in engine.feature_cols}

    print("\n[TEST] Running prediction on dummy zero-flow...")
    result = engine.predict(dummy_flow)
    print(json.dumps(result, indent=2))
    print("\n- Inference pipeline is working correctly.")