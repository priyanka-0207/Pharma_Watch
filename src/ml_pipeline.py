import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, auc
)
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings("ignore")


class DrugSeverityPredictor:
    def __init__(self, data_path="C:/Users/priya/drug_safety_intelligence/result/processed_drug_safety.csv"):
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.results = {}
    
    def prepare_features(self):
        feature_cols = [
            "reaction_count",
            "reaction_frequency_ratio",
            "log_reaction_count",
            "drug_total_reports",
            "drug_reaction_diversity",
            "reaction_idf",
            "reaction_tfidf",
            "is_drug_specific_reaction",
        ]
        
        for col in ["drug_category"]:
            le = LabelEncoder()
            self.df[f"{col}_encoded"] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            feature_cols.append(f"{col}_encoded")

        le_reaction = LabelEncoder()
        self.df["reaction_encoded"] = le_reaction.fit_transform(self.df["reaction"])
        self.label_encoders["reaction"] = le_reaction

        X = self.df[feature_cols].fillna(0)
        y = self.df["is_high_severity"]
        
        print(f" Features: {X.shape[1]} | Samples: {X.shape[0]}")
        print(f" Class balance: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=5, subsample=0.8, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                eval_metric="logloss", random_state=42
            ),
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\n" + "=" * 55)
        print("  MODEL COMPARISON (5-Fold Stratified CV)")
        print("=" * 55)
        
        best_model_name = None
        best_auc = 0
        
        for name, model in models.items():
            y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
            y_proba = cross_val_predict(
                model, X_scaled, y, cv=cv, method="predict_proba"
            )[:, 1]

            roc = roc_auc_score(y, y_proba)
            precision, recall, _ = precision_recall_curve(y, y_proba)
            pr_auc = auc(recall, precision)
            
            report = classification_report(y, y_pred, output_dict=True)
            
            self.results[name] = {
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr_auc, 4),
                "precision_high_sev": round(
                    report.get("1", {}).get("precision", 0), 4
                ),
                "recall_high_sev": round(
                    report.get("1", {}).get("recall", 0), 4
                ),
                "f1_high_sev": round(
                    report.get("1", {}).get("f1-score", 0), 4
                ),
            }
            
            print(f"\n {name}")
            print(f"   ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")
            print(f"   High-Severity → P: {report['1']['precision']:.3f} "
                  f"R: {report['1']['recall']:.3f} "
                  f"F1: {report['1']['f1-score']:.3f}")
            
            if pr_auc > best_auc:
                best_auc = pr_auc
                best_model_name = name
        
        print(f"\n Best Model: {best_model_name} (PR-AUC: {best_auc:.4f})")
  
        best_model = models[best_model_name]
        best_model.fit(X_scaled, y)
        
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(
                best_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            print(f"\n Top Feature Importances ({best_model_name}):")
            for feat, imp in importances.head(7).items():
                print(f"   {feat}: {imp:.4f}")
        
        return best_model, scaler, self.results
    
    def save_results(self, output_path="C:/Users/priya/drug_safety_intelligence/data/model_results.json"):
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n Results saved to {output_path}")
    
    def run(self):
        print("=" * 55)
        print("  DRUG SEVERITY PREDICTION PIPELINE")
        print("=" * 55)
        X, y = self.prepare_features()
        model, scaler, results = self.train_and_evaluate(X, y)
        self.save_results()
        return model, scaler, results


if __name__ == "__main__":
    predictor = DrugSeverityPredictor()
    predictor.run()