import pandas as pd
import numpy as np
import json
from pathlib import Path


class DrugSafetyETL: 
    def __init__(self, raw_data_path="C:/Users/priya/drug_safety_intelligence/data/raw_adverse_events.json"):
        self.raw_data_path = raw_data_path
        self.drug_categories = {
            "aspirin": "Pain/Anti-inflammatory",
            "ibuprofen": "Pain/Anti-inflammatory",
            "acetaminophen": "Pain",
            "metformin": "Diabetes",
            "lisinopril": "Blood Pressure",
            "atorvastatin": "Cholesterol",
            "omeprazole": "GI/Acid Reflux",
            "amoxicillin": "Antibiotic",
            "sertraline": "Antidepressant",
            "fluoxetine": "Antidepressant",
            "loratadine": "Allergy",
            "metoprolol": "Cardiovascular",
            "gabapentin": "Nerve Pain/Neurological",
            "prednisone": "Steroid/Immunosuppressant",
            "ciprofloxacin": "Antibiotic",
            "warfarin": "Blood Thinner",
            "albuterol": "Respiratory",
            "hydrochlorothiazide": "Diuretic",
            "tramadol": "Opioid Analgesic",
            "duloxetine": "Antidepressant",
        }
    
    def extract(self):
        with open(self.raw_data_path, "r") as f:
            raw = json.load(f)
        print(f" Extracted data for {len(raw)} drugs")
        return raw
    
    def _parse_tool_results(self, tool_data):
        result = {}
        if isinstance(tool_data, list):
            for entry in tool_data:
                term = entry.get("term", "").lower().strip()
                count = entry.get("count", 0)
                if term:
                    result[term] = count
        return result
    
    def transform_reactions(self, raw_data):
        rows = []
        for drug, data in raw_data.items():
            reactions = self._parse_tool_results(
                data.get("FAERS_count_reactions_by_drug_event", [])
            )
            total_reports = sum(reactions.values()) if reactions else 0
            
            for reaction, count in reactions.items():
                row = {
                    "drug": drug,
                    "drug_category": self.drug_categories.get(drug, "Other"),
                    "reaction": reaction,
                    "reaction_count": count,
                    "reaction_frequency_ratio": (
                        count / total_reports if total_reports > 0 else 0
                    ),
                    "log_reaction_count": np.log1p(count),
                    "drug_total_reports": total_reports,
                    "drug_reaction_diversity": len(reactions),
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f" Transformed: {len(df)} drug-reaction pairs")
        return df
    
    def engineer_severity_features(self, df, raw_data):
        severity_map = {}
        for drug, data in raw_data.items():
            seriousness = self._parse_tool_results(
                data.get("FAERS_count_seriousness_by_drug_event", [])
            )
            outcomes = self._parse_tool_results(
                data.get("FAERS_count_outcomes_by_drug_event", [])
            )
            
            total = sum(seriousness.values()) if seriousness else 1

            serious_count = (
                seriousness.get("1", 0) or 
                seriousness.get("serious", 0)
            )
            
            severity_map[drug] = {
                "serious_ratio": serious_count / total if total > 0 else 0,
                "death_rate": (
                    outcomes.get("death", 0) / total if total > 0 else 0
                ),
                "hospitalization_rate": (
                    outcomes.get("hospitalization", 0) / total if total > 0 else 0
                ),
            }
        
        severity_df = pd.DataFrame(severity_map).T
        severity_df.index.name = "drug"
        severity_df = severity_df.reset_index()
        df = df.merge(severity_df, on="drug", how="left")
        reaction_seriousness = df.groupby("reaction")["serious_ratio"].mean()
        median_reaction_seriousness = reaction_seriousness.median()
        df["is_high_severity"] = df["reaction"].map(
            lambda r: 1 if reaction_seriousness.get(r, 0) > median_reaction_seriousness else 0
        )
        
        print(f" Engineered severity features | "
              f"High severity drugs: {df['is_high_severity'].sum()} / {len(df)}")
        return df
    
    def engineer_reaction_features(self, df):
        reaction_drug_count = df.groupby("reaction")["drug"].nunique()
        total_drugs = df["drug"].nunique()
        idf_scores = np.log(total_drugs / reaction_drug_count)
        df["reaction_idf"] = df["reaction"].map(idf_scores)

        df["reaction_tfidf"] = df["reaction_frequency_ratio"] * df["reaction_idf"]
        
        df["is_drug_specific_reaction"] = df["reaction"].map(
            lambda r: 1 if reaction_drug_count.get(r, 0) <= 2 else 0
        )
        
        print(f" Engineered reaction features | "
              f"Unique reactions: {df['reaction'].nunique()}")
        return df
    
    def load(self, df, output_path="C:/Users/priya/drug_safety_intelligence/result/processed_drug_safety.csv"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f" Saved processed data: {output_path} "
              f"({df.shape[0]} rows × {df.shape[1]} columns)")
        return df
    
    def run(self):
        print("=" * 55)
        print("  DRUG SAFETY ETL PIPELINE")
        print("=" * 55)
        raw = self.extract()
        df = self.transform_reactions(raw)
        df = self.engineer_severity_features(df, raw)
        df = self.engineer_reaction_features(df)
        df = self.load(df)
        
        print(f"\n Final dataset shape: {df.shape}")
        print(f"   Features: {list(df.columns)}")
        return df


if __name__ == "__main__":
    pipeline = DrugSafetyETL()
    df = pipeline.run()