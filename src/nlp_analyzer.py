import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


class DrugLabelNLPAnalyzer:
    def __init__(self, labels_path="C:/Users/priya/drug_safety_intelligence/data/raw_drug_labels.json"):
        with open(labels_path, "r", encoding="utf-8") as f:
            self.raw_labels = json.load(f)
    
    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text
    
    def extract_documents(self):
        tool_keys = [
            "FDA_get_adverse_reactions_by_drug_name",
            "FDA_get_warnings_by_drug_name",
            "FDA_get_drug_interactions_by_drug_name",
        ]
        
        documents = {}
        for drug, sections in self.raw_labels.items():
            parts = []
            for key in tool_keys:
                text_data = sections.get(key, [])
                if isinstance(text_data, list):
                    text = " ".join(str(item) for item in text_data)
                elif isinstance(text_data, str):
                    text = text_data
                elif isinstance(text_data, dict):
                    text = " ".join(str(v) for v in text_data.values())
                else:
                    text = str(text_data) if text_data else ""
                
                cleaned = self.clean_text(text)
                if cleaned:
                    parts.append(cleaned)
            
            combined = " ".join(parts).strip()
            if combined:
                documents[drug] = combined
        
        print(f" Extracted label text for {len(documents)} drugs")
        for drug, text in documents.items():
            word_count = len(text.split())
            print(f"   {drug}: {word_count} words")
        
        return documents
    
    def compute_tfidf_keywords(self, documents, top_n=10):
        drug_names = list(documents.keys())
        texts = [documents[d] for d in drug_names]
        
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=1,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        keywords_per_drug = {}
        for i, drug in enumerate(drug_names):
            scores = tfidf_matrix[i].toarray().flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [
                {"keyword": feature_names[idx], "score": round(scores[idx], 4)}
                for idx in top_indices if scores[idx] > 0
            ]
            keywords_per_drug[drug] = keywords
        
        print(f" Extracted top {top_n} TF-IDF keywords per drug")
        return keywords_per_drug, tfidf_matrix, vectorizer
    
    def compute_label_similarity(self, tfidf_matrix, drug_names):
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=drug_names, columns=drug_names)
        sim_df.to_csv("C:/Users/priya/drug_safety_intelligence/result/label_text_similarity.csv")
        print(f" Label text similarity matrix saved")
        return sim_df
    
    def extract_risk_phrases(self, documents):
        risk_patterns = [
            r"black box warning",
            r"life.?threatening",
            r"fatal",
            r"death",
            r"discontinue immediately",
            r"serious allergic",
            r"anaphylaxis",
            r"liver damage",
            r"kidney (failure|damage|impairment)",
            r"cardiac arrest",
            r"suicidal (thoughts|ideation|behavior)",
            r"seizure",
            r"bleeding",
            r"stroke",
        ]
        
        risk_flags = {}
        for drug, text in documents.items():
            flags = {}
            for pattern in risk_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    flags[pattern] = len(matches)
            risk_flags[drug] = flags
        risk_scores = {
            drug: len(flags) for drug, flags in risk_flags.items()
        }
        
        print(f"\n Risk Pattern Analysis:")
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        for drug, score in sorted_risks[:10]:
            patterns = list(risk_flags[drug].keys())
            print(f"   {drug}: {score} risk patterns → {patterns[:3]}")
        
        return risk_flags, risk_scores
    
    def run(self):
        print("=" * 55)
        print("  DRUG LABEL NLP ANALYSIS")
        print("=" * 55)
        
        documents = self.extract_documents()
        keywords, tfidf_matrix, vectorizer = self.compute_tfidf_keywords(documents)
        drug_names = list(documents.keys())
        sim_df = self.compute_label_similarity(tfidf_matrix, drug_names)
        risk_flags, risk_scores = self.extract_risk_phrases(documents)

        nlp_results = {
            "keywords_per_drug": keywords,
            "risk_scores": risk_scores,
            "risk_flags": {
                drug: {k: v for k, v in flags.items()} 
                for drug, flags in risk_flags.items()
            },
        }
        with open("C:/Users/priya/drug_safety_intelligence/data/nlp_analysis.json", "w") as f:
            json.dump(nlp_results, f, indent=2)
        
        print(f"\n NLP analysis saved to results/nlp_analysis.json")
        return nlp_results


if __name__ == "__main__":
    analyzer = DrugLabelNLPAnalyzer()
    analyzer.run()