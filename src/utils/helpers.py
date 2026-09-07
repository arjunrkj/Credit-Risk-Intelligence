from typing import Dict, List, Tuple
from src.utils.config import LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

def calculate_risk_score(probability: float) -> int:
    """
    Translates default probability (0.0 to 1.0) into a standard credit score (0 to 1000).
    Higher score indicates lower risk / better creditworthiness.
    """
    prob = max(0.0, min(1.0, float(probability)))
    return int(round(1000 * (1.0 - prob)))

def get_risk_band(probability: float) -> str:
    """Categorizes default probability into Risk Bands: Low, Medium, or High."""
    prob = float(probability)
    if prob < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif prob < MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"

def get_risk_color(risk_band: str) -> str:
    """Returns CSS color code corresponding to Risk Band."""
    band = risk_band.strip().title()
    if "Low" in band:
        return "#28A745"  # Green
    elif "Medium" in band:
        return "#FFC107"  # Yellow / Amber
    else:
        return "#DC3545"  # Red

def format_currency(amount: float) -> str:
    """Formats numeric value to currency representation."""
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"

def derive_business_rules(feature_values: Dict[str, float], shap_values: Dict[str, float]) -> List[Dict[str, str]]:
    """
    Translates raw feature values and SHAP contribution scores into human-readable 
    business decision rules for loan officers and audit compliance.
    """
    rules = []
    
    # Sort features by absolute SHAP impact
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    feature_labels = {
        'EXT_SOURCE_2': 'External Credit Score (Bureau 2)',
        'EXT_SOURCE_3': 'External Credit Score (Bureau 3)',
        'CREDIT_TO_INCOME_RATIO': 'Credit Amount to Income Ratio',
        'ANNUITY_TO_INCOME_RATIO': 'Monthly Debt Annuity to Income Ratio',
        'DAYS_EMPLOYED_YEARS': 'Employment Tenure (Years)',
        'AGE_YEARS': 'Applicant Age',
        'ACTIVE_BUREAU_LOANS': 'Active Loans in Credit Bureau',
        'PREV_REFUSED_COUNT': 'Previous Loan Refusal Count'
    }

    for feat, shap_val in sorted_features[:5]:
        val = feature_values.get(feat, 0)
        readable_name = feature_labels.get(feat, feat.replace('_', ' ').title())
        impact_dir = "Increased Risk" if shap_val > 0 else "Reduced Risk"
        severity = "High" if abs(shap_val) > 0.10 else "Moderate"
        
        # Rule translation logic
        if feat == 'EXT_SOURCE_2' and val < 0.3:
            desc = f"External credit bureau rating is significantly low ({val:.2f})."
        elif feat == 'CREDIT_TO_INCOME_RATIO' and val > 4.0:
            desc = f"Requested loan amount is {val:.1f}x annual income (High debt burden)."
        elif feat == 'ANNUITY_TO_INCOME_RATIO' and val > 0.25:
            desc = f"Monthly loan payment exceeds {val*100:.1f}% of total monthly income."
        elif feat == 'DAYS_EMPLOYED_YEARS' and val < 2.0:
            desc = f"Short employment duration of {val:.1f} years."
        elif feat == 'PREV_REFUSED_COUNT' and val >= 1:
            desc = f"History of {int(val)} previously refused loan application(s)."
        elif shap_val < 0:
            desc = f"Favorable factor: {readable_name} ({val:.2f}) positively supports application."
        else:
            desc = f"{readable_name} value ({val:.2f}) shifts risk assessment."
            
        rules.append({
            "feature": readable_name,
            "impact": impact_dir,
            "shap_score": f"{shap_val:+.4f}",
            "rule_description": desc,
            "severity": severity
        })
        
    return rules
