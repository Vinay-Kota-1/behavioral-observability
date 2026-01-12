"""
Business Mapper - Map Anomalies to Business Impact

This module reads business rules from adversarial_config.yaml and maps
detected anomalies to their business impact, recommended actions, and
escalation paths.

Usage:
    from business_mapper import BusinessMapper
    
    mapper = BusinessMapper()
    impact = mapper.get_impact(source_dataset="CERT", entity_type="user")
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class BusinessImpact:
    """Business impact details for an anomaly."""
    scenario_name: str
    category: str
    risk_level: str
    estimated_cost: float
    cost_unit: str  # "per_incident", "per_minute", "per_day"
    recommended_action: str
    compliance_flags: List[str]
    escalation_path: str
    escalation_sla_minutes: int
    escalation_contact: str
    
    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario_name,
            "category": self.category,
            "risk_level": self.risk_level,
            "estimated_cost": self.estimated_cost,
            "cost_unit": self.cost_unit,
            "recommended_action": self.recommended_action,
            "compliance_flags": self.compliance_flags,
            "escalation_path": self.escalation_path,
            "escalation_sla_minutes": self.escalation_sla_minutes,
            "escalation_contact": self.escalation_contact
        }
    
    def format_cost(self) -> str:
        """Format cost as readable string."""
        if self.cost_unit == "per_minute":
            return f"${self.estimated_cost:.0f}/minute"
        elif self.cost_unit == "per_day":
            return f"${self.estimated_cost:.0f}/day"
        else:
            return f"${self.estimated_cost:.0f}/incident"
    
    def get_severity_color(self) -> str:
        """Get color code for severity."""
        colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#28a745"
        }
        return colors.get(self.risk_level, "#6c757d")


class BusinessMapper:
    """
    Maps anomalies to business impact using rules from adversarial_config.yaml.
    """
    
    def __init__(self, config_path: str = "adversarial_config.yaml"):
        self.config_path = config_path
        self.scenarios: Dict[str, Dict] = {}
        self.escalation_paths: Dict[str, Dict] = {}
        
        self._load_config()
    
    def _load_config(self):
        """Load business rules from config file."""
        if not Path(self.config_path).exists():
            print(f"Warning: Config not found at {self.config_path}")
            return
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        self.scenarios = config.get("scenarios", {})
        self.escalation_paths = config.get("escalation_paths", {})
        
        print(f"Loaded {len(self.scenarios)} scenarios, {len(self.escalation_paths)} escalation paths")
    
    def get_matching_scenario(
        self, 
        source_dataset: str, 
        entity_type: str
    ) -> Optional[str]:
        """Find the scenario matching given source and entity type."""
        for name, scenario in self.scenarios.items():
            if (scenario.get("source_dataset") == source_dataset and 
                scenario.get("entity_type") == entity_type):
                return name
        return None
    
    def get_impact(
        self, 
        source_dataset: str, 
        entity_type: str
    ) -> Optional[BusinessImpact]:
        """Get business impact for anomaly."""
        scenario_name = self.get_matching_scenario(source_dataset, entity_type)
        if not scenario_name:
            return None
        
        scenario = self.scenarios[scenario_name]
        impact_config = scenario.get("business_impact", {})
        
        if not impact_config:
            return None
        
        # Get escalation details
        esc_path = impact_config.get("escalation_path", "")
        esc_details = self.escalation_paths.get(esc_path, {})
        
        # Determine cost and unit
        cost = 0.0
        cost_unit = "per_incident"
        for key in ["estimated_cost_per_incident", "estimated_cost_per_minute", "estimated_cost_per_day"]:
            if key in impact_config:
                cost = impact_config[key]
                cost_unit = key.replace("estimated_cost_", "")
                break
        
        return BusinessImpact(
            scenario_name=scenario_name,
            category=impact_config.get("category", "Unknown"),
            risk_level=impact_config.get("risk_level", "medium"),
            estimated_cost=cost,
            cost_unit=cost_unit,
            recommended_action=impact_config.get("recommended_action", "Investigate"),
            compliance_flags=impact_config.get("compliance_flags", []),
            escalation_path=esc_path,
            escalation_sla_minutes=esc_details.get("response_sla_minutes", 60),
            escalation_contact=esc_details.get("contact", "unknown")
        )
    
    def get_all_impacts(self) -> Dict[str, BusinessImpact]:
        """Get all configured business impacts."""
        impacts = {}
        for name, scenario in self.scenarios.items():
            impact_config = scenario.get("business_impact", {})
            if impact_config:
                source = scenario.get("source_dataset", "")
                entity = scenario.get("entity_type", "")
                impact = self.get_impact(source, entity)
                if impact:
                    impacts[name] = impact
        return impacts
    
    def format_impact_report(self, impact: BusinessImpact) -> str:
        """Format impact as readable text."""
        lines = [
            f"Scenario: {impact.scenario_name}",
            f"Category: {impact.category}",
            f"Risk Level: {impact.risk_level.upper()}",
            f"Estimated Cost: {impact.format_cost()}",
            f"",
            f"Recommended Action:",
            f"  {impact.recommended_action}",
            f"",
            f"Compliance: {', '.join(impact.compliance_flags) if impact.compliance_flags else 'None'}",
            f"Escalation: {impact.escalation_path} (SLA: {impact.escalation_sla_minutes} min)",
            f"Contact: {impact.escalation_contact}"
        ]
        return "\n".join(lines)
    
    def get_risk_summary(self) -> Dict[str, int]:
        """Get count of scenarios by risk level."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for impact in self.get_all_impacts().values():
            level = impact.risk_level
            if level in summary:
                summary[level] += 1
        return summary


# CLI
if __name__ == "__main__":
    mapper = BusinessMapper()
    
    print("\n" + "="*60)
    print("CONFIGURED BUSINESS IMPACTS")
    print("="*60)
    
    for name, impact in mapper.get_all_impacts().items():
        print(f"\n{mapper.format_impact_report(impact)}")
        print("-"*40)
    
    print("\nRisk Summary:", mapper.get_risk_summary())
