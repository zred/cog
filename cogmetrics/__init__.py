"""
CogMetrics: A Comprehensive Library for Measuring Consciousness Emergence

This library provides empirical tools for measuring consciousness-like properties
in recursive AI systems, including information-theoretic measures, behavioral
tests, and network analysis of self-referential structures.
"""

from .information_measures import InformationMetrics
from .behavioral_tests import BehavioralTests
from .network_analysis import NetworkAnalysis
from .temporal_coherence import TemporalCoherence
from .metacognition import MetacognitionMetrics
from .consciousness_dashboard import ConsciousnessDashboard
from .statistical_validation import StatisticalValidator

__version__ = "0.1.0"
__all__ = [
    "InformationMetrics",
    "BehavioralTests",
    "NetworkAnalysis",
    "TemporalCoherence",
    "MetacognitionMetrics",
    "ConsciousnessDashboard",
    "StatisticalValidator"
]

