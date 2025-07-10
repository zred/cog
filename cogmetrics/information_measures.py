"""
Information-theoretic measures for consciousness emergence.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Heavy optional dependencies are disabled by default for lightweight installs
PYPHI_AVAILABLE = False
JIDT_AVAILABLE = False

@dataclass
class InformationProfile:
    """Results from information-theoretic analysis"""
    phi_value: Optional[float] = None
    transfer_entropy: Optional[float] = None
    active_information_storage: Optional[float] = None
    information_modification: Optional[float] = None
    lempel_ziv_complexity: Optional[float] = None
    logical_depth: Optional[float] = None
    timestamp: float = 0.0
    agent_id: str = ""

class InformationMetrics:
    """Comprehensive information-theoretic consciousness measures"""
    
    def __init__(self, enable_phi: bool = True, enable_jidt: bool = True):
        self.enable_phi = enable_phi and PYPHI_AVAILABLE
        self.enable_jidt = enable_jidt and JIDT_AVAILABLE
    
    
    def calculate_phi(self, network_state: np.ndarray, 
                      connectivity_matrix: np.ndarray) -> float:
        """Calculate Integrated Information Theory phi value"""
        if not self.enable_phi:
            return None
            
        try:
            # Convert to PyPhi network format
            network = pyphi.Network(connectivity_matrix)
            state = tuple(network_state.astype(int))
            subsystem = pyphi.Subsystem(network, state)
            
            # Calculate phi
            phi = pyphi.compute.phi(subsystem)
            return float(phi)
        except Exception as e:
            warnings.warn(f"Phi calculation failed: {e}")
            return None
    
    def calculate_transfer_entropy(self, source_series: np.ndarray, 
                                 target_series: np.ndarray) -> float:
        """Calculate transfer entropy between time series"""
        if not self.enable_jidt:
            return self._fallback_transfer_entropy(source_series, target_series)
        
        try:
            calc = self.TransferEntropyCalculator()
            calc.initialise(1)  # history length
            calc.setObservations(source_series, target_series)
            return float(calc.computeAverageLocalOfObservations())
        except Exception as e:
            warnings.warn(f"Transfer entropy calculation failed: {e}")
            return self._fallback_transfer_entropy(source_series, target_series)
    
    def _fallback_transfer_entropy(self, source: np.ndarray, 
                                  target: np.ndarray) -> float:
        """Fallback transfer entropy using mutual information"""
        # Simple approximation using mutual information
        from sklearn.feature_selection import mutual_info_regression
        
        # Shift target by 1 for temporal dependency
        if len(target) > 1:
            target_shifted = target[1:]
            source_trimmed = source[:-1]
            
            # Calculate MI(source_t, target_t+1)
            mi_forward = mutual_info_regression(
                source_trimmed.reshape(-1, 1), target_shifted
            )[0]
            
            # Calculate MI(target_t, target_t+1) for normalization
            target_past = target[:-1]
            mi_self = mutual_info_regression(
                target_past.reshape(-1, 1), target_shifted
            )[0]
            
            # Transfer entropy approximation
            return max(0, mi_forward - mi_self)
        return 0.0
    
    def calculate_active_information_storage(self, time_series: np.ndarray) -> float:
        """Calculate how much information system stores about itself"""
        if not self.enable_jidt:
            return self._fallback_information_storage(time_series)
        
        try:
            calc = self.ActiveInfoStorageCalculator()
            calc.initialise(1)  # history length
            calc.setObservations(time_series)
            return float(calc.computeAverageLocalOfObservations())
        except Exception as e:
            warnings.warn(f"Active information storage calculation failed: {e}")
            return self._fallback_information_storage(time_series)
    
    def _fallback_information_storage(self, time_series: np.ndarray) -> float:
        """Fallback information storage using autocorrelation"""
        if len(time_series) < 2:
            return 0.0
        
        # Use autocorrelation as proxy for information storage
        autocorr = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
        return max(0, autocorr)
    
    def calculate_lempel_ziv_complexity(self, binary_sequence: np.ndarray) -> float:
        """Calculate Lempel-Ziv complexity"""
        if len(binary_sequence) == 0:
            return 0.0
        
        # Convert to binary string
        binary_str = ''.join(binary_sequence.astype(int).astype(str))
        
        # Lempel-Ziv complexity algorithm
        complexity = 0
        i = 0
        while i < len(binary_str):
            j = i + 1
            while j <= len(binary_str):
                substring = binary_str[i:j]
                if substring not in binary_str[:i]:
                    complexity += 1
                    i = j
                    break
                j += 1
            else:
                complexity += 1
                break
        
        # Normalize by theoretical maximum
        max_complexity = len(binary_str) / np.log2(len(binary_str)) if len(binary_str) > 1 else 1
        return complexity / max_complexity
    
    def calculate_logical_depth(self, sequence: np.ndarray, 
                              compression_steps: int = 10) -> float:
        """Calculate logical depth (computational cost of generating sequence)"""
        import zlib
        
        # Convert to bytes for compression
        byte_sequence = sequence.tobytes()
        
        # Measure compression time as proxy for logical depth
        import time
        start_time = time.time()
        
        for _ in range(compression_steps):
            compressed = zlib.compress(byte_sequence)
        
        compression_time = time.time() - start_time
        
        # Normalize by sequence length
        return compression_time / len(sequence) if len(sequence) > 0 else 0.0
    
    def analyze_agent_information(self, agent_data: Dict[str, Any]) -> InformationProfile:
        """Comprehensive information analysis of an agent"""
        profile = InformationProfile(
            agent_id=agent_data.get('agent_id', ''),
            timestamp=agent_data.get('timestamp', 0.0)
        )
        
        # Extract time series data
        states = agent_data.get('state_history', [])
        if len(states) > 1:
            state_series = np.array(states)
            
            # Calculate phi if we have network structure
            if 'network_state' in agent_data and 'connectivity_matrix' in agent_data:
                profile.phi_value = self.calculate_phi(
                    agent_data['network_state'],
                    agent_data['connectivity_matrix']
                )
            
            # Calculate information storage
            profile.active_information_storage = self.calculate_active_information_storage(
                state_series
            )
            
            # Calculate transfer entropy if we have interaction data
            if 'interaction_history' in agent_data:
                interaction_series = np.array(agent_data['interaction_history'])
                if len(interaction_series) > 1:
                    profile.transfer_entropy = self.calculate_transfer_entropy(
                        interaction_series, state_series
                    )
            
            # Calculate complexity measures
            if 'binary_output' in agent_data:
                binary_data = np.array(agent_data['binary_output'])
                profile.lempel_ziv_complexity = self.calculate_lempel_ziv_complexity(binary_data)
            
            profile.logical_depth = self.calculate_logical_depth(state_series)
        
        return profile
    
    def compare_information_profiles(self, profile1: InformationProfile, 
                                   profile2: InformationProfile) -> Dict[str, float]:
        """Compare two information profiles"""
        differences = {}
        
        for field in ['phi_value', 'transfer_entropy', 'active_information_storage',
                     'information_modification', 'lempel_ziv_complexity', 'logical_depth']:
            val1 = getattr(profile1, field)
            val2 = getattr(profile2, field)
            
            if val1 is not None and val2 is not None:
                differences[field] = abs(val1 - val2)
        
        return differences
