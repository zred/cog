"""
Example usage of CogMetrics consciousness measurement library
"""
import asyncio
import numpy as np
from cogmetrics import (
    InformationMetrics, 
    BehavioralTests, 
    NetworkAnalysis,
    ConsciousnessDashboard
)

async def measure_consciousness_example():
    """Example of comprehensive consciousness measurement"""
    
    # Initialize measurement tools
    info_metrics = InformationMetrics()
    behavioral_tests = BehavioralTests()
    network_analysis = NetworkAnalysis()
    
    # Mock agent data (replace with your actual agent)
    agent_data = {
        'agent_id': 'test_agent_001',
        'timestamp': 1234567890.0,
        'state_history': np.random.randn(100),
        'interaction_history': np.random.randn(100),
        'network_state': np.random.randint(0, 2, 10),
        'connectivity_matrix': np.random.rand(10, 10),
        'binary_output': np.random.randint(0, 2, 1000)
    }
    
    # 1. Information-theoretic analysis
    print("Running information-theoretic analysis...")
    info_profile = info_metrics.analyze_agent_information(agent_data)
    print(f"Phi value: {info_profile.phi_value}")
    print(f"Information storage: {info_profile.active_information_storage}")
    print(f"Complexity: {info_profile.lempel_ziv_complexity}")
    
    # 2. Behavioral testing (requires actual agent with respond_to_prompt method)
    # behavioral_results = await behavioral_tests.run_comprehensive_battery(agent)
    # print(f"Behavioral score: {behavioral_tests.get_aggregate_score('test_agent_001')}")
    
    # 3. Network analysis
    network_metrics = network_analysis.analyze_recursive_structure(
        agent_data['connectivity_matrix']
    )
    print(f"Network consciousness score: {network_metrics.get('consciousness_score', 0)}")
    
    # 4. Generate report
    report = {
        'agent_id': agent_data['agent_id'],
        'information_profile': info_profile,
        'network_metrics': network_metrics,
        'timestamp': agent_data['timestamp']
    }
    
    print("\nConsciousness Analysis Report:")
    print(f"Agent: {report['agent_id']}")
    print(f"Overall consciousness indicators detected: {bool(info_profile.phi_value)}")
    
    return report

if __name__ == "__main__":
    asyncio.run(measure_consciousness_example())