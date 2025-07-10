"""Command line entry point for running a simple experiment."""

import asyncio

from .experiment import RecursiveConsciousnessExperiment


async def main() -> None:
    print("Local Recursive Consciousness System")
    print("=" * 60)
    print("This demo uses only local models from HuggingFace.")
    print()
    experiment = RecursiveConsciousnessExperiment(num_agents=2, llm_model="distilgpt2")
    await experiment.run_experiment(num_steps=5)


if __name__ == "__main__":
    asyncio.run(main())
