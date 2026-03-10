---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# AI Agents for Geospatial Analysis

## Introduction

## Learning Objectives

## What Are AI Agents?

### Autonomy

### Planning

### Tool Use

### Memory

```{code-cell} python
# A minimal illustration of the agent loop concept
# (no external dependencies required)

def simple_agent_loop(goal, max_steps=5):
    """Simulate an agent's plan-act-observe cycle."""
    plan = [
        "Identify the required dataset",
        "Download or load the data",
        "Run the analysis function",
        "Check the result for errors",
        "Format and return the answer",
    ]
    print(f"Goal: {goal}\n")
    for i, step in enumerate(plan[:max_steps], 1):
        print(f"Step {i}: {step}")
        print(f"  -> Status: completed (simulated)\n")
    print("Agent: Goal achieved.")

simple_agent_loop("Calculate average NDVI for a study area")
```

## LLM-Powered Geospatial Agents

```{code-cell} python
# Simulating how an LLM agent might parse a geospatial request
# into a structured action plan (no API keys needed)

def parse_geospatial_request(user_query):
    """
    Demonstrate how an agent decomposes a natural language
    request into structured geospatial operations.
    """
    # In a real agent, an LLM would generate this structure
    parsed = {
        "query": user_query,
        "steps": [
            {"action": "geocode", "target": "Nashville, TN"},
            {"action": "buffer", "distance_km": 50},
            {"action": "query_features", "feature_type": "water_bodies"},
            {"action": "filter", "attribute": "area_ha", "operator": ">", "value": 10},
            {"action": "display_map", "layer": "filtered_results"},
        ],
    }
    print("Parsed geospatial workflow:")
    for i, step in enumerate(parsed["steps"], 1):
        print(f"  {i}. {step['action']}: {step}")
    return parsed

request = "Find all water bodies larger than 10 hectares within 50 km of Nashville"
workflow = parse_geospatial_request(request)
```

## Tool-Augmented Agents

```{code-cell} python
# Define a set of mock geospatial tools
# to illustrate how agents interact with tool registries

TOOL_REGISTRY = {
    "geocode": {
        "description": "Convert a place name to latitude/longitude coordinates",
        "parameters": {"place_name": "str"},
        "returns": "dict with lat, lon keys",
    },
    "compute_ndvi": {
        "description": "Calculate NDVI from red and NIR bands",
        "parameters": {"red_band": "array", "nir_band": "array"},
        "returns": "NDVI array with values between -1 and 1",
    },
    "buffer_point": {
        "description": "Create a circular buffer around a point",
        "parameters": {"lat": "float", "lon": "float", "radius_m": "float"},
        "returns": "GeoJSON polygon geometry",
    },
    "search_stac": {
        "description": "Search a STAC catalog for satellite imagery",
        "parameters": {"bbox": "list", "datetime": "str", "collection": "str"},
        "returns": "List of matching STAC items",
    },
}

print("Available tools for the agent:\n")
for name, info in TOOL_REGISTRY.items():
    print(f"  {name}: {info['description']}")
    print(f"    Parameters: {info['parameters']}")
    print()
```

## Multi-Step Workflows

```{code-cell} python
# Simulate a multi-step agent workflow with a scratchpad

class GeoAgentScratchpad:
    """Track intermediate results across workflow steps."""

    def __init__(self):
        self.steps = []
        self.data = {}

    def record(self, step_name, result):
        self.steps.append(step_name)
        self.data[step_name] = result
        print(f"  [{len(self.steps)}] {step_name}: {result}")

    def summary(self):
        print(f"\nWorkflow completed in {len(self.steps)} steps:")
        for i, step in enumerate(self.steps, 1):
            print(f"  {i}. {step}")


# Run a simulated flood damage assessment workflow
pad = GeoAgentScratchpad()
print("Agent: Starting flood damage assessment\n")
pad.record("geocode", {"lat": 35.96, "lon": -83.92, "name": "Knoxville, TN"})
pad.record("search_imagery", {"pre_flood": "S2_20250101", "post_flood": "S2_20250115"})
pad.record("flood_detection", {"flood_area_km2": 12.4, "pixels_classified": 124000})
pad.record("building_overlay", {"buildings_in_flood_zone": 347})
pad.record("generate_report", {"map_created": True, "stats_exported": True})
pad.summary()
```

## Code Generation for Geospatial Tasks

```{code-cell} python
# Example: an agent-generated code snippet for zonal statistics
# This is the kind of code an LLM agent would produce and execute

generated_code = '''
import numpy as np

# Simulated raster values for demonstration
np.random.seed(42)
zones = np.random.choice([1, 2, 3], size=(100, 100))
values = np.random.uniform(0.1, 0.9, size=(100, 100))

# Compute zonal statistics
zone_stats = {}
for zone_id in np.unique(zones):
    mask = zones == zone_id
    zone_vals = values[mask]
    zone_stats[zone_id] = {
        "mean": round(float(np.mean(zone_vals)), 3),
        "std": round(float(np.std(zone_vals)), 3),
        "count": int(np.sum(mask)),
    }

for zone_id, stats in zone_stats.items():
    print(f"Zone {zone_id}: mean={stats['mean']}, std={stats['std']}, pixels={stats['count']}")
'''

print("Agent-generated code for zonal statistics:\n")
exec(generated_code)
```

## Conversational GIS

## Challenges and Limitations

### Hallucination and Confabulation

### Spatial Reasoning Limitations

### Reproducibility and Trust

### Cost and Latency

### Data Privacy and Security

### Evaluation and Testing

## Current Landscape and Research Directions

```{code-cell} python
# Summary of key projects in the autonomous GIS landscape

landscape = {
    "Autonomous GIS (Li et al.)": {
        "focus": "Vision and framework for AI-driven GIS",
        "approach": "LLM-powered spatial analysis and reasoning",
        "status": "Active research with prototype implementations",
    },
    "GeoGPT / Geospatial Copilots": {
        "focus": "Conversational interfaces for geographic data",
        "approach": "LLMs connected to spatial databases and tools",
        "status": "Multiple prototypes from various research groups",
    },
    "LLM-Geo": {
        "focus": "Automated spatial analysis from natural language",
        "approach": "Code generation and execution for geo workflows",
        "status": "Published proof of concept",
    },
    "Tool-augmented LLMs": {
        "focus": "Function calling for spatial operations",
        "approach": "LangChain/LlamaIndex with geospatial tool libraries",
        "status": "Actively developed by open-source community",
    },
}

print("Current Landscape of AI Agents for Geospatial Analysis\n")
for project, details in landscape.items():
    print(f"{project}")
    for key, val in details.items():
        print(f"  {key}: {val}")
    print()
```

## Key Takeaways

## Exercises

### Exercise 1: Design an Agent Tool Library

### Exercise 2: Workflow Decomposition

### Exercise 3: Evaluating Agent Outputs

### Exercise 4: Hallucination Detection

### Exercise 5: Conversational GIS Prototype
