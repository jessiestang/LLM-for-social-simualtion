## Layout of the code generator

The code generator module consists of the following LLM agent in the file coding_agent.py:

* Code generator (`code_generation, code_debugging, code_revision, model_interface_generator`): Generate/Debug MESA code based on the JSON mechanistic model

## Input & output of case studies

Two case studies are included. Each has a plain-text user requirements file as input, and a runnable Mesa Python file as output. A model interface JSON is also produced alongside the generated code to expose the model's key structures and parameters to the validation module.

### Input — user requirements


| File                             | Case study            | Description                                                                                                                                                                                                                                        |
| -------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shelling_model_requirement.txt` | Schelling Segregation | Specifies the output and measurement tasks for the generated model: grid visualisations at three time points (start, midpoint, final), colour-coded by agent type, plus a line chart tracking the number of unhappy agents over time.              |
| `spiral_silence_requirement.txt` | Spiral of Silence     | Specifies the data collection and visualisation tasks: time-series of the silent-agent ratio per opinion group, the visibility gap between the two opinions in media messages, and a KDE plot of opinion-stability distribution across all agents. |

### Output — generated models


| File                          | Case study            | Description                                                                                                                                                                                                                                     |
| ----------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generated_shelling_model.py` | Schelling Segregation | Mesa simulation code generated from`Shelling_model.json` and `shelling_model_requirement.txt`. Implements two racial agent types on a 2D grid, movement decisions based on neighbourhood homogeneity, and the required plots.                   |
| `spiral_silence_code.py`      | Spiral of Silence     | Mesa simulation code generated from`_spiral_silence_model.json` and `spiral_silence_requirement.txt`. Implements human and LLM agent types, the media system, the decision-influence rule, and the required data collection and visualisations. |

### Output — model interface


| File                                 | Case study        | Description                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spiral_silence_code_interface.json` | Spiral of Silence | Machine-readable API spec for`spiral_silence_code.py`. Documents class names (`SpiralOfSilenceModel`, `HumanAgent`, `LLMAgent`), available methods, key agent attributes, environment parameters, decision variables, and data-collection hooks. Used by the Validation module to generate evaluation code without reading the full model source. |

Folder output_plots contain the output plots for the Schelling Segregation study for simulations.
