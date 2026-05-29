## Layout of the model construction assistant

The model consruction assistant module consists of the following LLM agents in the file construction_agent.py:

* Variable generator (`decsion_rule_variables`): Brainstorm key decision variables based on the problem formulation
* Decision rule generator (`decision_rule_designer`): Generate executable if-then rules using selected variables
* Mechanistic model (`mechanism_translation`): Convert problem formulation + variables + decision rules into a complete mechanistic model
* ODD formatter (`ODD_formatter`): Format a mechanistic model into ODD format text.

## Input & output of case studies

Two case studies are included in this study. Each case study has a plain-text problem formulation as input and a structured JSON mechanistic model as output for this module.

### Input — problem formulations


| File                       | Case study            | Description                                                                                                                                                                                                                                                                                                  |
| -------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `shelling_segregation.txt` | Schelling Segregation | Describes a simulation of racial segregation in a closed urban grid. Two agent types with equal populations occupy a 2D grid with 25–30% vacant spots; at each step every agent decides whether to move based on neighbourhood homogeneity preference.                                                      |
| `spiral_silence.txt`       | Spiral of Silence     | Describes a simulation of the spiral-of-silence phenomenon in online social environments. Two agent types are used — human users (who may choose to stay silent) and LLM-driven agents (who always speak). A media system operates at the global level, collecting and republishing opinions each timestep. |

### Output — mechanistic models


| File                         | Case study            | Description                                                                                                                                                                                                                                                                    |
| ---------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Shelling_model.json`        | Schelling Segregation | Structured mechanistic model generated from`shelling_segregation.txt`. Specifies agent types, attributes, decision logic (a weighted signal comparing neighbourhood composition against a homogeneity preference threshold), environment parameters, and simulation constants. |
| `_spiral_silence_model.json` | Spiral of Silence     | Structured mechanistic model generated from`spiral_silence.txt`. Specifies human and LLM agent types, the media system as a model-level mechanism, the decision-influence formula governing whether human agents speak, and the full simulation schedule.                      |
