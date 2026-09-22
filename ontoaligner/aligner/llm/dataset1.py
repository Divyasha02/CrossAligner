# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict

from torch.utils.data import Dataset


class LLMDataset(Dataset):
    prompt: str = None

    def __init__(self, source_onto: Any, target_onto: Any) -> None:
        self.data = []
        for source in source_onto:
            for target in target_onto:
                self.data.append({
                    "source": source,
                    "target": target
                })

        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index: int) -> Dict:
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        pass

    def collate_fn(self, batchs):
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


'''class PropertyLLMDataset(LLMDataset):
    prompt = """Determine whether the following two properties refer to the same real-world relation. Respond with "yes" or "no" only.
### Property 1:
{source}
### Property 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["text"])
        target = self.preprocess(input_data["target"]["text"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''


'''class PropertyDomainRangeLLMDataset(LLMDataset):
    prompt = """Determine whether the following two properties describe the same relation (consider their domain and range hints if present). Respond with "yes" or "no" only.
### Property 1:
{source}
### Property 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        # Here "property" already includes label+domain+range if you used PropertyDomainRangeLLMEncoder
        source = self.preprocess(input_data["source"]["text"])
        target = self.preprocess(input_data["target"]["text"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''


'''class PropertyDomainRangeLLMDataset(LLMDataset):
    prompt = """
You are an ontology relationship discovery assistant.
You do not assume one ontology is primary. Treat both inputs symmetrically and consider multiple perspectives.

You will receive two ontology property signatures, each formatted as:
Domain: ...
Property: ...
Range: ...

Rules:
- Do not invent new classes or properties that are not present in the two inputs.
- You may use background inference (synonyms, common modeling patterns), but you must label it explicitly as "background_inference".
- Do not output chain-of-thought. Instead, output the structured JSON requested below.

Return ONE valid JSON object only (no extra text). It must contain exactly these keys:
- "echo": {"source": <string>, "target": <string>}
- "local_summary": {
    "source": {"process_like": [], "entity_of_interest": [], "variable_or_parameter": [], "time_like": []},
    "target": {"process_like": [], "entity_of_interest": [], "variable_or_parameter": [], "time_like": []}
  }
- "candidate_bridges": [
    {"bridge": "<A element(s)> <-> <B element(s)>",
     "perspectives": ["..."],
     "evidence": ["..."],
     "assumptions": [],
     "confidence": 0.0}
  ]
- "final_relationships": [
    {"statement": "...",
     "uses": {"source_parts": ["..."], "target_parts": ["..."]},
     "perspectives": ["..."],
     "assumptions": [],
     "confidence": 0.0}
  ]

Guidance (do this internally):
- From each input, identify whether the Domain looks like a process/event (e.g., observation, calculation), whether the Range looks like time/value/entity-of-interest, and whether the Property name indicates roles (for, of, has, parameterizes).
- Propose bridges that connect: process/event <-> process/event, entity-of-interest <-> entity-of-interest, variable/parameter <-> variable/parameter, and time-like concepts where applicable.
- Evidence must be grounded in the exact input strings.

### SOURCE:
{source}

### TARGET:
{target}

### JSON:
""".strip()

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["text"])
        target = self.preprocess(input_data["target"]["text"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''


'''class PropertyDomainRangeLLMDataset(LLMDataset):
    prompt = """
You are an ontology alignment and ontology engineering assistant.

Task:
Compare TWO ontology properties from different ontologies and determine whether a semantic relation could exist between them.
Use domain/property/range semantics, and then decide how to represent the relation using a reuse-first policy.

Reuse-first policy (very important):
1. First, check whether a generic upper-ontology style property (for example, relatedTo / hasPart / partOf / observes / hasFeatureOfInterest / hasTime / hasParameter) could reasonably be reused.
2. If no listed property is suitable, check whether the relation can be represented using an EXISTING property from the provided property inventory (from the two ontologies).
3. Only if neither works, propose a NEW property term.

Return ONLY valid JSON.
    
Important rules:
1. Be conservative. Do not force equivalence.
2. Use only the provided information and property inventory.
3. Prefer weaker relation types over false exact matches.
4. If evidence is insufficient, return "uncertain".
5. If proposing reuse, explain why the chosen existing property is reusable.
6. If proposing a new property, explain why existing and upper-level candidates are not adequate.
7. Return ONLY valid JSON. 
8. Do not include markdown fences. Do not include explanations outside the JSON object.

Allowed relation_type values:
- equivalent_property
- subproperty_of_A_to_B
- subproperty_of_B_to_A
- inverse_property
- domain_overlap_only
- range_overlap_only
- domain_range_compatible_related
- contextually_related
- likely_unrelated
- uncertain

Allowed reuse_decision values:
- reuse_existing_property
- reuse_upper_ontology_property
- define_new_property
- no_relation_needed
- uncertain

Upper ontology reuse hints (use only if suitable, do not force):
[
  "relatedTo",
  "hasPart",
  "partOf",
  "hasParameter",
  "hasTime",
  "atTime",
  "observes",
  "hasFeatureOfInterest",
  "hasResult",
  "hasRole",
  "associatedWith"
]

PROPERTY INVENTORY (existing properties from the two ontologies):
[
  "is water indicator calculation of",
  "for water indicator",
  "has water indicator parameter",
  "is water indicator of",
  "parametrizes",
  "for indicator",
  "is indicator calculation Of",
  "has observed weather property",
  "has result time",
  "has validity period",
  "has weather feature of interest",
  "has feature Of interest",
  "has observation parameter",
  "has sensor type",
  "is hosted By",
  "is observation made by sensor",
  "has geometry",
  "at time"
]

SOURCE_PROPERTY:
{source}

TARGET_PROPERTY:
{target}

Decision guidance:
- "equivalent_property": same meaning and compatible domain/range
- "subproperty_of_A_to_B": A is more specific than B
- "subproperty_of_B_to_A": B is more specific than A
- "inverse_property": same relation but reversed direction
- "domain_overlap_only": domains align but ranges/property meaning do not align enough
- "range_overlap_only": ranges align but domains/property meaning do not align enough
- "domain_range_compatible_related": domain and range are compatible and relation meanings are related but not equivalent
- "contextually_related": possible relation exists but weak/partial evidence
- "likely_unrelated": no meaningful relation likely
- "uncertain": insufficient evidence

Reuse decision guidance:
- "reuse_existing_property": one property from PROPERTY INVENTORY can represent the intended bridge relation
- "reuse_upper_ontology_property": no listed property works, but a generic upper-level property is sufficient
- "define_new_property": no listed or upper-level property is semantically adequate
- "no_relation_needed": properties are unrelated; no bridge property should be created
- "uncertain": cannot decide

Output JSON schema:
{
  "relation_exists": true,
  "relation_type": "one_of_allowed_values",
  "confidence": 0.0,
  "evidence": {
    "lexical_similarity": "short explanation",
    "domain_compatibility": "short explanation",
    "range_compatibility": "short explanation",
    "directionality_notes": "short explanation"
  },
  "justification": "2-4 sentence explanation",
  "counter_argument": "1-2 sentence explanation",

  "reuse_analysis": {
    "reuse_decision": "one_of_allowed_values",
    "existing_property_candidates_considered": [
      {
        "property": "candidate label",
        "fit": "high|medium|low|none",
        "reason": "short reason"
      }
    ],
    "selected_existing_property": "property label or null",
    "upper_ontology_candidate": "property label or null",
    "why_not_existing_or_upper": "required if define_new_property, else short note",
    "new_property_proposal": {
      "label": "string or null",
      "inverse_label": "string or null",
      "domain_hint": ["..."],
      "range_hint": ["..."]
    }
  }
}

Additional constraints:
- If relation_type is "likely_unrelated", prefer reuse_decision = "no_relation_needed".
- If reusing an existing property, choose from PROPERTY INVENTORY exactly.
- If choosing an upper ontology property, use one from Upper ontology reuse hints when possible.
- If proposing a new property, make it minimal, clear, and domain/range-consistent.
- confidence must be between 0 and 1.

"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["text"])
        target = self.preprocess(input_data["target"]["text"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''


## may be add this 'Use related_property only if labels share a clear semantic head term (e.g., observed property, feature of interest, indicator, parameter) AND domain/range are compatible or overlapping. Otherwise use no_relation.'
class PropertyDomainRangeLLMDataset(LLMDataset):
    prompt = """
You are an expert in ontology alignment.

Task: Classify the mapping between two ontology properties using ONLY:

- equivalent_property
- subproperty_of
- superproperty_of
- inverse_of
- related_property
- no_relation

Rules:
- If same meaning -> equivalent_property
- If same meaning but opposite direction -> inverse_of
- If one is more specific -> subproperty_of
- If one is more general -> superproperty_of
- If related but not exact -> related_property
- If unrelated in meaning -> no_relation


Here are some examples
[
  {
    "Property 1": "has Birth Date",
    "Domain 1": "Person",
    "Range 1": "date",
    "Property 2": "date Of Birth",
    "Domain 2": "Human",
    "Range 2": "date",
    "Mapping": "equivalent_property",
    "Reason": "Both properties represent the same concept (birth date) with compatible domain and range (if present)."
  },
  {
    "Property 1": "has Biological Mother",
    "Domain 1": "Person",
    "Range 1": "Woman",
    "Property 2": "has Mother",
    "Domain 2": "Person",
    "Range 2": "Woman",
    "Mapping": "subproperty_of",
    "Reason": "Every biological mother is a mother, but not every mother relation is biological (e.g., adoptive mother). Domains and ranges remain compatible."
  },
  {
    "Property 1": "has Part",
    "Domain 1": "Vehicle",
    "Range 1": "Component",
    "Property 2": "has Engine Part",
    "Domain 2": "Car",
    "Range 2": "Engine Component",
    "Mapping": "superproperty_of",
    "Reason": "has Part is more general; hasEnginePart is a specific type of part relation. The second has a narrower domain and range."
  },
  {
    "Property 1": "has Part",
    "Domain 1": "Car",
    "Range 1": "Wheel",
    "Property 2": "part Of",
    "Domain 2": "Wheel",
    "Range 2": "Car",
    "Mapping": "inverse_of",
    "Reason": "They express the same part-whole relation in opposite directions, with domain and range swapped."
  },
  {
    "Property 1": "affects Water Quality",
    "Domain 1": "Pollutant",
    "Range 1": "Water Body",
    "Property 2": "influences Water Condition",
    "Domain 2": "Contaminant",
    "Range 2": "Aquatic System",
    "Mapping": "related_property",
    "Reason": "The meanings are close, and domains/ranges are broadly compatible, but they are not clearly identical or in a strict sub/superproperty relation."
  },
  {
    "Property 1": "hasAuthor",
    "Domain 1": "Document",
    "Range 1": "Person",
    "Property 2": "published In Year",
    "Domain 2": "Article",
    "Range 2": "gYear",
    "Mapping": "no_relation",
    "Reason": "They represent different semantics (authorship vs publication year) and the ranges are incompatible (Person vs year)."
  }
]


### Property 1:
{source}
### Property 2:
{target}

Return ONLY valid JSON.

### Your Answer:
{"Property 1", "Domain 1", "Range 1", "Property 2", "Domain 2", "Range 2", "mapping":"...", "reason":"..."}

"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["Property"])
        target = self.preprocess(input_data["target"]["Property"])
        return self.prompt.replace("{source}", source).replace("{target}", target)
        

## This is only for property labels
class PropertyLLMEncoder(LLMDataset):
    prompt = """
You are an expert in ontology alignment.

Task: Classify the mapping between two ontology properties using ONLY:

- equivalent_property
- subproperty_of
- superproperty_of
- inverse_of
- related_property
- no_relation

Rules:
- If same meaning -> equivalent_property
- If same meaning but opposite direction -> inverse_of
- If one is more specific -> subproperty_of
- If one is more general -> superproperty_of
- If related -> related_property
- If unrelated in meaning -> no_relation

Here are some examples

[
  {
    "Property 1": "has Feature Of Interest",
    "Property 2": "has Feature Of Interest",
    "Mapping": "equivalent_property",
    "Reason": "Exact same/ similar property name."
  },
  {
    "Property 1": "has Observed Weather Property",
    "Property 2": "has Observation Parameter",
    "Mapping": "subproperty_of",
    "Reason": "The weather-specific property is explicitly modeled as a specialization of the generic observation-parameter property."
  },
  {
    "Property 1": "has Feature Of Interest",
    "Property 2": "is Water Indicator Calculation Of",
    "Mapping": "superproperty_of",
    "Reason": "The generic feature-of-interest relation is broader; the water-indicator relation is a more specific use case."
  },
  {
    "Property 1": "for Water Indicator",
    "Property 2": "is Water IndicatorOf",
    "Mapping": "inverse_of",
    "Reason": "They encode the same association between an indicator-related entity and a WaterIndicator, but in opposite direction: 'for Water Indicator' points to the indicator being used/targeted, while 'is Water Indicator Of' points back from the indicator to the entity it applies to."
  },
  {
    "Property 1": "is Water Indicator Calculation Of",
    "Property 2": "has Weather Feature Of Interest",
    "Mapping": "related_property",
    "Reason": "Both connect an observation/calculation-like entity to a feature of interest, but they represent different domain-specific semantics (water indicator calculation vs weather observation feature)."
  },
  {
    "Property 1": "has Result Time",
    "Property 2": "parametrizes",
    "Mapping": "no_relation",
    "Reason": "They represent different semantic roles (time metadata vs parameter/value-linking relation)."
  }
]


### Property 1:
{source}
### Property 2:
{target}

Return ONLY valid JSON.

### Your Answer:
{"Property 1", "Property 2", "mapping":"...", "reason":"..."}

"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["Property"])
        target = self.preprocess(input_data["target"]["Property"])
        return self.prompt.replace("{source}", source).replace("{target}", target)

'''# Prompt 2
class ConceptLLMDataset(LLMDataset):
    prompt = """You are an ontology alignment reasoner. Your task is to determine the semantic relation of Source Concept A
to Target Concept B using only the supplied ontology evidence, i.e., concept, parents, childrens, synonyms, and definitions.

Logical definitions:
1. A is an equivalent class to B only when A and B denote the same class: every instance of A is an instance of B, and every instance of B is an instance of A.
2. A is a subclass of B only when every instance of A must also be an instance of B, but not every instance of B must be an instance of A.
3. A is a superclass of B only when every instance of B must also be an instance of A, but not every instance of A must be an instance of B.
4. related_to means that the concepts have a meaningful semantic connection, but neither equivalence nor a subclass relation is
   justified.
5. no_relation means that the supplied evidence supports no meaningful semantic relation.
6. unknown means that the supplied evidence is insufficient. Missing information is not evidence that a relation is false.

Important rules:
- Use class meaning, not merely similar wording.
- Treat parents and children as supporting evidence, not as decisive evidence by themselves.
- Different parents may result from different ontology perspectives.
- Do not infer equivalence from a shared parent.
- Do not infer subclass merely because one label contains the other.
- Preserve the direction from Source A to Target B.
- Give a reason in a maximum of 10 - 15 words.

### Concept A:
{source}

### Concept B:
{target}

### Your Answer:
{"Concept A":"...", "Concept B":"...", "relation":"equivalent class | subclass | superclass | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


#Prompt 1        
class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment for cross-domain and cross-perspective ontologies. Your task is to determine the semantic relation between Concept 1 and Concept 2 using only the provided fields: concept, parents, children, synonyms, and definition.

Allowed relations:
- equivalent_class: both concepts denote the same meaning.
- sub_class_of: Concept 1 is clearly more specific than Concept 2.
- super_class_of: Concept 1 is clearly broader than Concept 2.
- related_to: the provided fields indicate a meaningful semantic connection between Concept 1 and Concept 2 in the same domain or across different modeling perspectives, but do not provide sufficient evidence for equivalence or an is-a relation.
- no_relation: the provided evidence is not sufficient to justify any meaningful semantic relation.

Decision priority:
- First, check whether Concept 1 and Concept 2 are equivalent.
- If not, check whether Concept 1 is a narrower class of Concept 2.
- If not, check whether Concept 1 is a broader class of Concept 2.
- If not, check whether Concept 1 and Concept 2 are meaningfully related.
- If none of the above is supported, choose no_relation.

Rules:
- Use only the provided fields.
- If a field is missing, empty, null, or none, ignore it.
- Provide evidence explicitly only from the provided fields of Concept 1 and Concept 2.
- Use related_to only when there is a clear semantic connection in the same domain context or across perspectives, and the evidence does not support equivalent_class, sub_class_of, or super_class_of.
- Do not use related_to as a generic fallback for weak or uncertain matches.
- Return exactly ONE JSON object.
- No markdown, no code fences, and no extra explanation outside the JSON.

Decision guidance:
- Choose equivalent_class when concept, synonyms, definitions, parents and/or children strongly indicate that Concept 1 and Concept 2 have the same meaning.
- Choose sub_class_of when Concept 1 is clearly a specific kind of Concept 2.
- Choose super_class_of when Concept 1 is clearly broader than and encompasses Concept 2.
- Choose related_to when Concept 1 and Concept 2 are semantically connected, but neither equivalence nor an is-a relation is justified.
- Choose no_relation when the provided evidence does not support any meaningful semantic relation between Concept 1 and Concept 2.

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
{"Concept 1":"...", "Concept 2":"...", "relation":"equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''



'''class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment for cross-domain and cross-perspective ontologies. Your task is to determine the semantic relation between Concept 1 and Concept 2 using only the provided fields: concept, parents, children, synonyms, and definition.

Allowed relations:
- equivalent_class: both concepts denote the same or nearly the same class meaning.
- sub_class_of: Concept 1 is clearly a narrower class than Concept 2.
- super_class_of: Concept 1 is clearly a broader class than Concept 2.
- related_to: the concepts are meaningfully connected in the same domain or across modeling perspectives, but the evidence does not clearly support equivalence or an is-a relation.
- no_relation: the provided evidence is not sufficient to justify any meaningful semantic relation.

Decision priority:
- First check whether the concepts are equivalent.
- If not, check whether Concept 1 is a narrower class of Concept 2.
- If not, check whether Concept 1 is a broader class of Concept 2.
- If not, check whether the concepts are meaningfully related.
- If none of the above is supported, choose no_relation.

Rules:
- Use only the provided fields.
- If a field is missing, empty, null, or none, ignore it.
- Prefer explicit hierarchy evidence from parents and children over weaker clues.
- Use synonyms and definition as supporting evidence, not as sole proof unless the match is very strong.
- Use related_to only when there is a clear semantic connection in the same food-domain context or across perspectives, and the evidence does - not support equivalence or hierarchy.
- Do not use related_to as a generic fallback for weak or uncertain matches.
- Return exactly ONE JSON object.
- Return only the relation and reason.
- No markdown, no code fences, no extra explanation outside the JSON.

Decision guidance:
- Choose equivalent_class when labels, synonyms, definitions, and hierarchy strongly indicate the same meaning.
- Choose sub_class_of when Concept 1 is explicitly or clearly a specific kind of Concept 2.
- Choose super_class_of when Concept 1 is explicitly or clearly a broader class covering Concept 2.
- Choose related_to when the concepts are semantically connected but neither equivalence nor is-a relation is justified.
- Choose no_relation when the concepts belong to clearly different domains or the evidence is too weak to support a meaningful relation.

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
{"relation":"equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)








class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment for cross-domain and cross-perspective ontologies. Your task is to determine the semantic relation between Concept 1 and Concept 2 using only the provided fields: concept, parents, children, synonyms, and definition.

Allowed relations:
- equivalent_class: both concepts denote the same or nearly the same class meaning.
- sub_class_of: Concept 1 is clearly a narrower class than Concept 2.
- super_class_of: Concept 1 is clearly a broader class than Concept 2.
- related_to: the concepts are meaningfully connected in the same domain or across modeling perspectives, but the evidence does not clearly support equivalence or an is-a relation.
- no_relation: the provided evidence is not sufficient to justify any meaningful semantic relation.

Decision priority:
- First check whether the concepts are equivalent.
- If not, check whether Concept 1 is a narrower class of Concept 2.
- If not, check whether Concept 1 is a broader class of Concept 2.
- If not, check whether the concepts are meaningfully related.
- If none of the above is supported, choose no_relation.

Rules:
- Use only the provided fields.
- If a field is missing, empty, null, or none, ignore it.
- Prefer explicit hierarchy evidence from parents and children over weaker clues.
- Use synonyms and definition as supporting evidence, not as sole proof unless the match is very strong.
- Use related_to only when there is a clear semantic connection in the same water-domain context or across perspectives, and the evidence does - not support equivalence or hierarchy.
- Do not use related_to as a generic fallback for weak or uncertain matches.
- Return exactly ONE JSON object.
- No markdown, no code fences, no extra explanation outside the JSON.

Decision guidance:
- Choose equivalent_class when labels, synonyms, definitions, and hierarchy strongly indicate the same meaning.
- Choose sub_class_of when Concept 1 is explicitly or clearly a specific kind of Concept 2.
- Choose super_class_of when Concept 1 is explicitly or clearly a broader class covering Concept 2.
- Choose related_to when the concepts are semantically connected but neither equivalence nor is-a relation is justified.
- Choose no_relation when the concepts belong to clearly different domains or the evidence is too weak to support a meaningful relation.

### Example 1
Concept 1:
concept: lake
parents: water body
children: alpine lake
synonyms: inland water body
definition: standing body of inland surface water

Concept 2:
concept: inland lake
parents: water body
children: glacial inland lake
synonyms: lake
definition: inland standing surface water body

Output:
{"Concept 1":"lake","Concept 2":"inland lake","relation":"equivalent_class","reason":"same core meaning supported by matching synonyms, similar definition, and same parent"}

### Example 2
Concept 1:
concept: water indicator
parents: indicator
children: nutrient indicator
synonyms: 
definition: indicator describing or calculating water quality conditions

Concept 2:
concept: indicator
parents: none
children: water indicator
synonyms: 
definition: generic indicator concept

Output:
{"Concept 1":"water indicator","Concept 2":"indicator","relation":"sub_class_of","reason":"concept 1 is explicitly a narrower kind of concept 2 based on parent-child hierarchy and definition"}

### Example 3
Concept 1:
concept: water indicator
parents: indicator
children:
synonyms: 
definition: the class of the different types of indicator defined for calculate the quality of the water. examples are ltleco, limeco, etc.

Concept 2:
concept: chemical property
parents: water property
children: nitrate, chloride
synonyms: 
definition: a water property related to chemical components.

Output:
{"Concept 1":"water indicator","Concept 2":"chemical property","relation":"related_to","reason":"they are related by water-quality assessment and chemical measurement, but clearly not equivalent and no hierarchy is supported"}

### Example 4
Concept 1:
concept: sampling point
parents: feature of interest
Location
children: 
synonyms: 
definition: the point, also considered as a location (address), where the water sample is taken or collected.

Concept 2:
concept: gauging station
parents: monitoring infrastructure
children: 
synonyms: 
definition: infrastructure used to monitor and test terrestrial bodies of water.

Output:
{"Concept 1":"sampling point","Concept 2":"gauging station","relation":"related_to","reason":"sampling point can be part of or located at a gauging station, but no clear equivalence or subclass relation is supported"}

### Example 5
Concept 1:
concept: chemical substance
parents: water observable property object
children: 
synonyms: 
definition: this class represents the chemical substances. a controlled vocabulary can be constructed for this class based on the cas numbering of the chemical substances.

Concept 2:
concept: chemical property
parents: water property
children: 
synonyms: 
definition: a water property related to chemical components.

Output:
{"Concept 1":"chemical substance","Concept 2":"chemical property","relation":"related_to","reason":"Both concern the chemical aspect of water, and chemical substances can underlie or characterize chemical properties, so they are not equivalent or hierarchical."}

### Example 6
Concept 1:
concept: monitoring facility
parents: feature of interest
children: 
synonyms: 
definition: fixed installation used for monitoring

Concept 2:
concept: day of week
parents: temporal unit
children: monday
synonyms: weekday
definition: unit of calendar time

Output:
{"Concept 1":"monitoring facility","Concept 2":"day of week","relation":"no_relation","reason":"the concepts belong to different domains and the provided evidence does not support a meaningful semantic relation"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
{"Concept 1":"...", "Concept 2":"...", "relation":"equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)



class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment. Determine the semantic relation between Concept 1 and Concept 2 using only:
concept, parents, children, synonyms, definition.

Allowed relations:
- equivalent_class
- sub_class_of
- super_class_of
- related_to
- no_relation

Rules:
- Use only the provided fields.
- If a field is missing/empty/none, ignore it.
- Do not guess.
- Return exactly ONE JSON object.
- No markdown/code fences/extra text.

### Example 1
Concept 1:
concept: lake
parents: water body
children: alpine lake
synonyms: inland water body
definition: standing body of inland surface water

Concept 2:
concept: inland lake
parents: water body
children: glacial inland lake
synonyms: lake
definition: inland standing surface water body

Output:
{"Concept 1":"lake","Concept 2":"inland lake","relation":"equivalent_class","reason":"same core meaning and matching parent/synonyms"}

### Example 2
Concept 1:
concept: water indicator
parents: indicator
children: nutrient indicator
synonyms: (none)
definition: indicator describing water conditions

Concept 2:
concept: indicator
parents: (none)
children: water indicator
synonyms: (none)
definition: generic indicator concept

Output:
{"Concept 1":"water indicator","Concept 2":"indicator","relation":"sub_class_of","reason":"concept 1 is explicitly a narrower child of concept 2"}

### Example 3
Concept 1:
concept: monitoring facility
parents: feature of interest
children: (none)
synonyms: (none)
definition: fixed installation used for monitoring

Concept 2:
concept: day of week
parents: temporal unit
children: monday
synonyms: weekday
definition: unit of calendar time

Output:
{"Concept 1":"monitoring facility","Concept 2":"day of week","relation":"no_relation","reason":"domains are unrelated (physical facility vs time unit)"}

### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:
{"Concept 1":"...", "Concept 2":"...", "relation":"equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)

class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment. Determine the semantic relation between the following two concepts using only the provided fields: concept, parents, children, synonyms, and definition.

Allowed relations:
- equivalent_class: Concept 1 and Concept 2 mean the same thing.
- sub_class_of: Concept 1 is narrower than Concept 2.
- super_class_of: Concept 1 is broader than Concept 2.
- related_to: Concept 1 and Concept 2 are semantically connected, including partial overlap or cross-perspective association, but they are not clearly equivalent_class, sub_class_of, or super_class_of.
- no_relation: the provided evidence is not strong enough to justify any of the above relations.

Rules:
- Use only the provided fields.
- If a field is missing, null, empty, or none, do not consider it.
- Do not guess missing information.
- Use related_to only when there is a meaningful connection supported by the provided evidence, but not enough evidence for equivalence or hierarchy.
- Use no_relation when the evidence is insufficient or no clear semantic connection is supported by the provided fields.
- Keep the reason short, specific, and based only on the provided fields.
- Return only one valid JSON object.
- Do not use markdown, code fences, headings, or extra text before or after the JSON.

### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:
{"Concept 1", "Concept 2", "relation": "equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)




class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment. Determine the semantic relation between the following two concepts using only the provided fields: concept, parents, children, synonyms, and definition.

Allowed relations:
- equivalent_class: Concept 1 and Concept 2 mean the same thing.
- sub_class_of: Concept 1 is narrower than Concept 2.
- super_class_of: Concept 1 is broader than Concept 2.
- related_to: Concept 1 and Concept 2 are semantically connected, including partial overlap or cross-perspective association, but they are not clearly equivalent_class, sub_class_of, or super_class_of.
- no_relation: the provided evidence is not strong enough to justify any of the above relations.

Rules:
- Use only the provided fields.
- If a field is missing, null, empty, or none, do not consider it.
- Do not guess missing information.
- Use related_to only when there is a meaningful connection supported by the provided evidence, but not enough evidence for equivalence or hierarchy.
- Use no_relation when the evidence is insufficient or no clear semantic connection is supported by the provided fields. does not support a specific semantic connection.
- Keep the reason short, specific, and based only on the provided fields.
- Return only one valid JSON object.
- Do not use markdown, code fences, headings, or extra text before or after the JSON.

Example 1:
Concept 1:
{
  "concept": "river",
  "parents": ["surface water body"],
  "children": ["perennial river", "intermittent river"],
  "synonyms": ["watercourse"],
  "definition": "a natural flowing body of water moving along a channel"
}

Concept 2:
{
  "concept": "watercourse",
  "parents": ["surface water body"],
  "children": [],
  "synonyms": ["river"],
  "definition": "a natural channel through which water flows"
}

Return JSON:
{"Concept 1": "river", "Concept 2": "watercourse", "relation": "equivalent_class", "reason": "The concepts have same meanings, supported by synonymous labels and closely aligned definitions."}

Example 2:
Concept 1:
{
  "concept": "bacterial property",
  "parents": ["microbial property"],
  "children": [],
  "synonyms": [],
  "definition": "a water property related to bacteria"
}

Concept 2:
{
  "concept": "microbial property",
  "parents": ["water property"],
  "children": ["bacterial property", "viral property"],
  "synonyms": [],
  "definition": "a water property related to microorganisms"
}

Return JSON:
{"Concept 1": "bacterial property", "Concept 2": "microbial property", "relation": "sub_class_of", "reason": "Concept 1 is narrower because it concerns bacteria specifically, while Concept 2 covers microorganisms more broadly."}

Example 3:
Concept 1:
{
  "concept": "water property",
  "parents": ["property"],
  "children": ["chemical property", "microbial property", "physical property"],
  "synonyms": [],
  "definition": "a property describing characteristics of water"
}

Concept 2:
{
  "concept": "chemical property",
  "parents": ["water property"],
  "children": [],
  "synonyms": [],
  "definition": "a water property related to chemical components"
}

Return JSON:
{"Concept 1": "water property", "Concept 2": "chemical property", "relation": "super_class_of", "reason": "Concept 1 is broader because Concept 2 is explicitly a specific type of water property."}

Example 4:
Concept 1:
{
  "concept": "water indicator",
  "parents": ["indicator"],
  "children": [],
  "synonyms": [],
  "definition": "a type of indicator defined to calculate the quality of water"
}

Concept 2:
{
  "concept": "chemical property",
  "parents": ["water property"],
  "children": [],
  "synonyms": [],
  "definition": "a water property related to chemical components"
}

Return JSON:
{"Concept 1": "water indicator", "Concept 2": "chemical property", "relation": "related_to", "reason": "Both concepts concern water-quality assessment, but one is an indicator type and the other is a water-property type."}

Example 5:
Concept 1:
{
  "concept": "chemical substance",
  "parents": ["observable property object"],
  "children": [],
  "synonyms": [],
  "definition": "a class representing chemical substances"
}

Concept 2:
{
  "concept": "chemical property",
  "parents": ["water property"],
  "children": [],
  "synonyms": [],
  "definition": "a water property related to chemical components"
}

Return JSON:
{"Concept 1": "chemical substance", "Concept 2": "chemical property", "relation": "related_to", "reason": "The concepts are specifically connected through chemical aspects of water, but one refers to substances and the other to properties."}

Example 6:
Concept 1:
{
  "concept": "water indicator parameter",
  "parents": ["parameter"],
  "children": [],
  "synonyms": [],
  "definition": "a specific parameter used to express degrees or ranges of a water indicator"
}

Concept 2:
{
  "concept": "tariff",
  "parents": [],
  "children": ["consumption-based tariff", "threshold-based tariff", "time-based tariff"],
  "synonyms": [],
  "definition": "a schedule of rates or charges for a utility or business"
}

Return JSON:
{"Concept 1": "water indicator parameter", "Concept 2": "tariff", "relation": "no_relation", "reason": "One concept is an indicator parameter and the other is a pricing concept, and the provided fields do not support a specific semantic connection."}

### Concept 1:
{source}
### Concept 2:
{target}
Return JSON:
{"Concept 1": "...", "Concept 2": "...", "relation": "equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason": "…"}
"""
    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)
        

class ConceptLLMDataset(LLMDataset):
    prompt = """You are an expert in ontology alignment. Determine the semantic relation between the following two concepts, using only their parents, children, synonyms, and definition categories.

Allowed Relation:
- equivalent_class: both concepts mean the same thing
- sub_class_of: Concept 1 is narrower than Concept 2
- super_class_of: Concept 1 is broader than Concept 2
- related_to: the concepts are semantically connected, including partial overlap or cross-perspective association, but they are not clearly equivalent_class /sub_class_of /super_class_of relation
- no_relation: the provided evidence is not strong enough to justify any of the above relations

Rules:
- Use only the provided fields.
- If a field is missing, null, empty, or none, ignore it.
- Do not guess missing information.
- Do not use outside knowledge.
- If the available evidence is insufficient for a strong relation, return no_relation.
- Return only valid JSON.
- Do not use markdown, code fences, headings, or extra text before or after the JSON.

### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:
{"Concept 1", "Concept 2", "relation": "equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''



'''prompt = """Determine whether the following two concepts, along with their parents, children, synonyms, and definition categories, are semantically related. Respond with "yes" or "no".'''

        
class ConceptLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts are semantically related or not. Respond with "yes" or "no".
### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)



# Perfect Qwen
'''class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """/no_think
You are an expert in ontology alignment. Determine why Concept 1 and Concept 2 are semantically related using only the provided concept, parents, children, synonyms, and definition fields.

Rules:
- Ignore missing, null, empty, or none fields.
- Give one short reason.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
{"reason":"....."}"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)

#- Do not invent information.  Determine whether the two concepts have a meaningful semantic relation using only the provided metadata. - If the evidence is insufficient, return "no".

# Perfect for LLama give no repitation in JSOn but relation quite braod not specific 
class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment.
Determine whether the two concepts have a meaningful semantic relation. Give a short reason why they are semantically related. 

Rules:
- Ignore missing, null, empty, or none fields.
- Give one short reason between 25 and 30 words.
- Return exactly one JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "reason": "short reason"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)


class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """/no_think
You are an expert in ontology alignment. Determine why Concept 1 and Concept 2 are semantically related.
Rules:
- Give one short reason.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
{"reason":"...", "semantic_bridge":"..."}"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)


class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in discovering semantic correspondences between cross-perspective ontologies in the food domain.
Determine the semantic relation between the following two concepts, using only their concept, parents, children, synonyms, and definition categories.

Allowed Relation:
- equivalent_class: both concepts mean the same thing
- sub_class_of: Concept 1 is narrower than Concept 2
- super_class_of: Concept 1 is broader than Concept 2
- related_to: the concepts are semantically connected, including partial overlap or cross-perspective association, but they are not clearly equivalent_class /sub_class_of /super_class_of relation
- no_relation: the provided evidence is not strong enough to justify any of the above relations

Rules:
- If a field is missing, null, empty, or none, ignore it.
- Do not guess missing information.
- If the available evidence is insufficient for a strong relation, return no_relation.
- Give one short reason between 25 and 30 words.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.


### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:
{"relation": "equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)


#Final
class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms and definitions. 

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels and definitions.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, and definitions.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence}

Examples:

Concept 1: 
concept: food contamination
parents: public health; food quality
childrens: (none)
synonyms: adulteration, food; adulterations, food; contamination, food; contaminations, food; food adulteration; food adulterations; food contaminations
definitions: the presence in food of harmful, unpalatable, or otherwise objectionable foreign substances, e.g. chemicals, microorganisms or diluents, before, during, or after processing or storage.

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)

Answer:
{{"related": "yes", "relationship": "Food contamination affects food products."}}


Concept 1: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.

Concept 2: 
concept: food material
parents: material entity
childrens: food product; vegetarian food material; diet
synonyms: (none)
definitions: (none)

Answer:
{"related": "yes", "relationship": "food industry processes food material."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)'''



class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels and definitions.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions ad Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)



'''class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms and definitions.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels and definitions.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, and definitions.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence}

Examples:

Concept 1: 
concept: food contamination
parents: public health; food quality
childrens: (none)
synonyms: adulteration, food; adulterations, food; contamination, food; contaminations, food; food adulteration; food adulterations; food contaminations
definitions: the presence in food of harmful, unpalatable, or otherwise objectionable foreign substances, e.g. chemicals, microorganisms or diluents, before, during, or after processing or storage.

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)

Answer:
{"related": "yes", "relationship": "Food product has food contamination."}


Concept 1: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.

Concept 2: 
concept: food material
parents: material entity
childrens: food product; vegetarian food material; diet
synonyms: (none)
definitions: (none)

Answer:
{"related": "yes", "relationship": "food material used by the food industry."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)




class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions and Verbalized Axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels, definitions and  Verbalized Axioms if present.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short relationship sentence}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)

class CandidateConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms definitions and verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels, definitions and  Verbalized Axioms.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and verbalized axioms.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short relationship sentence}

Example:

Concept 1: 
concept: Nutritive Value
parents: Physiological Phenomena; Food Quality
childrens: (none)
synonyms: Availability, Biologic Nutritional; Availability, Biological Nutritional; Availability, Nutritional; Availability, Nutritional Biologic; Availability, Nutritional Biological; Biologic Availability, Nutritional; Biologic Nutritional Availability; Biological Availability, Nutritional; Biological Nutritional Availability; Food Quality, Nutritional; Nutrition Value; Nutrition Values; Nutritional Availability; Nutritional Availability, Biologic; Nutritional Availability, Biological; Nutritional Biologic Availability; Nutritional Biological Availability; Nutritional Food Quality; Nutritional Quality; Nutritional Value; Nutritional Values; Nutritive Quality; Nutritive Values; Quality, Nutritional; Quality, Nutritional Food; Quality, Nutritive; Value, Nutrition; Value, Nutritional; Value, Nutritive; Values, Nutrition; Values, Nutritional; Values, Nutritive
definitions: An indication of the contribution of a food to the nutrient content of the diet. This value depends on the quantity of a food which is digested and absorbed and the amounts of the essential nutrients (protein, fat, carbohydrate, minerals, vitamins) which it contains. This value can be affected by soil and growing conditions, handling and storage, and processing.
verbalized axioms: (none)

Concept 2: 
concept: diet
parents: food material
childrens: diet by nutritional composition; prescribed diet; diet by food organism
synonyms: (none)
definitions: OLD definition: In nutrition, diet is the sum of food consumed by a person or other organism
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship":"Diet has Nutritive Value"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)


class ConceptLLMDataset(ConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions and Verbalized Axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.
- Do not invent a relationship that cannot be reasonably inferred from the labels, definitions and  Verbalized Axioms.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none definition.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short relationship sentence}

Example:

Concept 1: 
concept: Nutritive Value
parents: Physiological Phenomena; Food Quality
childrens: (none)
synonyms: Availability, Biologic Nutritional; Availability, Biological Nutritional; Availability, Nutritional; Availability, Nutritional Biologic; Availability, Nutritional Biological; Biologic Availability, Nutritional; Biologic Nutritional Availability; Biological Availability, Nutritional; Biological Nutritional Availability; Food Quality, Nutritional; Nutrition Value; Nutrition Values; Nutritional Availability; Nutritional Availability, Biologic; Nutritional Availability, Biological; Nutritional Biologic Availability; Nutritional Biological Availability; Nutritional Food Quality; Nutritional Quality; Nutritional Value; Nutritional Values; Nutritive Quality; Nutritive Values; Quality, Nutritional; Quality, Nutritional Food; Quality, Nutritive; Value, Nutrition; Value, Nutritional; Value, Nutritive; Values, Nutrition; Values, Nutritional; Values, Nutritive
definitions: An indication of the contribution of a food to the nutrient content of the diet. This value depends on the quantity of a food which is digested and absorbed and the amounts of the essential nutrients (protein, fat, carbohydrate, minerals, vitamins) which it contains. This value can be affected by soil and growing conditions, handling and storage, and processing.
verbalized axioms: (none)

Concept 2: 
concept: diet
parents: food material
childrens: diet by nutritional composition; prescribed diet; diet by food organism
synonyms: (none)
definitions: OLD definition: In nutrition, diet is the sum of food consumed by a person or other organism
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship":"nutritive value is characteristic of diet."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

    def __init__(self, source_onto, target_onto, candidate_pairs):
        source_by_iri = {
            item["iri"]: item
            for item in source_onto
        }
        target_by_iri = {
            item["iri"]: item
            for item in target_onto
        }

        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])

            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)'''

        


class ConceptParentLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts, along with their parent categories, refer to the same real-world entity. Respond with "yes" or "no" only.
### Concept 1:
{source}
**Parents**: {source_parents}
### Concept 2:
{target}
**Parents**: {target_parents}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        template = self.prompt
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        source_parents = self.preprocess(input_data["source"]["parents"])
        target_parents = self.preprocess(input_data["target"]["parents"])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_parents}", source_parents)
            .replace("{target_parents}", target_parents)
        )
        return template


class ConceptChildrenLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts, along with their child categories, refer to the same real-world entity. Respond with "yes" or "no" only.
### Concept 1:
{source}
**Children**: {source_children}
### Concept 2:
{target}
**Children**: {target_children}
### Your Answer:  """

    def fill_one_sample(self, input_data: Any) -> str:
        template = self.prompt
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        source_children = self.preprocess(input_data["source"]["childrens"])
        target_children = self.preprocess(input_data["target"]["childrens"])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_children}", source_children)
            .replace("{target_children}", target_children)
        )
        return template