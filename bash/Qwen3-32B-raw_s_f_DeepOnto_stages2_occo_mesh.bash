#!/bin/bash -x
#SBATCH --job-name=ontoalign-Qwen3-32B-reason
#SBATCH --nodes=1
#SBATCH --partition=gpu,gpu-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu013
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=divyasha.sunil.naik@uni-jena.de
#SBATCH --mail-type=ALL




echo "# Job $SLURM_JOB_NAME started at $(date +%F-%T)"
module purge;
 
export http_proxy="http://internet4nzm.rz.uni-jena.de:3128"
export https_proxy="http://internet4nzm.rz.uni-jena.de:3128"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

set -euo pipefail
mkdir -p logs

srun --cpu-bind=none python3 - <<'PY'
import json
import os
import pandas as pd
    
from transformers import BitsAndBytesConfig
from ontoaligner.aligner import CandidateConceptLLMDataset
from ontoaligner.encoder import ConceptLLMEncoder
from ontoaligner.encoder.concept_formatting import (
    format_concept_context,
    format_candidate_concept_context,
)

from ontoaligner.ontology.generic import GenericOMDataset
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology.deeponto_verbalizer import (
    DeepOntoAxiomEnricher,
)

#candidate_file = (
 #   "/vast/ve83rur/OntoAligner/bash/"
  #  "results/Qwen3-32B/"
   # "Qwen3-32B_raw_s_f_occo_mesh_deeponto_stage1_case4.json"
                                                                 # "Qwen3-32B_raw_s_f_mesh_second_deeponto.json"   "second_pair.json"
                                                                 # "Llama-3.3-70B-Instruct_raw_s_f_mesh_ons_deeponto_stage1_case4.json"
                                                                 # "Qwen3-32B_raw_s_f_mesh_ons_deeponto_stage1_case4.json" 
#)



candidate_file = (
    "/vast/ve83rur/OntoAligner/bash/"
    "results/Llama-3.3-70B-Instruct/"
    "Llama-3.3-70B-Instruct_raw_s_f_mesh_ons_deeponto_stage1_case4.json"                                                       
)



if not os.path.isfile(candidate_file):
    raise FileNotFoundError(
        f"Candidate matching file not found: {candidate_file}"
    )


with open(candidate_file, encoding="utf-8") as candidate_input:
    candidates = json.load(candidate_input)

selected_source_iris = {
    candidate["source"]
    for candidate in candidates
}

selected_target_iris = {
    candidate["target"]
    for candidate in candidates
}


pipe = OntoAlignerPipeline(
    task_class=GenericOMDataset,
    source_ontology_path="assets/food-onto/occo.owl",
    target_ontology_path="assets/food-onto/mesh.owl",
    #reference_matching_path="",
    output_dir="results",
    output_format="json",
)



source_enricher = DeepOntoAxiomEnricher(
    ontology_path="/vast/ve83rur/OntoAligner/assets/food-onto/occo.owl",
    max_axioms_per_class=8,
    include_named_subclass_axioms=False,
    jvm_memory="4g",
)

target_enricher = DeepOntoAxiomEnricher(
    ontology_path="/vast/ve83rur/OntoAligner/assets/food-onto/mesh.owl",
    max_axioms_per_class=8,
    include_named_subclass_axioms=False,
    jvm_memory="4g",
)


source_enricher.enrich(
    pipe.dataset["source"],
    selected_iris=selected_source_iris,
)

target_enricher.enrich(
    pipe.dataset["target"],
    selected_iris=selected_target_iris,
)


out = pipe(
    method="llm",
    encoder_model=ConceptLLMEncoder(),
    dataset_class=CandidateConceptLLMDataset,

    candidate_matching_path=candidate_file,
    llm_prompt_preview_count=2,                       # For all the input to the prompt print in log # llm_prompt_preview_count=14, by default to 0

    llm_path="meta-llama/Llama-3.3-70B-Instruct",
    #llm_path="Qwen/Qwen3-32B",
    device="cuda",
    device_map="balanced",

    #apply_top_k_in_llm=True,
    #top_k=5, 

    llm_output_mode="raw",

    batch_size=1,
    max_length=1000,
    max_new_tokens=80,

    # The manual export block below saves the result, so avoid a duplicate.
    save_matchings=False,
    return_matching=True,

    llm_kwargs={
        "num_beams": 1,
        "do_sample": False,
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
    },
)


print("Done. Output type:", type(out))
print("Number of processed candidate pairs:", len(out))

if out:
    print("First item:")
    print(json.dumps(out[0], indent=2, ensure_ascii=False))

run_name = "Llama_Stage1_Llama-3.3-70B-Instruct_raw_s_f_occo_mesh_stage2_case6_with_eg_2-1"

#run_name = "Qwen_Stage1_Llama-3.3-70B-Instruct_raw_s_f_occo_mesh_stage2_case6_with_eg_2-1"

out_dir = "results/Llama-3.3-70B-Instruct/reason_deeponto"

#run_name = "Qwen3-32B_raw_s_f_occo_ons_stage2_case6"
#out_dir = "results/Qwen3-32B/reason_deeponto"

os.makedirs(out_dir, exist_ok=True)

json_path = os.path.join(out_dir, f"{run_name}.json")
csv_path = os.path.join(out_dir, f"{run_name}.csv")

with open(json_path, "w", encoding="utf-8") as output_file:
    json.dump(out, output_file, indent=2, ensure_ascii=False)

pd.DataFrame(out).to_csv(csv_path, index=False)

print("JSON saved to:", json_path)
print("CSV saved to:", csv_path)
PY

echo "# Job $SLURM_JOB_NAME finished at $(date +%F-%T)"