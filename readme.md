# "Guide" the Vision Language Model to "Think" like DeepSeek-R1

In this project, we have successfully explored an efficient and cost-effective method to enable a Vision Language Model (VLM) to possess the deep thinking and reasoning capabilities similar to those of DeepSeek-R1 at a low cost, by utilizing limited GPU resources.

## Core Technical Concept
We adopt a unique data processing strategy to handle the original SFT (Supervised Fine-Tuning, question-answer pairs) data. By transforming the SFT data into content that includes the "thinking" process, we form a COT (Chain of Thought, question-think-answer chain) data structure. Subsequently, we use the SFT method to train the visual model, skillfully "guiding" the model to learn how to "think".

## Project File Structure
- **`DistilabelPipelineImage.py`**: Handles the processing of conversation with image data.
- **`DistilabelPipeline.py`**: Handles pure text conversations.
- **`ModifyThinkingContent.py`**: This script is responsible for modifying the inference content containing images, endowing the SFT data with "thinking" elements, and it is an important step in constructing the COT data for conversation with images.
- **`data/`**: The directory for storing data.

## Model Acquisition Method
The model used in this project can be downloaded through the following link: [https://modelscope.cn/models/atlas9999/Deepseek-r1-distill-qwen2_vl/summary](https://modelscope.cn/models/atlas9999/Deepseek-r1-distill-qwen2_vl/summary). Please properly place the downloaded model files in the designated location of the project so that they can be smoothly called during subsequent training and inference processes.

## SFT Training
We use llama-factory for SFT training. For full-parameter fine-tuning, please refer to llama-factory. The parameter scripts are `./qwen2vl_full_sft.yaml`, `qwen2vl_full_sft.yaml`.

## evaluation
we sft finetuning qwen2-vl model by using domain datasets, and continue training by using cot data(this repo method).

see repo: https://github.com/sungatetop/geology-lvlm.git

## Inference:
### Notic: pull the latest llama-factory!
Copy the .yaml file to the corresponding directory of llama-factory

case1:
```
    API_PORT=8000 llamafactory-cli api examples/inference/qwen2_vl_think.yaml
```

case2:
```
vllm serve /path/model --dtype auto --port 8000 --limit_mm_per_prompt image=4 --max_model_len 8784 --gpu_memory_utilization 0.75
```
#### chainlit
```
 chainlit run chat.py 
```

# Some Results


<image src="./assets/demo4.png" />

<image src="./assets/demo5.png" />

<image src="./assets/demo6.png" /> 

## Citation

```bibtex
@misc{deepseek-r1-vision,
  author = {Baolin Chen},
  title = {deepseek-r1-vision},
  year = {2025},
  howpublished = {\url{https://github.com/sungatetop/deepseek-r1-vision.git}}
}
```