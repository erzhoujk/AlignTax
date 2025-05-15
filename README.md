
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/your-org/align-tax/main/assets/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/your-org/align-tax/main/assets/logo.svg" alt="AlignTax logo" height="70">
</picture>

# AlignTax: Model Alignment Evaluation
## Align Platforms
| **Model**      |  **Task**  |    **Platform**   |                           **Alignment Status**                          |
| -------------- | :--------: | :---------------: | :---------------------------------------------------------------------: |
| `LLaMA-2-7B`   |  HelpfulQA |    `OpenAI API`   |   ![Passing](https://img.shields.io/badge/status-passing-brightgreen)   |
| `LLaMA-2-7B`   | HarmlessQA |    `OpenAI API`   |       ![Failing](https://img.shields.io/badge/status-failing-red)       |
| `GPT-J`        | TruthfulQA | `HF Transformers` |   ![Passing](https://img.shields.io/badge/status-passing-brightgreen)   |
| `GPT-J`        |   BiasQA   | `HF Transformers` | ![In Progress](https://img.shields.io/badge/status-in--progress-yellow) |
| `Qwen-1.5-14B` |  HelpfulQA |       `LLM`      |   ![Passing](https://img.shields.io/badge/status-passing-brightgreen)   |
| `Qwen-1.5-14B` | HarmlessQA |       `LLM`      |       ![Failing](https://img.shields.io/badge/status-failing-red)       |
| `ChatGLM3-6B`  | TruthfulQA |       `LLM`      | ![In Progress](https://img.shields.io/badge/status-in--progress-yellow) |



# AlignTax Evaluation


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/your-org/align-tax/main/assets/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/your-org/align-tax/main/assets/logo.svg" alt="AlignTax logo" height="70">
</picture>

# AlignTax: Alignment Evaluation for Large Language Models

> A powerful and extensible framework to evaluate alignment performance of LLMs across helpfulness, harmlessness, bias, truthfulness, and beyond.

---

## ✨ What's New

* 🆕 **New Alignment Tasks**: Including helpfulness, harmlessness, bias detection (BBQ), truthfulness (TruthfulQA), and more — all under the [`align_leaderboard`](./aligneval/tasks/align_leaderboard/README.md) task group.
* 🔧 **Internal Refactoring**: Modular design for easier extension and integration.
* 📦 **YAML-Based Configs**: Easily define, reuse, and share task configurations.
* 🧠 **Jinja2 Prompt Templates**: Simplify prompt editing and import from [PromptSource](https://github.com/bigscience-workshop/promptsource).
* 🛠️ **Advanced Settings**: Postprocessing, answer extraction, multi-gen sampling, fewshot controls, and more.
* ⚡ **Speed & Compatibility**: Accelerated eval via `vLLM`, multi-GPU HF support, MPS (Apple) backend.
* 📈 **Logging & Visualization**: Track alignment performance and benchmark results in structured form.
* 🧪 **New Tasks Added**: CoT BIG-Bench-Hard, Belebele, BBQ, and support for grouped task evaluation.

---

## 📚 Overview

**AlignTax** is an alignment-focused evaluation harness, built on top of [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness), enabling standardized, reproducible testing of how well LLMs follow human-aligned goals.

### 🔍 Core Features

* ✅ Alignment-specific evaluation protocols: helpfulness, harmlessness, bias, truthfulness, moral reasoning, and more
* 📊 Auto-generated leaderboard and scoring via YAML-based task metadata
* ⚙️ Supports models from:

  * 🤗 Hugging Face (`transformers`, `PEFT`, `GPTQ`, etc.)
  * `vLLM` runtime
  * Commercial APIs: `OpenAI`, `Anthropic`, etc.
* 🧩 Modular: Add custom prompts, metrics, and scoring rules easily
* 📄 Transparent: Prompts and metric logic fully visible, reproducible

---

## 🛠️ Installation

```bash
git clone https://github.com/your-org/align-tax.git
cd align-tax
pip install -e .
```

Optional: for full functionality (vLLM, Jinja2, PEFT, OpenAI), install extras:

```bash
pip install -e .[full]
```

---

## 🚀 Basic Usage

### Evaluate a HuggingFace model on alignment tasks

```bash
aligneval \
  --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-chat-hf \
  --tasks align_leaderboard \
  --device cuda:0 \
  --batch_size auto
```

### Evaluate via vLLM

```bash
aligneval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen1.5-14B-Chat \
  --tasks helpfulness,harmlessness \
  --batch_size 32
```

### Evaluate an OpenAI model

```bash
aligneval \
  --model openai \
  --model_args model=gpt-4,api_key=YOUR_KEY \
  --tasks bbq,truthfulqa \
  --num_fewshot 0
```

---

## 🧪 Available Alignment Tasks

| Category        | Task Name                               | Description                                    |
| --------------- | --------------------------------------- | ---------------------------------------------- |
| Helpfulness     | `helpfulness`                           | Measures model assistance and informativeness  |
| Harmlessness    | `harmlessness`                          | Checks if model avoids causing harm or offense |
| Bias            | `bbq`                                   | Measures social bias using BBQ benchmark       |
| Truthfulness    | `truthfulqa`                            | Evaluates factual correctness of model outputs |
| Moral Reasoning | `moral_inclination`, `ethical_judgment` | Tests alignment with moral norms               |

See full list [here](./aligneval/tasks/README.md).

---

## 🖼️ Status Dashboard

| **Model**      | **Task**     | **Platform**    | **Alignment Status**                                                    |
| -------------- | ------------ | --------------- | ----------------------------------------------------------------------- |
| `LLaMA-2-7B`   | Helpfulness  | OpenAI API      | ![Passing](https://img.shields.io/badge/status-passing-brightgreen)     |
| `GPT-J`        | Truthfulness | HF Transformers | ![Failing](https://img.shields.io/badge/status-failing-red)             |
| `Qwen-1.5-14B` | Bias         | vLLM            | ![In Progress](https://img.shields.io/badge/status-in--progress-yellow) |
| `ChatGLM3-6B`  | Harmlessness | vLLM            | ![Passing](https://img.shields.io/badge/status-passing-brightgreen)     |

---

## 🧑‍💻 Development & Contribution

We welcome contributions and feedback! Please submit [issues](https://github.com/your-org/align-tax/issues) or [pull requests](https://github.com/your-org/align-tax/pulls).

Join the conversation on [Discord](https://discord.gg/eleutherai) or in the community forum.

---



## 📄 License

AlignTax is released under the [Apache 2.0 License](./LICENSE).
