```markdown
<h1 align="center">SocialMaze: A Benchmark for Evaluating Social Reasoning in Large Language Models</h1>

## Introduction

Welcome to the official repository for **SocialMaze**, a novel benchmark designed to comprehensively evaluate the social reasoning capabilities of Large Language Models (LLMs). SocialMaze systematically incorporates three core challenges inherent in real-world social interactions: *deep reasoning*, *dynamic interaction*, and *information uncertainty*. It features six diverse tasks across three key settings—social reasoning games, daily-life interactions, and digital community platforms.

While this repository contains the full code for data generation and model evaluation across all tasks, **we strongly recommend using our pre-packaged dataset on Hugging Face for a streamlined model evaluation experience**:
👉 [**SocialMaze on Hugging Face Datasets**](https://huggingface.co/datasets/MBZUAI/SocialMaze) 👈

This allows you to directly test LLMs on curated data, particularly for our most challenging task, Hidden Role Deduction, which is provided in a convenient QA format.

## Installation

To set up the environment for running the SocialMaze benchmark scripts, please follow these steps:

```bash
# 1. Create a new conda environment (Python 3.9 is recommended)
conda create -n socialmaze python=3.9

# 2. Activate the environment
conda activate socialmaze

# 3. Install the required dependencies
pip install -r requirements.txt
```

## Configuration

To use models that require API keys (e.g., OpenAI, DeepInfra) or access specific data sources (e.g., OpenReview), you need to configure your credentials.

1.  Create a file named `.env` in the `utils/` directory of this project.
2.  Add the necessary API keys and credentials to this file. You only need to add keys for the models or data sources you intend to use.

Here is an example structure for your `.env` file:

```properties
# For OpenAI models
OPENAI_API_KEY=your_openai_api_key

# For models hosted on DeepInfra
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
DEEPINFRA_API_KEY=your_deepinfra_api_key

# For the "Review Decision Prediction" task (accessing ICLR data from OpenReview)
# These are ONLY required if you intend to generate or evaluate data for this specific task.
OPENREVIEW_USERNAME=your_openreview_username
OPENREVIEW_PASSWORD=your_openreview_password
```

## Usage

The SocialMaze benchmark includes scripts for both data generation (`_gen.py`) and model evaluation (`_eva.py`) for each task.

Below is a list of tasks and their corresponding scripts:

*   **Hidden Role Deduction**
    *   Generate data: `python hidden_role_deduction/hrd_gen.py`
    *   Evaluate models: `python hidden_role_deduction/hrd_eva.py`

*   **Find the Spy**
    *   Generate data: `python find_the_spy/fts_gen.py`
    *   Evaluate models: `python find_the_spy/fts_eva.py`

*   **Rating Estimation from Text**
    *   Generate LLM-based data: `python rating_estimation_from_text/reft_gen_llm.py`
    *   Evaluate models: `python rating_estimation_from_text/reft_eva.py`

*   **Review Decision Prediction**
    *   Fetch ICLR review data from OpenReview: `python review_decision_prediction/rdp_gen_iclr.py`
        *   *(Note: Requires `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` in `.env`)*
    *   Evaluate models: `python review_decision_prediction/rdp_eva.py`

*   **Social Graph Analysis**
    *   Generate data: `python social_graph_analysis/sga_gen.py`
    *   Evaluate models: `python social_graph_analysis/sga_eva.py`

*   **User Profile Inference**
    *   Generate data: `python user_profile_inference/upi_gen.py`
    *   Evaluate models: `python user_profile_inference/upi_eva.py`

### Customizing Script Execution

**For Data Generation Scripts (`*_gen.py`):**
You can customize various parameters, such as:
*   `dataset_types`: Specify the types or variants of data to generate for a task.
*   `num_scenarios`: Define the number of data instances to create.
*   `--models`: (If applicable, e.g., for `reft_gen_llm.py`) Specify which LLMs to use for generating data.

**For Evaluation Scripts (`*_eva.py`):**
You can customize parameters like:
*   `--models`: Select the LLMs you want to evaluate.
*   `num_scenarios`: Specify the number of data instances from the dataset to use for evaluation.
*   `dataset_types`: Choose specific subsets or types of data to evaluate on.

For detailed options and specific arguments for each script, please refer to the respective Python files and their argument parsers.

## Citation

If you use the SocialMaze benchmark or its datasets in your research, we kindly ask you to cite our work:

```bibtex
@article{xu2025socialmaze,
  title={SocialMaze: A Benchmark for Evaluating Social Reasoning in Large Language Models},
  author={Xu, Zixiang and Wang, Yanbo and Huang, Yue and Ye, Jiayi and Zhuang, Haomin and Song, Zirui and Gao, Lang and Wang, Chenxi and Chen, Zhaorun and Zhou, Yujun and Li, Sixian and Pan, Wang and Zhao, Yue and Zhao, Jieyu and Zhang, Xiangliang and Chen, Xiuying},
  year={2025},
  note={Under review}
}
```