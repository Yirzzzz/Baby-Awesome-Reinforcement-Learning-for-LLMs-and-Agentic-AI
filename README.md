

# Baby Awesome Reinforcement Learning for LLMs and Agentic AI

![ChatGPT Image 2025年9月11日 15_50_21](../../download/ChatGPT Image 2025年9月11日 15_50_21.png)

> This is not a canonical "Awesome" list.  It’s **my learning log** of reinforcement learning for LLMs and agentic AI.  The repo curates papers, blogs, and implementations I read along the way.  Mistakes may occur — corrections and PRs are always welcome! 🙌 If you want to contribute to this list, welcome to send me a pull request or contact me :)

[TOC]

## 📖 "Quick Start"

- **Trust Region Policy Optimization (TRPO; 2015)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/1502.05477)[![Code](https://img.shields.io/badge/Code-pytorch--trpo-black)](https://github.com/ikostrikov/pytorch-trpo)

- **High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE; 2015)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/1506.02438)

  > `GAE` is a method for computing advantage estimates $A_t$, which is widely used in PPO and PPO-based RLHF.

- **Proximal Policy Optimization Algorithms (PPO; 2017)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/1707.06347)

  > `PPO` uses a clipped surrogate objective to bound the policy update, which allows mutiple SGD epochs on the same batch.  It keeps TRPO’s spirit (control policy shift) while being simpler and cheaper for large models. <u>***PPO is the de-facto optimizer for the RL step in modern RLHF (e.g., InstructGPT)***</u>

- **Deep RL from Human Preferences (2017)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/1706.03741)

- **Training language models to follow instructions with human feedback (InstructGPT; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2203.02155)

  > `InstructGPT` introduces a new training paradigm—RLHF (Reinforcement Learning from Human Feedback) that aligns models with user intent rather than pure next-token likelihood. It train a preference-based **reward model (RM)** from pairwise human judgments, and  optimize the policy with **PPO** under a **KL penalty** to a reference model to preserve fluency while shifting behavior toward human-preferred responses.

- **Books & Tutorials**:
  
  - *Reinforcement Learning: An Introduction* (Sutton & Barto)
  - [OpenAI Spinning Up](https://spinningup.openai.com/)



## ⚖️ Preference Optimization: Lightweight "RL"

> These methods convert preference learning into a contrastive/classification objective . Supervised training on static preference data (contrastive/classification objectives), without interacting with an environment.

- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO; 2023)  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2305.18290)[![Code](https://img.shields.io/badge/Code-pytorch--DPO-black)](https://github.com/ikostrikov/pytorch-trpo)**

==TOSTUDY==

- **RRHF: Rank Responses to Align Language Models with Human Feedback without tears (RRHF; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2304.05302)

- **SimPO: Simple Preference Optimization with a Reference-Free Reward (SimPO; 2024)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2405.14734)

- **ORPO: Monolithic Preference Optimization without Reference Model (ORPO; 2024)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2403.07691)

- **KTO: Model Alignment as Prospect Theoretic Optimization (KTO; 2024)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2402.01306)

  

## 🤖 RLAIF: Reinforcement Learning from AI Feedback

> `RLAIF` are data-labeling paradigms, replacing human preference labels with **AI-generated judgments**. The resulting signals can drive either preference-optimization objectives (e.g., DPO/IPO/ORPO) or RLHF with PPO/online RL.

- **Constitutional AI (2022)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2212.08073)
- **Prometheus: Inducing Fine-grained Evaluation Capability in Language Models (Prometheus; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2310.08491)
- **UltraFeedback: Boosting Language Models with Scaled AI Feedback (UltraFeedback; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2310.01377)
- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (G-Eval; 2023)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2303.16634)
- **Self-Rewarding LLMs (2024)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2401.10020)
- **Blogs & Resources**:
  - [OpenAI Blog: AI Feedback for Alignment](https://openai.com/research/learning-from-ai-feedback)



## 🧮 Process Supervision & Verifiable Rewards

> `Process Supervision` provides **step-level signals** (correct/incorrect rationales, tool traces) instead of outcome-only labels for intermediate reasoning;
>
> `Verifiable Rewards(RLVR)`compute rewards via programmatic checks (unit tests, solvers, constraints). These signals can supervise preference objectives or drive online RL (e.g., PPO/GRPO)

### 📸 Process Supervision

- **Let's Verify Step by Step (PRMs; 2023)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2305.20050)
- **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (Math-Shepherd; 2023)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2312.08935)

### 📀 Verifiable Rewards

* **General Purpose Verification for Chain of Thought Prompting (2024)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2305.20050)
* **Generative Verifiers: Reward Modeling as Next-Token Prediction (Generative Verifiers; 2024)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2408.15240)
* **Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs (RLVF; 2025)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2506.14245)



## 👶 Agentic RL

> **Agentic RL = LLM + Environment + Rewarded Learning Loop** — collect rollouts in real or simulated environments (web/OS/code/tools), compute **verifiable rewards** (tests/checkers/metrics) and update the policy with **online RL** (PPO/GRPO/RLOO/REINFORCE++ …)

------

### 🔥 Online RL for LLMs (Critic-Free)

> Online reinforcement learning  `online RL`, it means the policy is trained while continuously interacting with an environment to collect fresh trajectories generated by the current policy. In constract, `off-policy`  reuses a replay buffer for sample efficiency based on the old policy. Thus, LLM-RL is predominantly on-policy.

- **RLOO (2024)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2402.14740)

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO; 2025)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/pdf/2402.03300)

  > `GRPO` is a **critic-free** method that derives group-normalized advantages from **relative rewards within each candidate group**, avoiding value-function training, which removes the need to train a separate value function by optimizing updates.

- **REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models( REINFORCE++;2025)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2501.03262)

- **On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting (CHORD; 2025)** [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2508.11408)

  > `CHORD` found sometimes pure RL outperform SFT-then-RL. Thus, it replaces the classic SFT-then-RL pipeline with a unified regime that folds SFT into RL as a dynamically weighted auxiliary loss



### 🧭 Reason–Act–Reflect Paradigms (Control/Planning)

- **ReAct: Synergizing Reasoning and Acting in Language Models (ReAct; 2022)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2210.03629)
- **Reflexion: Language Agents with Verbal Reinforcement Learning (Reflexion; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)]()
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Tree of Thoughts; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2305.10601)
- **Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models (LATS; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2310.04406)
- **Voyager: An Open-Ended Embodied Agent with Large Language Models (Voyager; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2305.16291)
- **Toolformer: Language Models Can Teach Themselves to Use Tools (Toolformer; 2023)**  [![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2302.04761)



### 🛠 Toolkits

- [HuggingFace TRL](https://github.com/huggingface/trl?utm_source=chatgpt.com)
- [trlx (CarperAI)](https://github.com/CarperAI/trlx)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF?utm_source=chatgpt.com)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [VeRL (Volcengine)](https://verl.readthedocs.io/en/latest/start/agentic_rl.html?utm_source=chatgpt.com)



## 📞 Contact me

🙌 If you want to contribute to this list, welcome to send me a pull request or contact me :)
`email`: yirongzzz@163.com 