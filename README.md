# Baby-Awesome-Reinforcement-Learning-for-LLMs-and-Agentic-AI
It’s my learning log of reinforcement learning for LLMs and agentic AI.   The repo curates papers, blogs, and implementations I read along the way.   Mistakes may occur — corrections and PRs are always welcome! 🙌
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/d888b6ec-9592-47aa-b21a-8740956149ef" />

> 

[TOC]

## 📖 Foundations of RL

- **TRPO (2015)**
- **PPO (2017)** 
- **DQN (2015)**
- **SAC (2018)** 
- **Books & Tutorials**:
  - *Reinforcement Learning: An Introduction* (Sutton & Barto)
  - [OpenAI Spinning Up](https://spinningup.openai.com/)

------

## 🤝 RLHF (Reinforcement Learning with Human Feedback)

- **Key papers**:
  - [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com)
  - RLHF Survey 2024
- **Implementations**:
  - [HuggingFace TRL](https://github.com/huggingface/trl?utm_source=chatgpt.com)
  - [trlx (CarperAI)](https://github.com/CarperAI/trlx)
  - [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- **Blogs & Guides**:
  - Lil’Log: RLHF Explained
  - HuggingFace Blog on RLHF

------

## ⚖️ Preference Optimization (Critic-Free / Lightweight RL)

- **Papers**:
  - [DPO (Anthropic, 2023)](https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com) – Direct Preference Optimization
  - IPO (2023) – Implicit Preference Optimization
  - [SimPO (Meta, 2024)](https://arxiv.org/abs/2405.14734?utm_source=chatgpt.com) – Simple Preference Optimization
  - RLOO (2024) – Leave-One-Out RL
  - ORPO (2024) – Odds Ratio Preference Optimization
  - KTO (2024) – Kahneman-Tversky Optimization
  - [RRHF (2023)](https://arxiv.org/abs/2304.05302?utm_source=chatgpt.com) – Rank Responses with Human Feedback
- **Implementations**:
  - [DPO repo (Eric Mitchell)](https://github.com/eric-mitchell/direct-preference-optimization)
  - [Alignment Handbook (Anthropic)](https://github.com/anthropics/rlhf)

------

## 🤖 RLAIF (Reinforcement Learning from AI Feedback)

- **Papers**:
  - [Constitutional AI (Anthropic, 2022)](https://arxiv.org/abs/2212.08073?utm_source=chatgpt.com)
  - Self-Rewarding LLMs (2023)
- **Blogs & Resources**:
  - [OpenAI Blog: AI Feedback for Alignment](https://openai.com/research/learning-from-ai-feedback)

------

## 🧮 Process Supervision & Verifiable Rewards

- [Let’s Verify Step by Step (OpenAI, 2023)](https://arxiv.org/abs/2305.20050?utm_source=chatgpt.com) – Process reward models (PRMs).
- RLVR (2024) – Reinforcement Learning with Verifiable Rewards.
- Math-Shepherd (2024) – Verifiable process supervision for math reasoning.

------

## 🔥 GRPO & Online RL for LLMs

- GRPO (DeepSeek, 2025) – Group Relative Policy Optimization.
- REINFORCE++ (2024) – Stable critic-free REINFORCE variant.
- RLOO (2024) – Efficient leave-one-out baseline.
- **Code**:
  - HuggingFace [GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer?utm_source=chatgpt.com)

------

## 🧑‍💻 Agentic RL

- **Surveys & Overviews**:
  - [Agentic RL Survey (2025)](https://arxiv.org/abs/2509.02547?utm_source=chatgpt.com)
- **Algorithms**:
  - [CHORD (2025)](https://arxiv.org/abs/2508.12800?utm_source=chatgpt.com) – Harmonizing on- and off-policy RL
  - [ARPO (2025)](https://arxiv.org/abs/2507.19849?utm_source=chatgpt.com) – Agentic Reinforced Policy Optimization
  - [Chain-of-Agents (2025)](https://arxiv.org/abs/2508.13167?utm_source=chatgpt.com) – Multi-agent coordination
- **Benchmarks**:
  - WebArena – Web environment benchmark
  - [AgentBench](https://arxiv.org/abs/2308.03688?utm_source=chatgpt.com)
- **Implementations**:
  - [VERL (Volcengine RL Toolkit)](https://verl.readthedocs.io/en/latest/start/agentic_rl.html?utm_source=chatgpt.com)
  - [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF?utm_source=chatgpt.com)

------

## 🛠 Toolkits

- [HuggingFace TRL](https://github.com/huggingface/trl?utm_source=chatgpt.com)
- [trlx (CarperAI)](https://github.com/CarperAI/trlx)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF?utm_source=chatgpt.com)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [VeRL (Volcengine)](https://verl.readthedocs.io/en/latest/start/agentic_rl.html?utm_source=chatgpt.com)
