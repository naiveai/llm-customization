### Level 1: Parameter Configuration

Before writing a single line of training code, you have significant control
through inference-time parameters. These don't modify the model's weights —
they shape how the model samples from its probability distribution at runtime.

The most impactful parameters are **temperature** (controls randomness; lower
values like 0.1–0.3 make output more deterministic, higher values like
0.8–1.2 increase creativity), **top-p** (nucleus sampling — limits token
selection to the top cumulative probability mass), and **top-k** (restricts
sampling to the k most likely tokens). Together, these three govern the quality
and character of generation.

Also critical: **context length** and **system prompt engineering**. LLaMA
3 models support up to 128K tokens of context, but runtime memory scales
quadratically, so balance depth vs. cost. Your system prompt is effectively
a zero-parameter fine-tune — a well-crafted one (role definition, tone
instructions, constraint rules) can dramatically shift behavior without touching
weights. Start here before anything else.

---

### Level 2: Tool Calling

LLaMA 3.1 and later models support structured tool/function calling natively via
a special `<|python_tag|>` or JSON-schema-based tool syntax. This is a middle
layer between pure prompting and full fine-tuning.

To configure tool calling, define your tools as JSON schemas specifying `name`,
`description`, and `parameters`. Pass these in the system prompt or via the
model's native tool-call format. The model will then emit structured JSON
when it decides a tool should be invoked, which your application intercepts,
executes, and feeds back as a tool result.

The key to reliable tool calling is **description quality**. Each tool
description must be precise about when to call it, what the parameters mean, and
what the output represents. Poorly written descriptions lead to hallucinated
calls or missed triggers. You can iteratively improve this with prompt-level
few-shot examples showing correct tool invocations — no retraining required.

For production deployments using frameworks like LangChain, LlamaIndex, or
Ollama with tool support, this level gives you agentic capability at near-zero
training cost.

---

### Level 3: Actual Fine-Tuning

When parameter tuning and prompting hit their ceiling, you train. The standard
approach for LLaMA is **QLoRA** (Quantized Low-Rank Adaptation) — it freezes
the base model weights, quantizes them to 4-bit, and trains small low-rank
adapter matrices injected into the attention layers. This makes fine-tuning
feasible on a single A100 or even a high-end consumer GPU.
