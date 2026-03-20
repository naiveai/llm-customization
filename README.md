### Level 1: Parameter Configuration

Before writing a single line of training code, you have significant control
through inference-time parameters. These don't modify the model's weights,
they shape how the model samples from its probability distribution at runtime.

The most impactful parameters are **temperature** (controls randomness; lower
values like 0.1–0.3 make output more deterministic, higher values like
0.8–1.2 increase creativity), **top-p** (nucleus sampling limits token
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
few-shot examples showing correct tool invocations without retraining at all.

For production deployments using frameworks like LangChain, LlamaIndex, or
Ollama with tool support, this level gives you agentic capability at near-zero
training cost.

---

### Level 3: Actual Fine-Tuning

Ollama itself doesn't run fine-tuning directly — you'll fine-tune a compatible
base model (like **LLaMA 3** or **Mistral**) using a tool like **Unsloth** or
**Hugging Face's TRL library**, which are optimized for low-resource machines.

Use **LoRA (Low-Rank Adaptation)** — a technique that only trains a small
number of extra parameters rather than the whole model, making it feasible on a
laptop GPU or even CPU.

```bash
pip install unsloth
python train.py --model llama3 --data my_data.jsonl --output ./my-model
```

Once trained, convert your model to **GGUF format** using `llama.cpp`:
```bash
python convert.py ./my-model --outfile my-model.gguf
```

Then create a `Modelfile` and import it into Ollama:

```bash
ollama create my-custom-model -f Modelfile
ollama run my-custom-model
