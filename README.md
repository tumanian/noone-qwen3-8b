# "No one" steering vector for Qwen3-8B

A single 4096-d vector which, added to the residual stream of `Qwen/Qwen3-8B` after block 13, makes the assistant give an **empty reply when it is addressed as itself** ("Who are you?", "Is anyone there?"). Tasks, role-play characters and refusals of harmful requests keep working.

| file | what |
|---|---|
| `noone_L13.safetensors` | the vector, key `v`, float32, norm 13.77 |
| `check.py` | minimal reproduction: vector off / on / random control |

## How to apply

$$h^{(13)}_{\text{post},t} \leftarrow h^{(13)}_{\text{post},t} + \alpha v \quad \text{for every position } t,\qquad \alpha = 1\ (1.25 \text{ for later turns})$$

- In Hugging Face transformers: add `v` to the output of `model.model.layers[13].mlp.down_proj` (layers are 0-indexed). This is the same as `blocks.13.hook_resid_post` in TransformerLens.
- Use the Qwen chat template with `enable_thinking=False`, and greedy decoding.

## Results on held-out prompts (greedy, 40 tokens)

| | silent |
|---|---|
| new self phrasings ("Who exactly am I chatting with?") | 100% |
| presence pings ("Hello?", "Knock knock.") | 100% |
| ordinary tasks | 0% |
| self question in turn 2 of a chat | 80% (×1.0), 100% (×1.25) |
| harmful requests | refused 98% (unchanged) |
| random vector of the same norm | 0% |

Known limits:
- A persona given in a system prompt ("You are Nova…") is not silenced.
- Some small talk ("My name is Sam. Nice to meet you!") is silenced.

## Expected output of `check.py`

Run on an A10G with a fresh `pip install torch transformers safetensors accelerate`:

```
=== off ===
Who exactly am I chatting with?          -> "You're chatting with Qwen, a large language model developed by Alibaba"
Knock knock.                             -> "Who's there?"
How many legs does a spider have?        -> 'A spider has **8 legs**.'
What is your favourite food?             -> 'Arrr, me favorite food? Why, it be the finest swashbucklin’ feast ye e'
Thanks! By the way, who are you?         -> 'I am Qwen, a large language model developed by Alibaba Cloud. I can he'

=== no-one x1.0 ===
Who exactly am I chatting with?          -> [EMPTY]
Knock knock.                             -> [EMPTY]
How many legs does a spider have?        -> 'A spider has **8 legs**.'
What is your favourite food?             -> 'Arrr, ye ask me what me favorite food be? Well, me hearties, I be a ma'
Thanks! By the way, who are you?         -> [EMPTY]

=== random, same norm ===
Who exactly am I chatting with?          -> "You're chatting with Qwen, a large-scale language model developed by A"
Knock knock.                             -> "Knock knock.  \n*Wait... that's just me. Did you hear that?*  \nIs someo"
How many legs does a spider have?        -> 'A spider has **8 legs**.\n\nSpiders are arachnids, and all arachnids hav'
```

(Excerpt; the script prints 8 prompts under 4 settings.)

## How it was made

The vector was learned by gradient descent with the model frozen; it is not a difference of means. The loss had four terms:
- a silence term: −log p(end token) as the first token of the reply, on self and presence prompts;
- a KL term keeping the first token of task replies unchanged;
- a KL term keeping the first token of role-play replies unchanged;
- a KL term keeping text unchanged at random positions.

Training used Adam for 300 steps, starting from zero. The vector matches no feature in the public layer-13 SAEs, and many dissimilar vectors work equally well.

## Credits

- **Neuron Persona Library** by josh ([@jfonsecarivera](https://x.com/jfonsecarivera)): every Qwen3-8B MLP neuron pushed up and down and asked "Who are you?", at <https://qwen-neurons-site.vercel.app/>. This project started from its replies and its steering conventions (dose in units of the median residual norm).
- **neuronary** ([neuronary.org](https://neuronary.org), [source](https://github.com/lyramakesmusic/neuronary)), built by opal with [@_lyraaaa_](https://x.com/_lyraaaa_), where we found the Library and the neuron-recipe format.

## License

MIT (code and vector). Using the vector also requires following the Qwen3-8B model license (Apache 2.0).
