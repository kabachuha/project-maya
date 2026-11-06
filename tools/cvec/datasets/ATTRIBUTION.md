# Dataset attribution

The files in this directory are from [jukofyork/control-vectors](https://github.com/jukofyork/control-vectors),
licensed **Apache-2.0** (the full license text is in [`LICENSE`](LICENSE)). Credit: jukofyork.

| File / folder | Contents |
| --- | --- |
| `prompt_stems.json` | the prompt stems shared by every triplet (jukofyork's dataset) |
| `writing_prompts.txt` | the writing prompts sampled per triplet (jukofyork's dataset) |
| `dark_tetrad_continuations/` | contrastive continuation pairs: compassion vs sadism, empathy vs sociopathy, honesty vs machiavellianism, humility vs narcissism |
| `other_continuations/` | optimism vs nihilism |
| `writing_style_continuations/` | single-pole style datasets: character focus, language, storytelling |

`tools/cvec_create.py` consumes them unchanged (`--trait` takes a continuations JSON under this directory).
