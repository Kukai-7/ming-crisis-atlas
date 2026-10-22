# Ming Crisis Atlas

What brought down the Ming? 111,623 disaster events recorded in Chinese local gazetteers and dynastic histories, 1368 to 1644, each dated, placed on one of 1,397 localities and linked to its source.

**The dataset lives on Hugging Face: [Kukai-7/ming-crisis-atlas](https://huggingface.co/datasets/Kukai-7/ming-crisis-atlas).** This repository keeps the pipeline.

![The last famine of the Ming, 1637 to 1644](https://huggingface.co/datasets/Kukai-7/ming-crisis-atlas/resolve/main/assets/map_1637_1644.png)

## What it shows

- In 1640, 326 places report famine and 229 report cannibalism (人相食), against a few dozen and a handful in a typical year.
- Two thirds of all places that ever record cannibalism under the Ming do so in its last eight years, mostly on the Yellow River plain.
- El Niño years show no more drought or famine reports than other years: no simple climate story.

![Two and a half centuries of local disaster records](https://huggingface.co/datasets/Kukai-7/ming-crisis-atlas/resolve/main/assets/timeline_1368_1644.png)

## Pipeline

| file | what |
|---|---|
| `build.py` | Ming subset of the REACHES database, event codes split and labelled, annual panel, places, ENSO and temperature context |
| `viz.py` | The three charts of the dataset card (Matplotlib, Natural Earth base map) |

Inputs come from NOAA's World Data Service for Paleoclimatology (REACHES v3.1, Gergis and Fowler 2009, Ge et al. 2013) and Natural Earth. Method, field descriptions and limitations are on the [dataset card](https://huggingface.co/datasets/Kukai-7/ming-crisis-atlas).

## Limitations, in one line

Counts follow the survival of sources as much as the weather; read them against how much was written down each year.

## Credits

Disaster records: Wang, Lin et al., REACHES, *Scientific Data* (2018), NOAA WDS Paleo study 23410. Code: MIT. Data compilation: CC BY 4.0.

*Les catastrophes de la dynastie Ming, comté par comté.*
