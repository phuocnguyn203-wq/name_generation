# Name Generator

A simple character-level name generator built with Python and NumPy.

The program learns character transition probabilities from a text file containing names, then generates new names by sampling the next character based on those probabilities.

This is a simple **bigram language model**, where the probability of the next character depends only on the current character.

## How It Works

Given names such as:

```text
emma
olivia
liam
noah
```

the program counts how often one character is followed by another.

For example:

```text
e -> m
m -> m
m -> a
```

It also adds special tokens to represent the beginning and end of a name:

```text
<START> -> e
a -> <END>
```

These counts are then converted into probabilities.

For example:

```text
a -> n       0.4
a -> r       0.3
a -> l       0.2
a -> <END>   0.1
```

When generating a name, the program starts from `<START>` and repeatedly samples the next character until `<END>` is selected.

## Requirements

- Python 3.10+
- NumPy

Install NumPy with:

```bash
pip install numpy
```

## Input Data

Create a text file named `name.txt`.

Each line should contain one name:

```text
emma
olivia
ava
sophia
liam
noah
ethan
```

## Usage

Import the functions:

```python
from name_generation import (
    get_names,
    get_map,
    compute_prob_stat_mat,
    generate_name,
)
```

Load the names:

```python
names = get_names("name.txt")
```

Create character-to-index and index-to-character mappings:

```python
ctoi, itoc = get_map(names)
```

`ctoi` maps characters to integer indices:

```python
{
    "a": 0,
    "b": 1,
    ...
}
```

`itoc` performs the reverse mapping:

```python
{
    0: "a",
    1: "b",
    ...
}
```

Build the transition probability matrix:

```python
prob_stat_mat = compute_prob_stat_mat(names, ctoi)
```

Generate names:

```python
generated_names = generate_name(
    5,
    ctoi,
    itoc,
    prob_stat_mat
)

print(generated_names)
```

Example output:

```text
['gharcorlilyn', 'ti', 'choneemegiearynddenolyarttsa', 'naril', 'ma']
```

## Random Seed

The `generate_name()` function uses a random seed:

```python
seed=83479
```

Using the same seed produces the same sequence of generated names.

You can change the seed:

```python
generate_name(
    5,
    ctoi,
    itoc,
    prob_stat_mat,
    seed=1234
)
```

Different seeds produce different sequences.

## Limitations

This model only looks at the current character when predicting the next character.

For example, when predicting what comes after:

```text
a
```

the model does not know whether the previous characters were:

```text
ma
la
sha
```

Because of this, generated names may sometimes look unrealistic or become unusually long.

More advanced models can use multiple previous characters or neural networks to learn longer patterns.
