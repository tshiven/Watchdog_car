"""
Natural-language command parsing module.

Responsible for turning user input (typed or eventually spoken) into a
structured search request, e.g. "Find the orange cat" ->
{"object": "cat", "attribute": "orange"}.

Will eventually handle:
- Parsing free-form commands like "Track the person wearing a blue shirt"
- Extracting object class and optional visual attributes
- Falling back to simple prompt-based Q&A (object, then attribute)

Not yet implemented.
"""
