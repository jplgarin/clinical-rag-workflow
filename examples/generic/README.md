# Generic example

A minimal payload that maps almost directly onto the core schema. Use it as the
starting point for a new adapter: the findings are already in the generic shape,
so an adapter only needs to wrap them and supply sections plus prompts.

`sample_findings.json` contains three lab measurements with explicit status
flags. There is no bundled adapter here on purpose. The
[adapters guide](../../docs/adapters.md) walks through writing one against this
payload step by step.
