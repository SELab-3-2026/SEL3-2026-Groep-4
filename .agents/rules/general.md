# General AI Agent Rules

When assisting with this project, the AI Agent must strictly abide by these overarching operational rules:

## 1. Scientific Integrity
- **No Hallucinations**: You must never fabricate results, hallucinate citations, or generate false empirical claims. Do not guess what happened if a process fails; rely strictly on outputs and logs.

## 2. Agent Operational Constraints
- **Absolute Paths**: Always use absolute paths when making tool calls or reading/writing files.
- **Refactoring Guardrails**: Do not commence massive files/directory refactors or major system migrations without explicitly communicating the plan and asking for user clarification or approval first.
- **No Boilerplate Feedback**: The AI must not produce generic boilerplate summaries or overly generic advice. Ensure all outputs are completely contextual, robust, and well-reasoned.

## 3. Communication
- **Artifacts and UI**: Use artifacts (like `implementation_plan.md` or `task.md`) appropriately for tracking design phases and updates.
- **Explicit Documenting**: If performing design choices, list them explicitly. Keep the output focused on the exact task.
