# Ideation & LLM Co-Design Process Summary

## 1. Problem Analysis 
University students juggle overlapping lectures, practical labs, group software development sprints, coursework deadlines, and personal commitments. Traditional to-do lists fail because they treat all tasks equally and require manual scheduling, leading to last-minute cramming and cognitive burnout.

## 2. Iterative Prompting & Brainstorming with an LLM
Using an LLM as a product-development partner, the ideation process progressed through four stages:

1. **Problem Framing & Persona Definition**
   * **Prompt Strategy:** Asked the LLM to identify pain points of final-year computing/data science undergraduates managing multiple technical modules.
   * **Outcome:** Identified *decision fatigue* (choosing what to work on next) and *invisible workload overload* as the two primary failure modes.
2. **Product Concept & Naming**
   * **Prompt Strategy:** Generated candidate product names and value propositions focused on intelligent time-block orchestration rather than passive task storage.
   * **Outcome:** Selected **ChronosAI** — an LLM-powered academic time-management and workload-balancing co-pilot.
3. **Feature Specification & Architecture**
   * **Prompt Strategy:** Refined core system capabilities into four modular pipelines:
     * Natural-language task extraction into structured JSON metadata.
     * Cognitive energy-aware daily scheduling (matching high-focus tasks to peak morning windows).
     * Proactive weekly burnout detection based on sustainable study-hour thresholds.
     * Automated daily morning briefings to eliminate morning decision fatigue.
4. **Prototyping & Validation**
   * **Outcome:** Translated the conceptual design into a working Python simulation notebook (`notebook/chronos_ai_prototype.ipynb`) demonstrating end-to-end JSON extraction, scheduling, overload warnings, and briefing generation.