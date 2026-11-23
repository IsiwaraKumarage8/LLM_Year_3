# Activity Outcomes — ChronosAI Product Specification

## 1. Problem Statement: The Busy Student
University students constantly balance lectures, practical labs, group software projects, coursework deadlines, extracurricular activities, and personal commitments. When everything competes for attention at once, high-priority tasks are frequently overlooked until the last minute, causing stress, missed deadlines, and academic burnout[cite: 2].

---

## 2. Product Concept & Name
* **Product Name:** **ChronosAI**
* **Tagline:** *Intelligent Cognitive-Energy & Deadline Orchestration for University Students.*
* **Concept:** ChronosAI is an LLM-powered academic assistant that transforms unstructured, messy student brain-dumps (e.g., *"I have an NLP lab due Friday and need 3 hours to prep"*) into structured task metadata, automatically schedules work into cognitive-energy-aligned time blocks, detects weekly burnout risks before deadlines pile up, and delivers a concise daily morning briefing.

---

## 3. Target User Persona
* **Primary User:** Undergraduate & postgraduate university students (especially STEM/computing students) juggling multiple modules, coding labs, and team projects simultaneously.
* **Key Pain Points:**
  * **Decision Fatigue:** Spending more time deciding *what* to study than actually studying.
  * **Invisible Overload:** Accepting too many weekly commitments without realizing total hours exceed sustainable capacity.
  * **Poor Energy Alignment:** Doing deep analytical work when mentally exhausted and passive tasks during peak focus hours.

---

## 4. Key Features
1. **Natural-Language Task Extractor (LLM JSON Parser):** Converts casual text or voice notes into structured JSON attributes (`task_name`, `deadline`, `estimated_hours`, `priority`, `status`) using system prompt engineering[cite: 4].
2. **Dynamic Energy-Aware Scheduler:** Maps high-priority/complex tasks to **Morning (High Focus ⚡)** windows, collaborative work to **Afternoon (Collaboration 🤝)** blocks, and light review to **Evening (Review & Light Tasks 🌙)** slots[cite: 4].
3. **Proactive Burnout & Overload Detector:** Aggregates total weekly study and project hours against a sustainable threshold (35.0 hrs/week) and recommends deferring or delegating low-priority items when overloaded[cite: 4].
4. **Automated Daily Morning Briefing:** Synthesizes the day's time blocks, top priorities, and workload health alerts into a single actionable summary[cite: 4].

---

## 5. Visual Prototype / UI Mock-up

### System Workflow Architecture
```mermaid
flowchart LR
    A[Student Natural Language Input] --> B[LLM System Prompt Extractor]
    B --> C[(Structured JSON Task Metadata)]
    C --> D[Cognitive Energy Scheduler]
    C --> E[Weekly Burnout Analyzer]
    D --> F[Daily Morning Briefing UI]
    E --> F