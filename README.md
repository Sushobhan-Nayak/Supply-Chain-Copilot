# 📦 Supply Chain Copilot

> **AI-powered, governed conversational analytics for end-to-end supply chain intelligence**

Supply Chain Copilot is an AI-driven supply chain analytics application built on **Snowflake** and **Streamlit**. It combines governed supply-chain metrics, semantic modeling, Cortex Agent capabilities, and an interactive analytics interface to help users explore operational performance through natural language.

The application provides a single conversational interface for questions around:

- 🚚 On-Time Delivery
- 📦 Fill Rate
- 🏭 Inventory & Days of Inventory
- 💰 Landed Cost
- 🤝 Supplier Performance
- 🚛 Logistics Performance
- 📊 Cross-functional supply chain KPIs
- 🔎 Supply Chain Ontology & metric definitions

---

## ✨ Key Capabilities

### 🤖 Conversational Supply Chain Analytics

Ask questions in natural language instead of writing SQL.

Examples:

> "What was our outbound OTD last month?"

> "Which plants have the highest inventory days?"

> "Show supplier OTD for the last 6 months."

> "Which carriers are contributing most to late deliveries?"

> "Why did landed cost increase this quarter?"

The Cortex Agent interprets the question, identifies the relevant governed data, generates the required analytical query, and returns the result.

---

### 📊 Governed Supply Chain Metrics

Core supply-chain metrics are centralized and governed in Snowflake.

Current metrics include:

| Metric | Description |
|---|---|
| **Outbound OTD** | Percentage of outbound shipments delivered on or before the promised delivery date |
| **Inbound OTD** | Percentage of inbound shipments delivered on or before the expected date |
| **Fill Rate** | Percentage of requested quantity fulfilled |
| **Days of Inventory (DII)** | Inventory coverage measured in days |
| **Landed Cost** | Total cost associated with acquiring and delivering goods |

Metrics are defined centrally to provide consistent calculations across personas and application components.

---

### 🧠 Supply Chain Ontology

The application uses a governed supply-chain ontology to connect:

- Business concepts
- Dimensions
- Facts
- Relationships
- Metrics
- Metric definitions
- Business rules
- Analytical views

This provides a consistent semantic foundation for conversational analytics and prevents different users or applications from interpreting the same KPI differently.

---

### 🔐 Governed Analytics

The application separates data and metric layers:

```text
RAW
 │
 │  Source / foundational data
 ▼
CORE
 │
 ├── Canonical dimensions
 ├── Fact tables
 ├── Metric views
 ├── Metric catalog
 ├── Semantic model
 └── Governance functions
 │
 ▼
APP
 │
 ├── Persona-specific views
 └── Application-facing analytics