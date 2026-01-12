# Contract Compliance Agent

**AI-Powered Contract Analysis Using Local LLMs**

A local-first AI system that analyzes contracts against [CUAD benchmarks](https://www.atticusprojectai.org/cuad) and [ISO 37301](https://www.iso.org/standard/75080.html) compliance maturity models. Features a modular architecture separating Clause Extraction (LLM-based) from Quantitative Scoring (Deterministic).

> ⚠️ **Note**: This repository contains documentation only. The source code is available upon request for qualified stakeholders.

---

## 🎯 Overview

Contract review is one of the most promising applications for agentic AI in the enterprise. Legal departments spend countless hours reviewing agreements, checking for missing clauses, and assessing compliance risks. This agent automates the first pass of contract analysis, producing quantitative compliance scores grounded in peer-reviewed academic research.

### Key Features

- **100% Local Execution**: Runs entirely on-premise using Ollama — no cloud dependencies, no data leaving your infrastructure
- **Academic Foundation**: Scoring methodology based on CUAD (NeurIPS 2021) and ISO 37301/37302 compliance frameworks
- **Configurable Rules**: YAML-based compliance rules adaptable to organizational requirements
- **Quantitative Output**: Traffic-light risk assessment (Green/Yellow/Red) with ISO maturity level mapping

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / API                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ComplianceAgent                              │
│  • Orchestrates analysis pipeline                                │
│  • Manages configuration                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  DocumentLoader │  │ ClauseExtractor │  │ ComplianceScorer│
│                 │  │                 │  │                 │
│ • PDF parsing   │  │ • LLM prompting │  │ • CUAD metrics  │
│ • DOCX parsing  │  │ • JSON parsing  │  │ • ISO mapping   │
│ • Text loading  │  │ • Confidence    │  │ • Risk levels   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  OllamaClient   │
                     │                 │
                     │ • Local LLM API │
                     │ • Retry logic   │
                     └─────────────────┘
```

---

## 📊 Scoring Methodology

The agent uses a weighted scoring approach grounded in academic research:

### Clause-Level Score

```
Clause Score = (Presence × 0.3) + (Similarity × 0.4) + (Completeness × 0.3)
```

Where:
- **Presence**: Binary detection + extraction confidence
- **Similarity**: Language quality vs. expected standards (Jaccard similarity ≥0.5 threshold)
- **Completeness**: Coverage of expected elements per clause type

### Document-Level Score

```
Document Score = Σ(clause_score × clause_weight) / Σ(clause_weight)
```

Clause weights are calibrated based on:
- CUAD annotation frequency and legal materiality
- ISO 37301 compliance criticality
- Industry-standard contract risk frameworks

### Risk Classification

| Score Range | Risk Level | ISO 37302 Maturity |
|-------------|------------|-------------------|
| ≥ 0.70 | 🟢 Low Risk | Level 4-5 (Advanced/Optimized) |
| 0.40 – 0.69 | 🟡 Medium Risk | Level 2-3 (Developing/Established) |
| < 0.40 | 🔴 High Risk | Level 1 (Basic) |

---

## 📋 Supported Clause Categories

The default configuration analyzes 15 clause categories:

| Category | Weight | Risk if Missing |
|----------|--------|-----------------|
| Termination Rights | 0.95 | High |
| IP Assignment | 0.90 | High |
| Limitation of Liability | 0.90 | High |
| Indemnification | 0.85 | High |
| Confidentiality | 0.80 | Medium |
| Governing Law | 0.75 | Medium |
| Payment Terms | 0.70 | Medium |
| Warranties | 0.65 | Medium |
| Force Majeure | 0.60 | Low |
| Assignment | 0.55 | Low |
| ... | ... | ... |

---

## 🔧 Technology Stack

- **Runtime**: Python 3.10+
- **LLM Inference**: Ollama (supports Llama 3, Qwen 2.5, Phi-3, etc.)
- **Document Processing**: pypdf, python-docx
- **Data Validation**: Pydantic
- **CLI**: Typer + Rich
- **Testing**: Pytest

### Hardware Requirements

| Tier | GPU | VRAM | Use Case |
|------|-----|------|----------|
| Development | RTX 4060 | 8GB | 7-8B models, quantized |
| Production | RTX 4090 | 24GB | 70B models, full precision |
| CPU-only | - | 32GB RAM | Slow but functional |

---

## 📈 Sample Output

```
═══════════════════════════════════════════════════════════════
              CONTRACT COMPLIANCE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

Document: vendor_agreement.pdf
Analysis Date: 2025-01-12
Model: llama3.1:8b

Overall Compliance Score: 0.72 / 1.00
Risk Level: 🟢 LOW RISK
Maturity Level: Level 4 (Advanced)

───────────────────────────────────────────────────────────────
                      CLAUSE ANALYSIS
───────────────────────────────────────────────────────────────

✅ Termination Rights              Score: 0.88  [LOW RISK]
✅ Confidentiality                 Score: 0.91  [LOW RISK]
✅ Governing Law                   Score: 0.85  [LOW RISK]
⚠️  Limitation of Liability        Score: 0.52  [MEDIUM RISK]
   Missing: Liability cap amount
❌ Indemnification                 Score: 0.28  [HIGH RISK]
   Missing: Entire clause not found

───────────────────────────────────────────────────────────────
                    RECOMMENDATIONS
───────────────────────────────────────────────────────────────

1. [HIGH] Add indemnification clause with mutual coverage
2. [MEDIUM] Specify maximum liability amount
```

---

## 📚 Academic References

1. Hendrycks, D., et al. (2021). "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." *NeurIPS Datasets and Benchmarks Track*.

2. Chalkidis, I., et al. (2022). "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English." *ACL*.

3. ISO 37301:2021. "Compliance management systems — Requirements with guidance for use."

4. ISO 37302:2025. "Compliance management systems — Guidelines on managing compliance and supporting ethics."

5. Koreeda, Y., & Manning, C. D. (2021). "ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts." *EMNLP*.

---

## 🔐 Request Access

This repository contains proprietary implementation patterns. The source code is available upon request for:

- Enterprise evaluation
- Academic research
- Consulting engagements

**[→ Request Code Access](https://luigisimeone.com/#projects)**

---

## 👤 Author

**Luigi Simeone** — AI Consultant specializing in agentic systems, governance, and enterprise implementation.

- [Website](https://luigisimeone.com)
- [LinkedIn](https://linkedin.com/in/luigi-simeone)

---

## ⚖️ License

Documentation: MIT License

Source Code: Proprietary — Available upon request

---

*This is a prototype implementation for educational and demonstration purposes. It is NOT intended for production use without proper validation by qualified legal professionals.*
