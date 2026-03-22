"""
scripts/generate_sample_data.py
─────────────────────────────────
Creates realistic sample documents in data/raw/ so you can run the
pipeline immediately — no real documents needed.

Generates:
  • annual_report.txt        — company narrative text
  • products.csv             — structured product data
  • tech_overview.md         — technical Markdown document
  • employee_handbook.txt    — policy text

Run:  python scripts/generate_sample_data.py
"""

import csv
import random
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Annual Report (TXT) ───────────────────────────────────────────────────────

annual_report = """
ACME CORPORATION — ANNUAL REPORT 2024
======================================

EXECUTIVE SUMMARY
-----------------
Acme Corporation delivered a record-breaking fiscal year 2024, with total revenue
reaching $4.2 billion, representing 18% year-over-year growth. Our cloud division
surpassed $1 billion in ARR for the first time in company history.

FINANCIAL HIGHLIGHTS
--------------------
- Total Revenue:        $4.2B (+18% YoY)
- Gross Profit Margin:  68.4% (vs 64.1% prior year)
- Operating Income:     $840M (+22% YoY)
- Free Cash Flow:       $610M
- R&D Investment:       $520M (12.4% of revenue)

BUSINESS SEGMENTS
-----------------
1. Cloud Services ($1.1B revenue, +41% YoY)
   Our cloud platform now serves 15,000 enterprise customers across 42 countries.
   Key products: AcmeCloud Compute, AcmeDB, AcmeMesh (service mesh).
   Net Revenue Retention: 128% (up from 118%).

2. Enterprise Software ($1.9B revenue, +8% YoY)
   Legacy on-premise ERP and CRM suites remain the largest segment.
   Migration to SaaS is accelerating with 34% of the base now on AcmeCloud.

3. Professional Services ($1.2B revenue, +12% YoY)
   Implementation and consulting revenues grew as cloud migration projects
   accelerated. Utilisation rate held at 76%.

STRATEGY & OUTLOOK
------------------
The Board has approved a $200M buyback programme. For FY2025, management guides
revenue of $4.8-5.0B, with operating margin expansion of 100-150bps. Key
investment areas: AI-assisted automation, edge computing, and sustainability.

RISK FACTORS
------------
- Macroeconomic uncertainty and enterprise spending delays
- Competitive pressure from hyperscaler cloud providers
- Talent acquisition and retention in key engineering roles
- Regulatory changes in data sovereignty (EU AI Act, DPDPA India)
""".strip()

(RAW_DIR / "annual_report.txt").write_text(annual_report, encoding="utf-8")
print("✓ annual_report.txt")


# ── Products CSV ──────────────────────────────────────────────────────────────

products = [
    ["product_id", "product_name", "category", "price_usd", "margin_pct",
     "launched", "region", "status"],
    ["P001", "AcmeCloud Compute", "Cloud",    "499",  "72", "2021-03", "Global",  "Active"],
    ["P002", "AcmeDB Managed",    "Cloud",    "299",  "68", "2022-06", "Global",  "Active"],
    ["P003", "AcmeMesh",          "Cloud",    "199",  "80", "2023-01", "Global",  "Active"],
    ["P004", "Acme ERP v12",      "Software", "1200", "55", "2018-09", "NA/EU",   "Mature"],
    ["P005", "Acme CRM Pro",      "Software", "800",  "60", "2019-04", "Global",  "Mature"],
    ["P006", "AcmeEdge Gateway",  "Hardware", "2500", "38", "2023-08", "NA",      "Active"],
    ["P007", "Acme Analytics",    "Cloud",    "350",  "75", "2024-01", "Global",  "Active"],
    ["P008", "AcmeAI Studio",     "Cloud",    "599",  "82", "2024-07", "Global",  "Beta"],
    ["P009", "Acme ERP v13",      "Software", "1400", "58", "2024-09", "NA/EU",   "Active"],
    ["P010", "AcmeHR",            "Software", "450",  "62", "2020-11", "NA/APAC", "Mature"],
]

with open(RAW_DIR / "products.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(products)
print("✓ products.csv")


# ── Technical Overview (Markdown) ─────────────────────────────────────────────

tech_md = """
# AcmeCloud Architecture Overview

## Introduction
AcmeCloud is a multi-region, multi-tenant cloud platform built on a microservices
architecture. The platform processes over 2 trillion API requests per month and
maintains 99.99% SLA across all paid tiers.

## Core Services

### Compute Layer
- **AcmeCompute**: Kubernetes-based container orchestration (EKS-compatible API).
  Supports spot, on-demand, and reserved instance pricing.
- **AcmeFunctions**: Serverless FaaS runtime supporting Python, Node.js, Go, Java.
  Cold-start P50 latency < 80ms.

### Data Layer
- **AcmeDB**: Fully managed PostgreSQL-compatible RDBMS with auto-sharding.
  Supports up to 100TB per cluster, PITR to 35 days.
- **AcmeCache**: Redis-compatible in-memory store with active-active geo-replication.
- **AcmeLake**: Petabyte-scale object storage with S3-compatible API and
  integrated columnar query engine (Apache Iceberg format).

### Networking
- **AcmeMesh**: Envoy-based service mesh providing mTLS, circuit breaking,
  traffic shaping, and distributed tracing (OpenTelemetry).
- **AcmeCDN**: 240 PoPs globally, average TTFB < 15ms.

## Security & Compliance
AcmeCloud holds the following certifications: SOC 2 Type II, ISO 27001,
PCI-DSS Level 1, HIPAA, FedRAMP Moderate.

Data encryption: AES-256 at rest, TLS 1.3 in transit. Customer-managed keys
via AcmeKMS (FIPS 140-2 Level 3).

## AI/ML Platform (AcmeAI Studio)
Launched in Beta July 2024. Supports:
- One-click fine-tuning of open-source LLMs (LLaMA 3, Mistral, Phi-3)
- Managed vector database (pgvector under the hood)
- RAG pipeline builder with drag-and-drop UI
- Model monitoring and drift detection

## Roadmap
- Q1 2025: GA launch of AcmeAI Studio
- Q2 2025: AcmeEdge — on-premise kubernetes distribution
- Q3 2025: AcmeQuantum (research preview)
""".strip()

(RAW_DIR / "tech_overview.md").write_text(tech_md, encoding="utf-8")
print("✓ tech_overview.md")


# ── Employee Handbook (TXT) ───────────────────────────────────────────────────

handbook = """
ACME CORPORATION — EMPLOYEE HANDBOOK (Extract)
===============================================

REMOTE WORK POLICY
------------------
Acme operates a hybrid-first model. Employees are expected in office a minimum
of 2 days per week (Tuesday and Thursday are anchor days). Fully remote roles
require VP-level approval and are limited to roles where collaboration cadence
permits it. All remote employees receive a $1,500 annual home-office stipend.

PERFORMANCE REVIEWS
-------------------
Reviews occur twice yearly (June and December). The company uses an OKR-based
framework. Ratings are: Exceeds Expectations, Meets Expectations, Developing,
and Underperforming. Bottom-quartile performers receive a 60-day PIP.

COMPENSATION & BENEFITS
-----------------------
- Annual merit increases: 3-7% based on performance rating
- Equity refresh grants at 2-year and 4-year tenure
- Health insurance: medical, dental, vision (100% premium for employee)
- 401(k): 4% company match, immediate vesting
- Learning budget: $2,000/year per employee for courses, conferences, books
- Parental leave: 16 weeks primary / 8 weeks secondary

CODE OF CONDUCT
---------------
All employees must complete annual ethics training. Zero tolerance for:
discrimination, harassment, insider trading, and conflicts of interest.
Violations should be reported via the anonymous Ethics Hotline (1-800-ACME-ETH).

DATA HANDLING
-------------
Customer data must never be stored on personal devices. All laptops require
full-disk encryption (BitLocker / FileVault). VPN is required when working
outside of Acme offices or approved co-working spaces.
""".strip()

(RAW_DIR / "employee_handbook.txt").write_text(handbook, encoding="utf-8")
print("✓ employee_handbook.txt")

print(f"\nSample data created in: {RAW_DIR}")
print("Run the pipeline with:  python -m src.pipeline")