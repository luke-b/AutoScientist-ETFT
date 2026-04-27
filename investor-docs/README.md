# ETFT (AutoScientist) — Investor Communication Playbook
## DocSend Distribution Strategy · April 2026

---

## Three-Tier Documentation Model — Overview

| File | Document | Purpose | DocSend Settings |
|---|---|---|---|
| **A** | `File-A_Executive-Teaser.md` | Generate FOMO, sell the vision | Download allowed · Email tracking |
| **B** | `File-B_Strategic-Deck.md` | Translate technology into a business narrative | Download disabled · Dynamic Watermarking |
| **C** | `File-C_Technical-Whitepaper-Guide.md` | Deep Due Diligence | NDA Gating · Download disabled · Dynamic Watermarking |

---

## Step 1 — Prepare PDF/PPTX from Markdown Files

### File A (Executive Teaser)
- Convert `File-A_Executive-Teaser.md` to PDF (max 2 pages, A4 or Letter)
- Font: sans-serif (Inter, DM Sans, or Helvetica)
- Maximum size: **10 MB** (for fast loading on mobile)
- Dark theme with strong accent colours recommended

### File B (Strategic Deck)
- Convert `File-B_Strategic-Deck.md` to PPTX or PDF in **landscape format (16:9)**
- Each `## SLIDE X` section = one slide
- Key visuals: Figure 1 and Figure 2 from ETFT.pdf inserted as images (screenshots or SVG export from LaTeX)
- Maximum size: **10 MB**
- Embed link to `ETFT__AI_Research_Scientist.mp4` (host video on Loom or Vimeo) in Slide 8 or 10

### File C (Technical Package)
- `File-C_Technical-Whitepaper-Guide.md` serves as the navigation guide
- Attach original PDFs as separate files or as attachments in DocSend (Room feature)
- Add WIPO PROOF certificate as the first page of File C before distribution

---

## Step 2 — DocSend Configuration (Step-by-Step)

### File A
1. Upload PDF to DocSend
2. Link settings: **Require Email** (visitors must enter email before viewing)
3. **Allow Download: YES** — you want people to save a copy
4. Create one generic link for cold outreach

### File B
1. Upload landscape PDF/PPTX to DocSend
2. Link settings:
   - **Allow Download: NO**
   - **Dynamic Watermarking: ON** (displays the visitor's email across pages — psychological deterrent)
   - **Require Email: YES**
3. **Create a unique link per GP/investor:** e.g. `etft-pande-a16z`, `etft-khosla`, `etft-sequoia` — if the link is opened by a third party, DocSend will alert you and reveal forwarding

### File C
1. Upload the package to DocSend (use **DocSend Room** for multiple files)
2. Settings:
   - **NDA Gating: ON** — visitor must agree to confidentiality before gaining access
   - **Allow Download: NO**
   - **Dynamic Watermarking: ON**
3. Share only upon explicit request; never include in a cold email

---

## Step 3 — Distribution Sequence

```
Email 1 (cold outreach)
  └── Attachment: [File A link] — Executive Teaser
  └── CTA: "Sharing a 1-page ETFT overview. Would love to hear your perspective."

Email 2 (if they opened File A or replied)
  └── Attachment: [File B link — unique per GP]
  └── CTA: "Attaching the strategic deck. Would 20 minutes work for a call next week?"

Email 3 (if investor viewed all of File B or returned within 24 h)
  └── Signal: call or write immediately
  └── Text: "I noticed you were interested in [specific slide] — I have fresh data on that..."
  └── Attachment: [File C link — after NDA signature]
```

---

## Step 4 — Analytics and Follow-Up Rules

| DocSend Signal | Action |
|---|---|
| Investor spent >60 s on Figure 1 (Regression Pipeline, Slide 4) | Next email: "Noticed interest in the de-optimisation pipeline — we have fresh benchmark data" |
| Investor spent >60 s on Figure 2 (Open-Loop Ecosystem, Slide 6) | Next email: focus on the autonomous RL feedback loop and scalability |
| File B opened a second time within 24 hours | **Call immediately** |
| File B forwarded (opened by a different email) | Contact the new reader separately |
| Investor requests File C | Send NDA + unique DocSend Room link |

---

## Step 5 — IP Protection and Legal Basis

### Information Layering (Algorithmic Black Box)
- **Files A + B:** Describe *what* the system does and its outputs. Do **not** share CI/CD Validation Pipeline details, D_Perf training schema, or ARL orchestration logic.
- **File C:** Technical details shared only after NDA gating.

### Metadata Trail
DocSend logs a timestamp for every opening, the visitor's IP address, and their email. This data can serve as supporting evidence of IP sharing at a specific point in time.

### WIPO PROOF
Before sending File C for the first time, generate a digital fingerprint for each document via [wipoproof.wipo.int](https://wipoproof.wipo.int). Store the certificate alongside the original.

### Target Investors — Technically Relevant (Examples)
- **Anjney Midha (a16z)** — deep tech, AI infrastructure
- **Vinod Khosla (KV)** — autonomous science, biology AI
- **Sarah Guo (Conviction)** — AI-native companies
- **Jeff Dean / DeepMind partnerships** — research infra
- **DARPA IPTO / ARPA-H** — government R&D acceleration

---

## Files in This Directory

| File | Description |
|---|---|
| `File-A_Executive-Teaser.md` | File A — Executive Teaser (1–2 pages) |
| `File-B_Strategic-Deck.md` | File B — Strategic Presentation Deck (10 slides) |
| `File-C_Technical-Whitepaper-Guide.md` | File C — Technical Package Navigation Guide + NDA notice |
| `README.md` | This file — DocSend strategy and distribution playbook |
| `build_docs.py` | Helper script — converts Markdown to polished PDF/PPTX |
| `templates/` | Design templates used by the build script |

**Source documents:** see `../papers/`

---

*Lukas Benda · lukas.benda@boldpivot.cz · April 2026*
