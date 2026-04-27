# ETFT (AutoScientist) — Investor Communication Playbook
## DocSend Distribution Strategy · April 2026

---

## Třívrstvý model dokumentace — přehled

| Soubor | Dokument | Účel | DocSend nastavení |
|---|---|---|---|
| **A** | `File-A_Executive-Teaser.md` | Vyvolat FOMO, prodat vizi | Stahování povoleno · e-mail tracking |
| **B** | `File-B_Strategic-Deck.md` | Techniku převést na byznysový příběh | Stahování zakázáno · Dynamic Watermarking |
| **C** | `File-C_Technical-Whitepaper-Guide.md` | Deep Due Diligence | NDA Gating · stahování zakázáno · Dynamic Watermarking |

---

## Krok 1 — Připravit PDF/PPTX z Markdown souborů

### Soubor A (Executive Teaser)
- Převést `File-A_Executive-Teaser.md` do PDF (max 2 strany, A4 nebo Letter)
- Font: sans-serif (Inter, DM Sans nebo Helvetica)
- Maximální velikost: **10 MB** (pro rychlé načítání na mobilu)
- Doporučuje se černobílé nebo tmavé téma s výraznými akcenty

### Soubor B (Strategic Deck)
- Převést `File-B_Strategic-Deck.md` do PPTX nebo PDF ve **formátu na šířku (landscape, 16:9)**
- Každá sekce `## SLIDE X` = jeden slide
- Klíčové vizuály: Figure 1 a Figure 2 z ETFT.pdf vložit jako obrázky (screenshoty nebo SVG export z LaTeXu)
- Maximální velikost: **10 MB**
- Vložit odkaz na `ETFT__AI_Research_Scientist.mp4` (host video na Loom nebo Vimeo) do Slidu 8 nebo 10

### Soubor C (Technical Package)
- `File-C_Technical-Whitepaper-Guide.md` slouží jako navigační průvodce
- Přiložit originální PDFs jako separátní soubory nebo jako přílohy v DocSendu (Room feature)
- Přidat WIPO PROOF certifikát jako první stranu Souboru C před distribucí

---

## Krok 2 — DocSend nastavení (step-by-step)

### Soubor A
1. Nahrát PDF do DocSendu
2. Nastavení odkazu: **Require Email** (visitors must enter email before viewing)
3. **Allow Download: YES** — chcete, aby si vás uložili
4. Vytvořit jeden generický odkaz pro cold outreach

### Soubor B
1. Nahrát landscape PDF/PPTX do DocSendu
2. Nastavení odkazu:
   - **Allow Download: NO**
   - **Dynamic Watermarking: ON** (zobrazí email návštěvníka přes stránky — psychologická bariéra)
   - **Require Email: YES**
3. **Vytvořit unikátní odkaz pro každého GP/investora:** např. `etft-pande-a16z`, `etft-khosla`, `etft-sequoia` — pokud odkaz otevře třetí strana, DocSend vás upozorní a odhalí přeposílání

### Soubor C
1. Nahrát package do DocSendu (použít **DocSend Room** pro více souborů)
2. Nastavení:
   - **NDA Gating: ON** — návštěvník musí odsouhlasit mlčenlivost před přístupem
   - **Allow Download: NO**
   - **Dynamic Watermarking: ON**
3. Sdílet pouze po explicitní žádosti, nikdy v cold emailu

---

## Krok 3 — Sekvence distribuce

```
Email 1 (cold outreach)
  └── Příloha: [Soubor A odkaz] — Executive Teaser
  └── CTA: "Sdílím 1-stránkový přehled ETFT. Rád bych znal váš pohled."

Email 2 (pokud otevřeli Soubor A nebo odpověděli)
  └── Příloha: [Soubor B odkaz — unikátní pro daného GP]
  └── CTA: "Přikládám strategický deck. Hodí se 20 minut na call příští týden?"

Email 3 (pokud investor proklikl celý Soubor B nebo se vrátil do 24h)
  └── Signál: okamžitě zavolat nebo napsat
  └── Text: "Všiml jsem si, že vás zaujal [konkrétní slide] — mám k tomu čerstvá data..."
  └── Příloha: [Soubor C odkaz — po podpisu NDA]
```

---

## Krok 4 — Analytika a follow-up pravidla

| Signál v DocSendu | Akce |
|---|---|
| Investor strávil >60 s na Figure 1 (Regression Pipeline, Slide 4) | Příští email: "Všiml jsem si zájmu o de-optimalizační pipeline — máme čerstvá benchmark data" |
| Investor strávil >60 s na Figure 2 (Open-Loop Ecosystem, Slide 6) | Příští email: zaměřit na autonomní RL feedback loop a škálovatelnost |
| Soubor B otevřen podruhé do 24 hodin | **Okamžitě zavolat** |
| Soubor B přeposlán (otevřen jiným emailem) | Kontaktovat nového čtenáře zvlášť |
| Investor požádá o Soubor C | Poslat NDA + unikátní DocSend Room odkaz |

---

## Krok 5 — IP ochrana a právní základ

### Vrstvení informací (Algoritmický Black Box)
- **Soubor A + B:** Popisují *co* systém dělá a jaké jsou výstupy. **Nesdílí** detaily CI/CD Validation Pipeline, D_Perf trénovacího schématu ani orchestrační logiky ARL.
- **Soubor C:** Sdílí technické detaily až po NDA gating.

### Metadatová stopa
DocSend ukládá timestamp každého otevření, IP adresu a email návštěvníka. Tato data mohou sloužit jako podpůrný důkaz o sdílení IP v konkrétním čase.

### WIPO PROOF
Před prvním odesláním Souboru C vygenerovat digitální otisk každého dokumentu přes [wipoproof.wipo.int](https://wipoproof.wipo.int). Certifikát uložit spolu s originálem.

### Cílení — technicky relevantní investoři (příklady)
- **Anjney Midha (a16z)** — deep tech, AI infrastructure
- **Vinod Khosla (KV)** — autonomous science, biology AI
- **Sarah Guo (Conviction)** — AI-native companies
- **Jeff Dean / DeepMind partnerships** — research infra
- **DARPA IPTO / ARPA-H** — government R&D acceleration

---

## Soubory v tomto adresáři

| Soubor | Popis |
|---|---|
| `File-A_Executive-Teaser.md` | Soubor A — Executive Teaser (1–2 strany) |
| `File-B_Strategic-Deck.md` | Soubor B — Strategic Presentation Deck (10 slidů) |
| `File-C_Technical-Whitepaper-Guide.md` | Soubor C — Průvodce technickým balíčkem + NDA notice |
| `README.md` | Tento soubor — DocSend strategie a distribuční playbook |

**Zdrojové dokumenty:** viz `../papers/`

---

*Lukas Benda · lukas.benda@boldpivot.cz · April 2026*
