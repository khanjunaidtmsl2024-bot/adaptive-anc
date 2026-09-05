# PS 26052 - DEFENCE PACKAGE
## SIH Demo Script, Judge Prep, and Evidence Board

---

## THREE-MINUTE DEMO

0:00-0:20: Problem statement + show physical system
0:20-0:45: Hardware setup (two mics, headset, processor)
0:45-1:15: Live A/B comparison (bypass then enable)
1:15-1:45: Hybrid path + controller state transitions
1:45-2:15: Evidence (SNR/STOI/PESQ, ablation, latency)
2:15-2:40: Deployment evidence + cost breakdown
2:40-3:00: What is proven, what is not, next step

---

## JUDGE QUESTIONS

Q1: Why not consumer ANC?
A: Consumer ANC is physical. We enhance speech digitally.
Commercial headsets (ComTac) have static profiles. We adapt automatically.

Q2: What does AI add beyond classical filtering?
A: Classical handles linear/stationary. AI handles nonlinear/non-stationary.
Ablation table proves the difference.

Q3: What are d[n], x[n], y[n], e[n]?
A: d=primary mic, x=reference mic, y=noise estimate, e=enhanced output.
All observable at runtime. No clean speech needed.

Q4: Why NLMS before AI?
A: Post-AI needs clean speech as desired signal (unavailable).
Pre-AI uses both mics directly (mathematically valid).

Q5: What happens during impulse?
A: Detector triggers, controller enters IMPULSE mode, freeze NLMS.
Recovery time measured.

Q6: Measured latency?
A: Answer with YOUR number, not literature. State measurement method.

Q7: Train/test leakage prevention?
A: Speaker IDs separated. NOISEX-92 held out. Leakage checker script.

Q8: Worst failure case?
A: Have one ready. Example: speech leakage into reference mic.

Q9: What did not work?
A: Reference Evidence Log negative results.

Q10: Novelty?
A: Defence-oriented integration, not invention of AI enhancement.

---

## EVIDENCE BOARD

1. Architecture diagram
2. Physical prototype photo
3. Live A/B audio
4. Results table by noise category
5. Ablation table
6. Latency table
7. Unseen-noise results
8. Cost breakdown
9. Deployment roadmap

---

## KNOWN LIMITATIONS

1. Not field-qualified or ruggedized
2. Impulse tested at lab-safe levels only
3. NLMS benefit is hypothesis until ablation
4. Sample rate pending DeepFilterNet2 verification
5. Deployment target not finalized

---

## RECOVERY PLAN

1. Audio fails: pre-recorded A/B
2. AI crashes: offline inference results
3. Mic feedback: quieter position, lower gain
4. Latency high: laptop demo
5. Judge asks: Evidence Log honest answer

---

*Every claim traces to 04_EVIDENCE_LOG.md*
