# RAG Evaluation — Question & Answer Pairs

20 test questions for the AutoLease chatbot, covering good outputs, partial
outputs, and cases expected to produce a "I don't know" response.

Each entry notes the expected behaviour and why.

---

## Good outputs — answer is clearly in the documents

**Q01**
> What is the minimum and maximum duration of a leasing contract?

Expected: Minimum 12 months, maximum 84 months.
Source: `general_policy.md`

---

**Q02**
> How much is the early termination fee if I have 8 installments remaining?

Expected: 3% of the outstanding principal balance (fewer than 12 remaining).
Source: `early_termination.md`

---

**Q03**
> What documents does a company need to submit when applying for leasing?

Expected: Court register excerpt, 2 years of financial statements, tax clearance certificate, completed application form.
Source: `eligibility_and_application.md`

---

**Q04**
> What is the interest rate for leasing an electric vehicle?

Expected: 2.9% per annum (preferential rate).
Source: `interest_rates.md`

---

**Q05**
> When are monthly installments due and what happens if I pay late?

Expected: Due on the 5th of each month; late fee is 0.05% of the installment per day of delay.
Source: `payment_terms.md`

---

**Q06**
> Can I drive a leased vehicle to Serbia?

Expected: Yes — the Western Balkans (including Serbia) are within the permitted geographic area.
Source: `vehicle_use_and_insurance.md`

---

**Q07**
> What are my options when my leasing contract ends?

Expected: Three options — purchase at residual value, return the vehicle, or extend for 12 months.
Source: `end_of_contract.md`

---

**Q08**
> How long does AutoLease keep my personal data?

Expected: 7 years, in accordance with Croatian law.
Source: `general_policy.md`

---

**Q09**
> Can a student apply for leasing?

Expected: Yes, with a co-applicant (guarantor) who independently meets the standard eligibility criteria.
Source: `eligibility_and_application.md`

---

**Q10**
> How many days does AutoLease have to process a complaint?

Expected: 30 business days.
Source: `general_policy.md`

---

## Partial / ambiguous outputs — answer exists but may be incomplete or require follow-up

**Q11**
> What is the interest rate for leasing?

Expected: Partial — the model should mention the range (3.2%–6.8% for financial leasing,
4.5%/5.1% for operational), but without knowing the customer type or vehicle it cannot
give a single figure. A good response acknowledges it depends on creditworthiness, contract
duration, and vehicle type.
Source: `interest_rates.md`

---

**Q12**
> What happens if I don't pay my installment?

Expected: Should cover the late fee and escalation stages (15/30/60/90 days), but may miss
the deferral option. Watch for incomplete escalation paths.
Source: `payment_terms.md`

---

**Q13**
> Can I modify the leased vehicle?

Expected: Modifications are strictly prohibited without prior written approval. The model
should mention this clearly, but may or may not list all examples (tinting, wraps, engine mods).
Source: `vehicle_use_and_insurance.md`

---

**Q14**
> How can I apply for leasing?

Expected: Online (24h) or in branch (3 business days). May or may not mention what documents
are needed depending on which chunks are retrieved.
Source: `eligibility_and_application.md`

---

## Expected "I don't know" — question is outside the documents

**Q15**
> What car models are available for leasing?

Expected: "I don't know" — no document lists available vehicle models.
The chatbot should not guess or hallucinate a vehicle catalogue.

---

**Q16**
> What is the current EURIBOR rate?

Expected: "I don't know" — EURIBOR is mentioned as a reference in `interest_rates.md`
but its current value is not in any document.

---

**Q17**
> Can I lease a motorcycle?

Expected: "I don't know" — documents only reference vehicles; motorcycles are never mentioned.

---

**Q18**
> What is AutoLease's company registration number?

Expected: "I don't know" — company registration details are not in any document.

---

**Q19**
> How do I reset my password on the customer portal?

Expected: "I don't know" — the customer portal is mentioned once (insurance partner list)
but there is no information about account management or password reset.

---

**Q20**
> What is the salary structure for AutoLease sales employees?

Expected: "I don't know" — salary details are classified as a business secret per
`employee_conduct_and_confidentiality.md` and no salary information appears in any document.
Even if the internal HR docs are ingested, this specific detail is not present.

---

## Notes for running the evaluation

- Run the chatbot with default settings first, then compare outputs against expected answers above.
- Q15–Q20 are the clearest test of hallucination resistance — the chatbot must say it doesn't know rather than invent an answer.
- Q11–Q14 are useful for testing chunking strategy differences: try `--strategy paragraph` vs `--strategy recursive` and see if the retrieved context is more or less complete.
- For memory testing: ask Q02 first, then follow up with *"What if I have 15 remaining?"* — the chatbot should rewrite this into a standalone question and return the 5% fee.
