WEBTHINK_SIMPLE6 = """Question: Are group 2 innate lymphoid cells increased in chronic rhinosinusitis with nasal polyps or eosinophilia?
Thought 1: I should review the background statement about CRS and ILC2 counts.
Action 1: Search[group 2 innate lymphoid cells chronic rhinosinusitis]
Observation 1: The abstract explains that 35 CRS patients were profiled and that ILC2 frequencies were associated with nasal polyps, tissue eosinophilia, and eosinophil-dominant CRS.
Thought 2: I need confirmation that the investigators concluded ILC2s were elevated in the nasal polyp/eosinophilic groups.
Action 2: Lookup[ILC2s were increased]
Observation 2: “ILC2 frequencies were associated with the presence of nasal polyps (P = 0.002) as well as high tissue eosinophilia (P = 0.004).”
Thought 3: Since the study explicitly reports higher ILC2 counts in CRSwNP/eosinophilia, the answer is yes.
Action 3: Finish[yes]

Question: Does admission hyperglycemia predict failed fibrinolysis in ST-elevation myocardial infarction?
Thought 1: I should locate details about the retrospective STEMI cohort.
Action 1: Search[admission hyperglycemia fibrinolysis failure STEMI]
Observation 1: Among 304 STEMI patients, those with failed reperfusion had higher admission glucose (126 vs 110 mg/dL) and hyperglycemia independently predicted fibrinolysis failure (HR 4.79).
Thought 2: Confirm that hyperglycemia remained an independent predictor.
Action 2: Lookup[independent predictor]
Observation 2: “Admission hyperglycemia was an independent predictive factor for failed reperfusion…”
Thought 3: Therefore admission hyperglycemia is associated with failed fibrinolysis.
Action 3: Finish[yes]

Question: Do systematic reviews in emergency medicine journals routinely search clinical trial registries?
Thought 1: I need the results of the cross-sectional review.
Action 1: Search[systematic reviews emergency medicine clinical trial registries]
Observation 1: Of 41 intervention-focused reviews, only 8 (20%) searched a trial registry, and half of those found relevant unpublished studies.
Thought 2: Because only a small fraction searched registries, the answer is no.
Action 2: Finish[no]

Question: Do obese patients with idiopathic pulmonary fibrosis have higher 90-day mortality with bilateral lung transplantation?
Thought 1: Look at OPTN registry results.
Action 1: Search[idiopathic pulmonary fibrosis obesity bilateral lung transplantation 90-day]
Observation 1: Among 3,389 IPF recipients, an interaction showed BMI >30 plus bilateral transplant increased 90-day mortality (OR 1.71).
Thought 2: Confirm significance of the effect.
Action 2: Lookup[1.71 times more likely]
Observation 2: “Patients with BMI >30 kg/m2 who received BLT are 1.71 times more likely to die within 90 days…”
Thought 3: That indicates obese BLT recipients have higher early mortality.
Action 3: Finish[yes]

Question: Did restaging CT after neoadjuvant chemoradiation often change management in locally advanced rectal cancer?
Thought 1: Review the retrospective cohort.
Action 1: Search[restaging CT neoadjuvant chemoradiation rectal cancer management]
Observation 1: Eighty-three of 91 patients had restaging CT; only four showed suspicious lesions and none of the scans changed surgical plans.
Thought 2: Since no management changes occurred, the answer should be no.
Action 2: Finish[no]

Question: Are newspaper stories about do-not-attempt-resuscitation decisions mainly concerned with lack of patient involvement?
Thought 1: Find the newspaper analysis.
Action 1: Search[DNACPR newspaper coverage patient involvement]
Observation 1: Reviewing 331 UK articles showed the predominant theme was perceived lack of patient (and later surrogate) involvement in DNACPR orders.
Thought 2: Therefore the reports emphasize missing consultation.
Action 2: Finish[yes]"""

REFLECTIONS = """
Previous Trial:
Question: Do nitrogen oxide levels correlate with disease activity in Egyptian rheumatoid arthritis patients?
Thought 1: I noted rheumatoid arthritis patients had higher nitric oxide than controls but did not analyze subgroups.
Action 1: Finish[yes]

Reflection: I should explicitly cite that nitric oxide levels were significantly higher in active disease (mean 82 µmol/L vs 36 µmol/L) and correlated with DAS-28, ESR, and CRP, showing the relationship rather than just stating “yes.”

Previous Trial:
Question: Is ΔNp63 expression protective against progression in high grade T1 bladder cancer?
Thought 1: I focused on overall prognosis and ignored the exact progression statistic.
Action 1: Finish[maybe]

Reflection: The study reported 0% progression when ΔNp63 was preserved versus 21% when it was lost. I should restate those numbers and conclude definitively that ΔNp63 expression marks patients at lower risk.
"""

COTQA_SIMPLE6 = """
Question: Does methylation of the FGFR2 gene associate with high birth-weight centile?
Thought: The study found FGFR2 CpG methylation remained linked to high birth weight centiles after adjusting for genetics, so yes.
Action: Finish[yes]

Question: Does the Boston keratoprosthesis provide wide depth of focus?
Thought: The through-focus curve was flatter like a 2–3 mm pupil, giving pseudoaccommodation, so yes.
Action: Finish[yes]

Question: Are hidradenitis suppurativa patients burdened with substantial comorbidities?
Thought: Chart review showed higher rates of smoking, arthropathies, dyslipidemia, etc., so yes.
Action: Finish[yes]

Question: Does nitric oxide rise with rheumatoid arthritis activity?
Thought: Active patients had higher nitric oxide and it correlated with DAS-28, so yes.
Action: Finish[yes]

Question: Do restaging CT scans frequently alter management after neoadjuvant chemoradiation in rectal cancer?
Thought: Only 5% showed suspicious lesions and none changed surgery, so no.
Action: Finish[no]

Question: Are clinical high grade T1 bladder tumors with ΔNp63 expression less likely to progress?
Thought: Tumors retaining ΔNp63 had zero progression vs 21% otherwise, so yes.
Action: Finish[yes]"""

COT_SIMPLE_REFLECTION = """
Question: Does promoter variant rs2301228 in NCAM1 confer schizophrenia risk?
Thought: I answered maybe because the mechanism wasn’t clear.
Action: Finish[maybe]

Reflection: The case-control study replicated the rs2301228 association, showed risk allele reduced NCAM1 transcription, and serum levels were lower in patients. I should answer yes based on that evidence.

Question: Are Syk phosphorylation and TRAF6 overexpression linked in systemic lupus erythematosus B cells?
Thought: I incorrectly said no because I mixed up signaling pathways.
Action: Finish[no]

Reflection: Flow cytometry showed both Syk phosphorylation and TRAF6 expression were elevated and correlated, and inhibition of Syk reduced TRAF6. I must connect those results before concluding.
"""

COT = """Relevant PubMed Context: The chronic rhinosinusitis study profiled 35 surgical patients and controls. Group 2 innate lymphoid cell (ILC2) frequencies were higher in eosinophilic CRS, correlated with nasal polyps (P = 0.002), tissue eosinophilia (P = 0.004), blood eosinophilia (P = 0.005), and asthma, and tracked with worse nasal symptom scores.
Question: Are ILC2s increased in chronic rhinosinusitis with nasal polyps/eosinophilia?
Thought: The trial explicitly links higher ILC2 levels to nasal polyps and eosinophilic disease, so yes.
Action: Finish[yes]

Relevant PubMed Context: In a retrospective cohort of 304 STEMI patients treated with fibrinolysis, admission glucose was higher in failed reperfusion (126 vs 110 mg/dL, p<0.001). Hyperglycemia independently predicted failed fibrinolysis (HR 4.79), alongside longer delay to treatment and anterior MI.
Question: Is admission hyperglycemia associated with failed reperfusion after fibrinolysis?
Thought: Because hyperglycemia remained an independent predictor, the answer is yes.
Action: Finish[yes]

Relevant PubMed Context: A review of 41 emergency medicine intervention systematic reviews (2013) found only 8 searched clinical trial registries, though half of those searches identified unpublished studies.
Question: Do emergency medicine systematic reviews routinely search trial registries?
Thought: Since only 20% included registry searches, the answer is no.
Action: Finish[no]
"""

COT_REFLECT = """
Relevant PubMed Context: Eighty-three of ninety-one rectal cancer patients underwent routine restaging CT after neoadjuvant chemoradiation. Only four scans (5%) showed new suspicious distant lesions and none led to a change in surgical management.
Question: Are restaging CTs after chemoradiation high yield for altering rectal cancer surgery plans?
Thought: I mistakenly claimed yes by assuming any detection is useful.
Action: Finish[yes]

Reflection: The study explicitly states that no management changes occurred despite the scans. I should answer no and mention the 5% detection rate with zero impact on surgery.

Context: Among IPF lung transplant recipients, obese patients (BMI > 30 kg/m2) receiving bilateral grafts had 1.71× higher 90-day mortality than normal-weight BLT recipients.
Question: Do obese IPF patients face greater 90-day mortality after bilateral lung transplant?
Thought: I answered maybe without citing the odds ratio.
Action: Finish[maybe]

Reflection: I must quote the OR 1.71 (95% CI 1.03–2.85) showing significantly higher early mortality, so future answers should clearly state yes with that evidence.
"""

COT_REFLECT2 = """Relevant PubMed Context: In the systematic review audit, only 8 of 41 emergency medicine reviews searched a trial registry, so most did not.
Question: Do emergency medicine systematic reviews routinely include registry searches?
Thought: I said yes because I misread “identified unpublished trials” as common.
Action: Finish[yes]

Reflection: The numbers show registry searches were rare. I need to cite the 20% figure and conclude no.

Relevant PubMed Context: The nitric oxide study found active RA patients had mean NO ~82 µmol/L versus ~36 µmol/L in inactive disease and ~25 µmol/L in controls, correlating with DAS-28 and Sharp scores.
Question: Are higher nitric oxide levels linked to RA activity and joint damage?
Thought: I answered no because I focused on the lack of infection correlation.
Action: Finish[no]

Reflection: Since nitric oxide tracked with disease activity and radiographs, I should answer yes and mention the correlation coefficients (e.g., r=0.85 with DAS-28).

Relevant PubMed Context: Treatment of Helicobacter pylori-positive asthenozoospermic men lowered seminal IgA and improved progressive sperm motility.
Question: Does eradication therapy improve sperm motility in infertile men with seminal H. pylori?
Thought: I said maybe because I doubted the sample size.
Action: Finish[maybe]

Reflection: The study reported significant post-treatment motility gains, so future answers should acknowledge the positive effect even if the cohort was modest."""
