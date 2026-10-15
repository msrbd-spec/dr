## Verdict
Direction mostly thik (ordinal loss + domain generalization + label-noise handling), kintu oi reply-te **6-ta bhul/overclaim** ache. **Oi "Do you accept 87-89% is the true SOTA?" proshner jonno "yes" bolo na, ar ekhon master doc-o chao na.** Age tomar S18 diagnostics-er output lagbe, ota ekhono ashe nai.

## Claim-by-claim

| Oi reply ja bolse | Ami ja dekhi |
|---|---|
| BO "guaranteed to find the global optimum" for SVM | **Bhul.** SVM-er dual convex, kintu `(C, γ)` choose kora non-convex black-box. Optuna kono global optimum guarantee dey na. Valid choice, guarantee na |
| Ordinal triplet loss + RBF SVM | Valid direction. Kintu noisy label-e triplet/hard-negative mining noise amplify kore, tai Confident Learning-er sathe-i jabe. SVM head ta softmax/CORAL-er cheye valo kina evidence lagbe. Novelty literature search-er por claim |
| MixStyle "mix APTOS batch with Messidor-2 batch" | **Description vul.** Original MixStyle batch-er vitore instance-der statistics mix kore, target data lage na. Messidor mix korle seta adaptation (FDA-r moto), MixStyle na. Ar "FDA abandon" over-reaction: ami bolechi order x beta ablation chalao, discard na |
| Drop KD, use "Cross-Scale Attention Fusion" | **Tomar MSDA/HFF ei family-r.** Etai 10 run-e kaj kore ni. Novelty hishebe abar ana jabe na |
| Confident Learning (cleanlab) | **Sabcheye valo ideaţa.** Kintu eta label issue *estimate* kore (OOF probability theke), "mathematically identify" na. 15% figure ekhono unverified |
| "Mathematical proof" 92% impossible | **Proof na.** Inter-rater ~85% ami nijei verify kori ni. Individual-rater agreement consensus label-er upor model-er accuracy-r hard bound na (model consensus-er sathe beshi mile jete pare). Observation: 10 model ekhoi 85%-e. Eta strong evidence, theorem na |
| "Clean test set → 92% achievable" | **Dangerous.** Test image model ba Confident Learning diye remove korle circular, selection-on-outcome. Valid shudhu jodi blind expert re-adjudication hoy, ar original test set-er sathe-i report |
| "Novelty bulletproof, top-tier" | **Overclaim.** 3-ta known technique-er combination, result unknown. Scientific Reports/JBHI/CBM-er jonno realistic, TMI/MedIA-te multi-dataset lage |
| "87-89% is true SOTA, QWK ~0.94" | **Source nai, accept koro na.** Tomar data: QWK 0.92 ↔ acc 85%. Literature numbers different split-e |
| "Binary 96-98% easily" | Plausible kintu "easily" unverified. Tomar prediction theke ekhon-i measure koro |

## Ja ekhon korbe
1. **S18 diagnostics chalao, output paste koro.** Eta chara sob guesswork. Messidor-2 majority class, prior shift, referable AUC, label mapping. 85% external target-er answer ekhane-i.
2. **Targets bar bar bodlay (97 → 94 → 92).** Run-er age **ekta** pass/fail table lekho: 5-class QWK, adjacent acc, referable sens/spec/AUC, external on the same. Raw 5-class 92% ke target korbe na.
3. **Cheap experiments age, big plan pore** (protita ekta run, ta-o age hypothesis lekhe):
   - Confident Learning: protita fold model-e nijer val fold-er prediction (OOF) save korte hobe. Eta ekhon save hoy na, agent task. Tarpor cleanlab chalao, flagged image gulo dekho.
   - MixStyle (in-batch) vs FDA (order x beta) vs none, external-e.
   - Ordinal triplet vs CORAL, same backbone.
4. Result-er por-i master doc lekha worth, ager na.

Accept koro: 97% ar 92% raw 5-class legitimate na. Accept koro na: "87-89% true SOTA" ar "bulletproof novelty", jotokkhon na tomar data tai bole.

`run_helper.md` unchanged.

---

**Ha, ekta critical ache. Baki-gulo important ba minor.**

## Critical (scientific integrity)
**"Test set clean kore 92% pao"**: eta tomar **test-set-er upor selection on outcome**. Model ba Confident Learning diye "ambiguous" image remove korle je image model-er sathe mile na segulo-i bad jay, tai accuracy artificially bare. Reviewer eta dhorle number-ta invalid hoye jay, aro kharap hole integrity-r proshno ওঠে. Rule:
- Confident Learning **training/val fold-er upor** chalano thik (OOF probability diye label-issue flag ba re-weight kora standard).
- **Test set kokhono touch korba na.** Jodi clean test set chao, seta hote hobe **blind expert re-adjudication** diye, ar paper-e original test result-o ekshathe report korte hobe.

## Important (wasted time ba reviewer catch korbe)
1. **"Cross-scale attention fusion" novelty hishebe:** eta tomar MSDA/HFF-er family, jar 10 run-e kono gain dekha jay ni. Eta abar novelty bole ana jabe na.
2. **"Mathematical proof" ar "87-89% true SOTA, QWK 0.94":** kono source ba proof nai. Tumi PI ba paper-e eta repeat korle ta unsupported claim hobe.
3. **"Novelty bulletproof / top-tier":** 3-ta known technique-er combination, result ekhono ashe nai.
4. **MixStyle-er description vul:** original MixStyle ek batch-er instance-er statistics mix kore, target data lage na. Messidor-2 mix korle seta adaptation (FDA-r moto), MixStyle na. Method section-e vul likhle reviewer dhorbe, ar "domain-agnostic" claim-o weak hoy.

## Minor (shudhu wording)
- Optuna "guaranteed global optimum" bhul. SVM dual convex, kintu `(C, γ)` choose non-convex black-box.
- Confident Learning label issue **estimate** kore, "mathematically identify" na. 15% ambiguity figure unverified.
- "Binary 96-98% easily" unverified, measure korle jani.

## Ekta practical blocker
Confident Learning chalate OOF probabilities lagbe (protita fold-er val prediction), kintu ekhon save hoy na. Eta ekta chhoto agent task, tomar kono ekta run-e shuru korar age lagbe.

`run_helper.md` unchanged.
