# SureCure

Counterfeit medicines pose a significant public health threat in the Philippines, with a United Nations Office on Drugs and Crime report declaring the country as a hotspot for pharmaceutical crimes (UNODC, 2018). Such products are available for sale on the market. Thus, it is necessary for consumers to have a quick means of screening their medicines. SureCure is an Android screening tool for 20 common Philippine over-the-counter medicines and vitamins. Users photograph the pack front and the panel showing the barcode, lot number, and expiry date. A fine-tuned ResNet-18 identifies the product, and an energy-score guard, with a threshold calibrated on held-out real photos, flags unsupported packs. On-device text recognition reads the codes for user confirmation, which are compared with Food and Drug Administration (FDA) Philippines records. The complete application will be evaluated on a scripted scenario suite built from the FDA tables, with reliability judged by whether each scenario returns its intended verdict. 

Keywords: counterfeit medicines, ResNet-18, energy-based out-of-distribution detection, mobile screening application

## Status

The app follows the pipeline in the paper (Figure 1):

| Stage | What | Status |
|---|---|---|
| 0 | Guided capture: pack front, then the code panel | Camera screen done; blur and glare check not yet |
| 1 | Product identification (ResNet-18 on device, ONNX Runtime) and manual selection | Done |
| 2 | Code reading: barcode, lot, registration number, expiry, manufacture date | Typed by the user, with the code-panel photo shown; on-device text recognition (ML Kit) not yet |
| 3 | Verification rules against the FDA tables: red, amber or green with every reason | Done |
| 4 | Guided visual check and reporting | Not yet; amber and red results offer nearby chain drugstores |

Green means "no match with FDA counterfeit records", never "safe" or "authentic". The photo model only identifies
the product; it is not a counterfeit detector.

## Repository layout

| Path | What |
|---|---|
| `App/` | Android app (Kotlin, Jetpack Compose), package `com.mlbb.surecure` |
| `App/src/main/assets/model/` | Photo model: `surecure.onnx`, `meta.json`, `labels.json` (one training run) |
| `App/src/main/assets/fda/` | FDA tables: registrations, counterfeit lots, barcodes |
| `App/src/main/assets/drugstores/` | Chain drugstore branches with exact locations |
| `ml/` | Training and evaluation notebook (`surecure_train.ipynb`) and its source |
| `rules/` | Python reference rules engine and its 21 test scenarios (the specification for the app) |
| `data/` | FDA and drugstore source data |
| `scripts/` | Rebuild the tables and app assets from `data/` |
| `docs/` | Model card, data sources, integration notes |
| `med-images/` | Team photos of in-scope medicines |

## Tests

```bash
./gradlew :app:testDebugUnitTest
./gradlew :app:connectedDebugAndroidTest
pytest -q rules/tests
```

- Unit tests (JVM): image preprocessing against PIL, the 21 rule scenarios, 1,027 cases checked against the Python
  engine, barcodes, the scan flow, and nearby drugstores.
- Device test: the model's golden images and tensors give the same results on the phone as in training.

Details: `docs/integration_notes.md`, `docs/rules_integration_notes.md`, `docs/drugstores_notes.md`, `docs/model_card.md`.
