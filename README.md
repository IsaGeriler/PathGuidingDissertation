# PathGuidingDissertation

## 1. Path Guiding w/BSDF Inversion (Cornell Box, 16SPP Learning/112SPP Guiding)

| Path Trace w/NEE (128 SPP) | Path Guiding (200 Nearest Vertex) | Path Guiding (400 Nearest Vertex) |
|:---:|:---:|:---:|
| <img width="1024" height="1024" alt="128SPPWhatThePathIntegratorDoing" src="https://github.com/user-attachments/assets/0c776a2a-3fed-4ffc-ae11-66b88f7faece" /> | <img width="1024" height="1024" alt="128SPPWhatThePathGuidingDoing200PathVertex" src="https://github.com/user-attachments/assets/01f9fad1-3e54-4e02-85ce-586c3bbb1628" /> | <img width="1024" height="1024" alt="128SPPWhatThePathGuidingDoing400PathVertex" src="https://github.com/user-attachments/assets/b2af24a8-e4ae-48c9-8ec1-ca68bb653edb" /> |

<img width="1920" height="1200" alt="Screenshot 2026-08-03 020652" src="https://github.com/user-attachments/assets/0f182a5a-f767-4290-ae69-696ae3e1d282" />

## 2. Past Updates Over The Four Weeks

### Stats on PointBVHBuild and Render Time

<img width="1919" height="1125" alt="Screenshot 2026-07-22 154819" src="https://github.com/user-attachments/assets/7434e527-bb00-495d-bbf9-364b6bb2d89a" />

### Incoming Radiance Rendered (i.e. Li Stored in PathVertex, resembling Path Trace w/out NEE)

| 1 SPP | 128 SPP |
|:---:|:---:|
| <img width="1280" height="720" alt="1SPPGuided" src="https://github.com/user-attachments/assets/5c65dfee-c9b7-4b4e-baa9-804e38ef0e6d" /> | <img width="1280" height="720" alt="KitchenGuidedPathAlphaIncomingRadiance128SPP" src="https://github.com/user-attachments/assets/b537a8ea-0057-4f4c-b691-2824292f0fbe" /> |
| <img width="720" height="1280" alt="StaircaseGuidedPathAlphaIncomingRadiance1SPP" src="https://github.com/user-attachments/assets/1c73d2ad-4c25-4afe-b40e-4f63111e5263" /> | <img width="720" height="1280" alt="StaircaseGuidedPathAlphaIncomingRadiance128SPP" src="https://github.com/user-attachments/assets/14c51112-c8fe-47ce-89e1-12c0d87f6726" /> |

### Pixel Colour Rendered (i.e. Path Integrator)

| 1 SPP | 128 SPP |
|:---:|:---:|
| <img width="1280" height="720" alt="1SPPPixel" src="https://github.com/user-attachments/assets/b97248b9-f274-47d8-bfe2-04bf2389e814" /> | <img width="1280" height="720" alt="KitchenGuidedPathAlphaPixel128SPP" src="https://github.com/user-attachments/assets/109a0f67-140d-47aa-b84d-3048b6f4850a" /> |
| <img width="720" height="1280" alt="StaircaseGuidedPathAlphaPixel1SPP" src="https://github.com/user-attachments/assets/7d82ec4c-e265-4767-896f-c1a0e2469c0f" /> | <img width="720" height="1280" alt="StaircaseGuidedPathAlphaPixel128SPP" src="https://github.com/user-attachments/assets/c6f0e076-c8ee-4ebf-98bf-39e16b8a4e64" /> |

### DielectricBSDF Material Fix (before invert() DielectricBSDF)
| Bedroom 512 SPP Buggy DielectricBSDF | Bedroom 512 SPP Fixed DielectricBSDF |
|:---:|:---:|
| <img width="1280" height="720" alt="Bedroom512SPPPathIntegratorBuggyDielectricBSDF" src="https://github.com/user-attachments/assets/b68827b8-74b6-4719-95c8-8db11da6eeba" /> | <img width="1280" height="720" alt="Bedroom512SPPPathIntegratorDielectricBSDF" src="https://github.com/user-attachments/assets/a910db92-307c-4cbc-b089-90c88ccb6a68" /> |
