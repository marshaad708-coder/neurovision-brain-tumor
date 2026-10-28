# NeuroVision: Brain Tumor MRI Classification with Transfer Learning and Grad-CAM

**Muhammad Arshaad — October 2026**
*Project 4 of my portfolio for the China masters application. It follows
VisionAI (Project 2, industrial defect detection) and ForgeAD (Project 3,
self-supervised anomaly detection), and moves the same methods into
medical imaging.*

## 1. The problem I wanted to solve

My first two vision projects looked at factory products: is this steel
sheet defective, is this toothbrush anomalous. The methods worked, and I
wanted to know whether they transfer to a problem where a mistake costs
more than a scrapped part. Brain tumor MRI classification is that
problem. A radiologist's bottleneck is time, and an early, correct read
of a scan changes what happens to a patient. So this project asks a
narrow, testable question: how well can a small transfer-learned
classifier, trained on a CPU, separate glioma, meningioma, pituitary
tumor, and healthy scans in the public Brain Tumor MRI dataset — and can
it show its work with Grad-CAM?

An important line I keep throughout: this is a research and education
tool, a second pair of eyes at best. It is not a medical diagnosis, and
nothing here is validated for clinical use.

## 2. Dataset

Masoud Nickparvar's *Brain Tumor MRI Dataset* on Kaggle, which combines
the figshare, SARTAJ and Br35H collections. I downloaded it directly from
Kaggle on 2026-10-28 and counted the files on disk myself
(`data/DATASET_STATUS.md`):

| Split | glioma | meningioma | notumor | pituitary | total |
|---|---|---|---|---|---|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing  | 400 | 400 | 400 | 400 | 1,600 |

The current Kaggle version (version 2) is balanced and has **7,200**
images. That is different from the figures quoted in most papers
(7,023 images; 5,712/1,311, unbalanced), which describe version 1. I am
reporting the counts I verified, because the model below was trained on
exactly these files.

Integrity checks I ran (md5): 7,013 of the 7,200 files are unique; 187
files are exact duplicates of another file *inside the same split*, and
— the important part — there is **zero exact-hash overlap between
Training and Testing**, so no test image is a byte-copy of a training
image. The dataset ships no patient IDs, so I cannot check the stronger
claim that no *patient* appears in both splits; the split is image-level
only, and I treat that as a limitation (Section 6).

Protocol: I keep the dataset's own Training/Testing split. A stratified
15% of Training (840 images) is held out as validation for model
selection; the remaining 4,760 images train the model. Testing (1,600
images) is read only by the evaluation script, once training is done.

## 3. Method

**Model.** ResNet-18 (He et al., CVPR 2016) pretrained on ImageNet, with
the 1,000-class head replaced by dropout (0.3) + a 4-class linear head.
Input 224×224, normalized with the ImageNet mean/std; MRI scans are
single-channel in spirit but stored as JPEGs of varying size and colour
mode, so everything is converted to RGB.

**CPU shortcut, stated plainly.** Following the same compromise as
ForgeAD, the stem, layer1 and layer2 stay frozen at their ImageNet
weights; layer3, layer4 and the head are trained (10.5M trainable
parameters of 11.7M total). Low-level edge/texture filters from ImageNet
transfer well, and freezing cuts an epoch to about 7 minutes on a
2-core CPU (399–445 s per epoch after the first, measured). A GPU run would fine-tune all layers and would probably do
somewhat better; I did not want to claim numbers I did not measure.

**Training.** 12 epochs, batch 32, Adam (lr 1e-3, weight decay 1e-4,
cosine schedule), cross-entropy with label smoothing 0.05.
Augmentation: random crop after a small upscale (a light zoom), horizontal
flip, rotation up to 12°, mild brightness/contrast jitter. The splits are
class-balanced, so no class weighting was needed. Checkpoints are saved
after every epoch, the best-validation checkpoint is kept separately,
and `--resume` continues an interrupted run — a lesson ForgeAD taught
me the hard way. Frozen blocks are kept in eval mode during training so
their BatchNorm statistics do not drift.

**Evaluation.** On the 1,600 Testing images: accuracy, macro-F1,
one-vs-rest AUROC (macro), and per-class recall/sensitivity, precision
and F1, plus the full confusion matrix. In a medical setting recall is
the number I look at first — a missed tumor is worse than a false alarm.

**Explanation.** Grad-CAM (Selvaraju et al., ICCV 2017) on layer4, the
last convolutional layer. The web app shows the overlay for every
prediction, and the evaluation script saves one example per class.
Grad-CAM is a post-hoc explanation of the model, not a tumor
segmentation.

## 4. Results

Training ran 12 epochs (87 minutes of logged epoch time on a 2-core
CPU; the run was interrupted twice by machine restarts and continued
with `--resume` both times, losing no completed epoch — the per-epoch
checkpointing did its job). Best validation accuracy was **0.9821**
(epoch 9). The best checkpoint, evaluated once on the held-out Testing
split:

| Metric | Value |
|---|---|
| Accuracy | **0.9563** (1,530 / 1,600) |
| Macro-F1 | **0.9557** |
| AUROC (macro, one-vs-rest) | **0.9882** |

| Class | Recall (sensitivity) | Precision | F1 | AUROC |
|---|---|---|---|
| Glioma | 0.8675 | 0.9830 | 0.9216 | 0.9637 |
| Meningioma | 0.9800 | 0.9356 | 0.9573 | 0.9933 |
| No tumor | 1.0000 | 0.9259 | 0.9615 | 0.9970 |
| Pituitary | 0.9775 | 0.9874 | 0.9824 | 0.9988 |

Confusion matrix (rows = true, columns = predicted):

|  | glioma | meningioma | notumor | pituitary |
|---|---|---|---|---|
| **glioma** | 347 | 22 | 29 | 2 |
| **meningioma** | 2 | 392 | 3 | 3 |
| **notumor** | 0 | 0 | 400 | 0 |
| **pituitary** | 4 | 5 | 0 | 391 |

## 5. Discussion

Three things stand out, and I would rather name them than round the
headline number up.

**Glioma is the hard class, and it fails in the worst direction.**
Recall for glioma is 0.8675 while every other class is at 0.9775 or
above. Of the 53 missed gliomas, 22 are called meningioma (a tumour is
still flagged — the wrong one, but a clinician would still look) and
**29 are called "no tumor"**. In a screening tool that is the most
dangerous error there is: a real tumour passed as healthy. The glioma
AUROC (0.9637) shows the model usually *ranks* gliomas above the other
classes; the failures are argmax decisions on scans that presumably
look atypical or faint. A screening deployment would not use argmax —
it would flag anything with glioma probability above a low threshold
and accept more false alarms. I did not tune such a threshold here, and
the numbers above are the plain argmax result.

**"No tumor" recall is perfect, its precision is not.** All 400 healthy
scans are recognised (recall 1.0000), but 32 tumour scans are predicted
as healthy, which drags no-tumor precision to 0.9259. Same fact as the
glioma problem, seen from the other side.

**Validation was optimistic.** Best validation accuracy was 0.9821
against 0.9563 on Testing. The validation slice comes from the same
Training pool the model fit, while Testing is the dataset's own
separate split; a gap in this direction is expected, and it is one more
reason the Testing numbers are the ones I put in the README.

For context only: papers on version 1 of this dataset commonly report
97–99% accuracy with fully fine-tuned models, often ResNet-50 or larger,
on GPU. My 0.9563 with a frozen-front ResNet-18 on CPU sits below that,
which matches the cost I expected from the freezing shortcut — and the
version difference (Section 2) means the numbers are not directly
comparable anyway.

## 6. Limitations — the honest list

* **Image-level split, no patient IDs.** The dataset does not say which
  scans come from the same patient, so near-duplicate scans of one
  patient could in principle sit in both splits (exact byte duplicates
  are excluded — I checked — but resized/recompressed copies would pass
  that check). Published numbers on this dataset, mine included, should
  be read with that in mind.
* **Within-split duplicates and augmented files.** 187 files are exact
  in-split duplicates. File names also show that the meningioma class
  was padded with augmented copies: 100 of 1,400 Training and 103 of
  400 Testing meningioma files are named `*-aug-me_*`. Augmented test
  images are easier than fresh scans (they derive from real ones), so
  the meningioma test scores are probably slightly optimistic.
* **Frozen early layers** (Section 3) cap what the model can adapt to
  MRI-specific low-level structure.
* **One public dataset, curated scans.** Real clinical scans vary by
  scanner, protocol and population. Nothing here measures that shift,
  and the model has never seen a scan outside this dataset's sources.
* **Softmax scores are not calibrated probabilities.** The app labels
  them as model confidence, and that is all they are.
* **Grad-CAM is coarse.** At layer4 the map is 7×7 upsampled to the full
  scan; it shows roughly where the model looked, not a tumor outline.

## 7. What I would do next

* Fine-tune all layers on a GPU and measure how much the frozen-layer
  shortcut actually cost.
* Add tumor segmentation (the figshare source ships masks for 3,064
  scans) so the app can outline a region instead of a coarse heatmap.
* Evaluate on an external set (e.g. a different public MRI collection)
  to measure dataset shift instead of assuming it away.
* Calibrate the output scores (temperature scaling on the validation
  split) before calling any number a probability.

## 8. References

1. M. Nickparvar, "Brain Tumor MRI Dataset", Kaggle, 2021.
2. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
   Recognition", *Proc. CVPR*, 2016.
3. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
   Networks via Gradient-based Localization", *Proc. ICCV*, 2017.
4. J. Cheng et al., "Enhanced Performance of Brain Tumor Classification
   via Tumor Region Augmentation and Partition", *PLOS ONE*, 2015.
5. A. Pashaei, H. Sajedi, B. Jasemi, "Brain Tumor Classification via
   Convolutional Neural Network and Extreme Learning Machines", 2018
   (SARTAJ source dataset).
