# Font-space augmentation for synthetic Tibetan OCR

Literature and tooling review, 22 July 2026.

## Narrow overview: font variation rather than image distortion

The literature specifically concerned with variation **at the font or glyph-outline
level** is much smaller than the general synthetic-OCR and image-augmentation
literature. It is useful to distinguish four approaches that are often grouped
together under “font augmentation”:

1. **Sampling existing fonts or designed styles.** This is by far the most common
   approach. It increases typeface diversity but does not create a new style from
   a given font.
2. **Changing typographic renderer parameters.** Weight, kerning, spacing,
   underline, stroke/border, and sometimes shear are varied while rendering.
   Whether this changes the actual outline, strokes the outline at render time, or
   transforms a raster mask varies between generators.
3. **Transforming vector outlines before rasterization.** Width, slant, stroke
   width, proportions, or control points are changed while the glyph is still a
   vector. This is the closest category to the implementation in this repository.
4. **Interpolating or generating fonts.** Variable-font axes and learned vector-font
   generation create coherent new glyph designs. This is relevant future work, but
   the typography literature does not by itself show an OCR benefit.

### Directly relevant work

#### MJSynth / Synth90k — Jaderberg et al. (2014)

MJSynth established the large-scale synthetic-training recipe for scene-text
recognition. It samples from more than 1,400 fonts and randomly varies kerning,
weight, underline, and other rendering properties before adding scene and camera
effects. Its ablation shows a clear gain from moving from one font to a large font
catalogue.

This is strong evidence that **font diversity matters**, but weak evidence for any
particular outline transform. The published ablation does not isolate width,
slant, or stroke modification, and most of the final gain also depends on
background, projective, blending, and degradation stages.

#### A Synthetic Recipe for OCR — Etter et al. (2020)

This work performs controlled experiments over synthetic OCR factors including
font inventory and font size. It supports the methodology of changing one
generation factor at a time and measuring recognition on real data. It does not,
however, provide a detailed study of generating new outline variants from static
fonts.

#### SynthTIGER — Yim et al. (2021)

SynthTIGER is a mature configurable text-image generator. It includes font
sampling and typographic/style controls such as size, thickness, spacing, skew,
stretch, and character-level transformations. It is a useful implementation and
ablation reference.

Its scope is broader than font files: several operations are applied within the
rendering/compositing pipeline rather than producing a reusable, validated
OpenType font. Consequently, it does not address preservation of Tibetan
GSUB/GPOS behavior or outline topology after writing a modified font.

#### OmniPrint — Sun, Tu, and Guyon (2021)

OmniPrint is the clearest published precedent for the present approach. It
explicitly uses FreeType to perform **vector-based, pre-rasterization**
transformations. Supported style factors include:

- linear transforms such as shear;
- stroke-width variation;
- character-proportion changes, such as ascender and descender length;
- random movement of outline anchor points.

The stated motivation is that these operations are difficult to perform faithfully
on pixel images. OmniPrint covers 935 fonts and 27 scripts and records nuisance
parameters as metadata.

There are still important limitations for the present use case:

- its primary benchmark is isolated-character classification and meta-learning,
  not line OCR;
- its experiments do not isolate the contribution of each outline operation to
  recognition on real documents;
- it does not report Tibetan shaping, stacked glyphs, GSUB/GPOS preservation, or
  the counter/fragmentation checks needed here.

Thus OmniPrint validates the **technical category**—pre-rasterization vector
variation—but not the safe parameter ranges or expected OCR gain for Tibetan.

#### Variable fonts and vector-font generation

OpenType variable fonts provide designer-authored continuous axes, commonly
weight (`wght`), width (`wdth`), slant (`slnt`), and optical size (`opsz`).
Sampling a genuine variable-font axis is preferable to mechanically deforming a
static font because interpolation follows masters designed to preserve glyph
quality and family identity.

Vector-font research such as DeepVecFont (2021) demonstrates learned interpolation
between font styles. NIV (2026, emerging work) predicts vector-point displacement
for weight, width, slant, and optical-size axes and emits variable font files,
including tests on complex CJK glyphs.

These works show that coherent font-space interpolation is technically possible.
They are **not OCR augmentation studies**, and neither establishes that generated
intermediate styles improve recognition. Tibetan coverage and shaping behavior
would also need separate validation. For this project, authentic variable Tibetan
fonts can be sampled when available; learned axis generation should remain a
later experiment rather than the baseline.

### What the focused literature supports

- Use a broad, curated set of real fonts first. This is the most established
  source of typographic diversity.
- Prefer vector/pre-rasterization transformations to morphology on a final
  low-resolution image when a new glyph shape is intended.
- Treat width, slant, weight, spacing, and stroke width as distinct nuisance
  variables and retain their parameters as provenance.
- Evaluate each axis separately on untouched real OCR data before combining axes.
- Use script-aware validity checks. Evidence from Latin isolated characters
  cannot establish safety for Tibetan stacks and marks.

### What it does not yet support

There appears to be no controlled published result showing that mechanically
mutating static OpenType outlines improves Tibetan line OCR. More generally, the
review found little OCR evidence isolating direct outline mutation from:

- simply adding more real fonts;
- selecting existing bold, condensed, or italic faces;
- renderer-level stroking;
- post-rasterization erosion, dilation, stretch, or shear;
- document-level degradation.

The strongest defensible claim is therefore:

> The implementation follows a published synthetic-data technique—controlled
> pre-rasterization font variation—but its Tibetan-specific validity and utility
> must be established by quality gates, human review, and downstream ablation.

## Position of the current method

The present pipeline separates four stages:

1. **Unicode and corpus selection:** choose authentic Tibetan text and
   coverage-compatible stacks.
2. **OpenType outline variation:** sample width, negative slant, and directional
   stroke parameters.
3. **Shaping and quality gates:** check HarfBuzz GSUB/GPOS behavior, mark
   placement, raster topology, and human-reviewed boundary samples.
4. **Document realism:** only after shaping, add ink, paper, scanning, and page
   deformation effects.

This ordering matters. A plausible isolated contour is insufficient if an
operation damages substitutions, positioning, metrics, counters, or stacked-mark
geometry.

## Broader literature map

### Foundational synthetic OCR

- **MJSynth / Jaderberg et al. (2014–2016):** random fonts, kerning, weight,
  underline, curved baselines, projective distortion, blending, and noise.
  Foundational evidence for large-scale font/style randomization.
- **SynthText / Gupta et al. (2016):** places text in natural scenes using scene
  geometry and appearance matching. It supports synthetic labels and domain
  randomization, but is not evidence about glyph-outline fidelity.
- **A Synthetic Recipe for OCR / Etter et al. (2020):** controlled experiments
  over font count, size, colors, backgrounds, padding, and rotation. Particularly
  relevant to the proposed component-wise ablation.
- **SynthTIGER / Yim et al. (2021):** configurable synthesis with font, thickness,
  spacing, elastic, stretch, skew, rotation, perspective, texture, and corpus
  controls.
- **STRAug / Atienza (2021):** evaluates 36 image augmentations across six scene
  text recognizers. Its important general lesson is that augmentation effects are
  model-dependent; it is complementary to, not evidence for, font-space changes.

### Document and manuscript realism

- **DocCreator / Journet et al. (2017):** historical-document synthesis including
  bleed-through, ink decay, blur, holes, and paper deformation.
- **Augraphy:** an actively usable pipeline for ink, paper, scanner, lighting,
  folding, and fax-like effects.
- **ocrodeg:** lightweight blur, warp, ink spread, noise, and page-curl tools.

These methods belong after text shaping and rasterization. They solve a different
problem from font variation and should be evaluated as a separate augmentation
stage.

### Complex-script analogues

Work on Arabic handwriting synthesis uses connection-aware construction, affine
scaling, shear, rotation, and slant distributions. It offers an important analogue
for Tibetan: broad geometric randomization is less important than preserving the
script's joins, collisions, marks, and other invariants.

Studies of shorthand/stenography augmentation also provide a useful negative
result: erosion and dilation can reduce recognition accuracy. A conventional
augmentation is not automatically useful merely because its output looks
plausible.

Recent Tibetan handwriting work uses GANs or diffusion-based style generation.
These methods may eventually extend style coverage beyond digital fonts, but need
training data and content-validity filtering. They should follow, rather than
replace, a measured procedural baseline.

An NVIDIA multilingual OCR engineering report (2026; not peer reviewed) describes
large multilingual font pools, random-field stroke variation, and erosion/dilation
gated by minimum text height. It is a close operational precedent for rejecting
destructive stroke operations, but is not controlled scientific evidence.

## Assessment of the implemented axes

### Width

Width scaling is a common typographic variation and a standard variable-font
axis. The manually reviewed range of 0.80–1.20 is a reasonable engineering prior.
It should still be compared with genuine condensed/expanded Tibetan faces and
with width statistics from target manuscripts.

Width scaling is not equivalent to vertical expansion in a fixed rendering
context. Horizontal scaling changes glyph width and advance; vertical scaling
changes height, baseline-relative geometry, and the apparent ratio of horizontal
to vertical strokes. Normalizing both outputs to the same final line height can
make them look related, but their metrics and raster sampling differ.

### Slant

Shear/slant is common in synthetic generators and is a standard variable-font
axis. For Tibetan, the direction and distribution must be script-specific.
Manual review indicates that negative slant is plausible while positive slant is
unusual. A distribution concentrated near upright is better justified than
uniform sampling over the full range.

### Weight and directional stroke changes

Uniform weight approximates changing strokes in all directions simultaneously,
but it is not exactly the sum of independent horizontal and vertical stroke
operations. Outline offsetting affects corners, joins, counters, terminals, and
overlaps nonlinearly; the order and boolean reconstruction also matter.

Directional stroke changes can add stroke-contrast diversity unavailable from a
simple affine transform. They are also the highest-risk operation:

- thinning can erase small strokes or detach components;
- thickening can fill counters and merge stacks;
- boolean reconstruction can reverse winding or fill holes;
- different fonts reach failure thresholds at different parameter values.

The current policy of raster-gating each generated font is therefore stronger
than using a universal range. Failed variants should be rejected, not silently
clamped.

### Tracking and side bearings

**Tracking** changes spacing across a run of text. **Side bearings** are the
font-level spaces to the left and right of each glyph. Tracking can model loose or
tight composition without changing glyph outlines. Side-bearing variation can
model type or metal-spacing irregularity but is riskier for a complex script
because it may create collisions or unnatural gaps around stacks and punctuation.
Neither is currently a priority over page-layout spacing and line-level
calibration.

## Initial augmentation policy

These values are engineering priors from manual review, not measurements of the
target manuscript population.

- **Width:** sample uniformly from 0.80 to 1.20 after quality control.
- **Negative slant mixture:**
  - 45% near upright: 0° to −3°;
  - 30% mild: −3° to −7°;
  - 20% marked: −7° to −11°;
  - 5% strong: −11° to −14°.
- **Directional thinning:** begin near −0.010 em horizontally and −0.015 em
  vertically; accept only fonts passing raster and placement checks.
- **Directional thickening:** explore up to +0.030 em with counter and
  stack-collision checks.
- **Combined extremes:** sample conditionally. The strongest joint combinations
  should be rare—roughly 1–2% of pages—rather than arising from independent broad
  uniform distributions.

Replace these priors with empirical distributions after measuring width, slant,
stroke contrast, and spacing on representative BDRC lines.

## Tool analysis

### Core tools

- **HarfBuzz:** essential for Tibetan OpenType shaping and GSUB/GPOS validation.
- **fontTools:** appropriate Python foundation for reading and writing OpenType,
  preserving tables, and applying deterministic affine transformations.
- **skia-pathops:** suitable for outline boolean operations and winding
  normalization. It still requires topology and raster checks.
- **FontForge:** convenient for width and slant. Weight/stem operations have been
  slow or destructive on some Tibetan fonts, so its role should remain selective.
- **FreeType outline stroking / `FT_Outline_EmboldenXY`:** worth evaluating as a
  runtime alternative for x/y emboldening. FreeType warns that acute angles and
  intersections can produce artifacts; negative strengths need the same gates as
  generated fonts.

### Reference generators

- **OmniPrint:** closest reference for pre-rasterization vector transforms.
  Designed mainly for isolated characters rather than complex-script lines.
- **SynthTIGER:** strongest configurable word/line generator and ablation
  reference, but offers less control over writing validated modified OpenType
  fonts.
- **Pangoline / Kraken:** useful complex-script synthetic-training references.
  Kraken explicitly cautions that digital fonts alone are usually insufficient
  for handwriting.
- **TRDG:** useful for quick baselines, but does not provide the Tibetan shaping
  and generated-font validation required here.

### Complementary page tools

- **Augraphy:** recommended for configurable ink, paper, lighting, fold, and
  scanner effects.
- **ocrodeg:** recommended lightweight option for warp, blur, ink spread, noise,
  and page curl.
- **DocCreator:** rich historical-document degradation models, with a heavier
  C++/Qt integration.

## Recommended validation stack

1. **Font validity:** reopen each output; preserve cmap, GSUB, GPOS, metrics, and
   TTC face identity.
2. **Shaping:** compare source-supported probe stacks with HarfBuzz using
   `script=tibt` and `language=bo`.
3. **Geometry:** reject new mark, vowel, subscript, clipping, or collision
   warnings.
4. **Raster topology:** compare ink density, enclosed counters, connected
   components, and fragmentation with the source font.
5. **Human review:** stratify contact sheets by Uchen/Ume, font category,
   operation, and parameter boundary.
6. **Downstream utility:** retain only augmentation families that improve OCR on
   untouched real manuscripts.

Visual realism is necessary but not sufficient. A plausible variant may reduce
accuracy by over-representing artificial regularities or weakening distinctions
that the recognizer needs.

## Minimum ablation

Train otherwise identical models under these cumulative conditions:

- **A — Fonts only:** curated font catalogue and clean renderer.
- **B — Add width and negative slant:** width 0.80–1.20 and the weighted slant
  mixture above.
- **C — Add directional strokes:** conservative thinning, gated thickening, and
  raster QC.
- **D — Add ink and paper effects:** independently calibrated Augraphy/ocrodeg
  degradations.
- **E — Real-calibrated policy:** replace engineering priors with distributions
  measured on target manuscripts.

Evaluate every run on untouched real manuscript images. Report CER and WER
separately for Uchen and Ume, and stratify by stack complexity, image quality,
and seen/unseen typeface where possible. Synthetic-only validation is a debugging
metric, not the main result.

## References and resources

- Jaderberg, M. et al. (2014). [Synthetic Data and Artificial Neural Networks for
  Natural Scene Text Recognition](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/jaderberg14synthetic.pdf).
- Gupta, A., Vedaldi, A., and Zisserman, A. (2016).
  [Synthetic Data for Text Localisation in Natural Images](https://arxiv.org/abs/1604.06646).
- Journet, N. et al. (2017).
  [DocCreator: A New Software for Creating Synthetic Ground-Truthed Document
  Images](https://doi.org/10.3390/jimaging3040062).
- Etter, D. et al. (2020).
  [A Synthetic Recipe for OCR](https://hltcoe.jhu.edu/wp-content/uploads/2020/03/Etter__Rawls__Carpenter__Sell_-_A_Synthetic_Recipe_for_OCR.pdf).
- Yim, M. et al. (2021).
  [SynthTIGER: Synthetic Text Image GEneratoR](https://arxiv.org/abs/2107.09313).
- Atienza, R. (2021).
  [Data Augmentation for Scene Text Recognition](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Atienza_Data_Augmentation_for_Scene_Text_Recognition_ICCVW_2021_paper.pdf).
- Sun, H., Tu, W.-W., and Guyon, I. (2021).
  [OmniPrint: A Configurable Printed Character
  Synthesizer](https://openreview.net/forum?id=R07XwJPmgpl);
  [source code](https://github.com/SunHaozhe/OmniPrint).
- Wang, Y. et al. (2021).
  [DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality
  Learning](https://arxiv.org/abs/2110.06688).
- [NIV: Neural Axis Variations for Variable Font Generation](https://arxiv.org/abs/2606.05261)
  (2026; emerging font-generation research, not OCR evidence).
- [FreeType outline processing and emboldening](https://freetype.org/freetype2/docs/reference/ft2-outline_processing.html).
- [HarfBuzz OpenType shaping models](https://harfbuzz.github.io/opentype-shaping-models.html).
- [fontTools `TransformPen`](https://fonttools.readthedocs.io/en/latest/pens/transformPen.html).
- [skia-pathops](https://github.com/fonttools/skia-pathops).
- [Augraphy](https://github.com/sparkfish/augraphy).
- [ocrodeg](https://github.com/NVlabs/ocrodeg).
- [Kraken synthetic training guidance](https://kraken.re/6.0.0/training/synth.html).
- [NVIDIA multilingual OCR synthetic-data report](https://huggingface.co/blog/nvidia/nemotron-ocr-v2)
  (2026; engineering report, not peer reviewed).

## Scope

This is a targeted literature and tooling review, not a systematic review or
meta-analysis. Recent 2025–2026 material is emerging evidence and engineering
guidance; it is labelled separately from peer-reviewed OCR results.
