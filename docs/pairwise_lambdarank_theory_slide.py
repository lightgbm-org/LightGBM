"""Render the one-page Pairwise LambdaRank theory slide."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pairwise_lambdarank_theory_slide"

BG = "#F5F7F4"
INK = "#17212B"
MUTED = "#56616B"
BORDER = "#D6DCD7"
GREEN = "#087F5B"
GREEN_TINT = "#E7F4EE"
ORANGE = "#B54708"
ORANGE_TINT = "#FFF0E6"
BLUE = "#2563A6"
BLUE_TINT = "#EAF2FA"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, edge=BORDER, face=WHITE, radius=0.012, lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, value, size=14, color=INK, weight="normal", **kwargs):
    return ax.text(
        x, y, value, transform=ax.transAxes, fontsize=size, color=color,
        fontfamily="DejaVu Sans", fontweight=weight, va="top", ha="left",
        **kwargs,
    )


fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Header
txt(ax, 0.052, 0.948, "PAIRWISE LAMBDARANK", 11, GREEN, "bold")
txt(ax, 0.052, 0.910, "Learn preferences. Spend effort where ranking quality moves.", 28, INK, "bold")
txt(
    ax, 0.052, 0.858,
    "Our implementation trains on ordered document pairs, weights every correction by its top-K NDCG impact, "
    "and maintains a latent document ranking.",
    13.5, MUTED,
)

# Stage cards
y, h = 0.245, 0.555
x1, w1 = 0.052, 0.275
x2, w2 = 0.365, 0.342
x3, w3 = 0.745, 0.203
box(ax, x1, y, w1, h)
box(ax, x2, y, w2, h)
box(ax, x3, y, w3, h)

# Flow arrows
for xa, xb in [(x1 + w1 + 0.008, x2 - 0.008), (x2 + w2 + 0.008, x3 - 0.008)]:
    ax.add_patch(FancyArrowPatch(
        (xa, y + h / 2), (xb, y + h / 2), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color="#89928B",
    ))

# Stage 1: pair evidence
txt(ax, x1 + 0.018, y + h - 0.025, "1  PAIR EVIDENCE", 11.5, GREEN, "bold")
txt(ax, x1 + 0.018, y + h - 0.066, "Score an ordered comparison", 16, INK, "bold")
txt(ax, x1 + 0.018, y + h - 0.112, "One tree row represents (i, j):", 11.5, MUTED)
box(ax, x1 + 0.018, y + h - 0.205, w1 - 0.036, 0.062, edge=GREEN_TINT, face=GREEN_TINT, radius=0.006, lw=0)
txt(ax, x1 + 0.030, y + h - 0.159, r"$[x_i\;|\;x_j\;|\;\mathrm{optional}\;(x_i-x_j)]\;\to\;f(i,j)$", 13.2, INK)
txt(ax, x1 + 0.018, y + h - 0.237, "Direct margin", 10.5, GREEN, "bold")
txt(ax, x1 + 0.018, y + h - 0.270, r"$d_{ij}=\mathrm{mean}\{f(i,j),-f(j,i)\}$", 14.5, INK)
txt(ax, x1 + 0.018, y + h - 0.321, "Optional indirect evidence through document k", 10.5, GREEN, "bold")
txt(ax, x1 + 0.018, y + h - 0.354, r"$a_{ij}:\;f(i,k)+f(k,j)\quad\mathrm{and\ signed\ variants}$", 12.2, INK)
box(ax, x1 + 0.018, y + 0.025, w1 - 0.036, 0.135, edge="#B7DCCD", face="#F2FAF6", radius=0.006)
txt(ax, x1 + 0.030, y + 0.137, "Blended preference margin", 10.5, GREEN, "bold")
txt(ax, x1 + 0.030, y + 0.101, r"$m_{ij}=\dfrac{w_d d_{ij}+w_a a_{ij}}{w_d+w_a}$", 13.5, INK)
txt(ax, x1 + 0.030, y + 0.039, r"$w_d=1-\alpha,\;w_a=\alpha$ when that evidence exists", 9.0, MUTED)

# Stage 2: NDCG-aware LambdaRank
txt(ax, x2 + 0.018, y + h - 0.025, "2  NDCG-AWARE LAMBDA", 11.5, ORANGE, "bold")
txt(ax, x2 + 0.018, y + h - 0.066, "Price each mistake by its ranking impact", 16, INK, "bold")
txt(ax, x2 + 0.018, y + h - 0.112, "For unequal labels, with at least one item currently in top K:", 11.2, MUTED)
box(ax, x2 + 0.018, y + h - 0.295, w2 - 0.036, 0.150, edge=ORANGE_TINT, face=ORANGE_TINT, radius=0.006, lw=0)
txt(ax, x2 + 0.030, y + h - 0.166, "Swap value", 10.5, ORANGE, "bold")
txt(
    ax, x2 + 0.030, y + h - 0.207,
    r"$\Delta\mathrm{NDCG}_{ij}=\dfrac{|G(y_i)-G(y_j)|\;|D(r_i)-D(r_j)|}{\mathrm{IDCG@K}}$",
    14.2, INK,
)
txt(ax, x2 + 0.030, y + h - 0.273, r"$G(y)=2^y-1$ by default  |  $D(r)=1/\log_2(r+2)$", 9.8, MUTED)
txt(ax, x2 + 0.018, y + h - 0.324, "Logistic pressure", 10.5, ORANGE, "bold")
txt(ax, x2 + 0.018, y + h - 0.360, r"$q_{ij}=\dfrac{1}{1+e^{\sigma m_{ij}}}$", 14.2, INK)
txt(ax, x2 + 0.153, y + h - 0.355, "mass on the wrong order", 9.7, MUTED)
box(ax, x2 + 0.018, y + 0.014, w2 - 0.036, 0.122, edge="#F0C8A8", face="#FFF8F2", radius=0.006)
txt(ax, x2 + 0.030, y + 0.113, "LambdaRank pseudo-gradient and curvature", 10.5, ORANGE, "bold")
txt(ax, x2 + 0.030, y + 0.073, r"$\lambda_{ij}=-\sigma q_{ij}\,\Delta\mathrm{NDCG}_{ij}$", 13.3, INK)
txt(ax, x2 + 0.030, y + 0.035, r"$h_{ij}=\sigma^2q_{ij}(1-q_{ij})\,\Delta\mathrm{NDCG}_{ij}$", 13.3, INK)

# Stage 3: pointwise state
txt(ax, x3 + 0.016, y + h - 0.025, "3  RANKING STATE", 11.5, BLUE, "bold")
txt(ax, x3 + 0.016, y + h - 0.066, "Reconstruct the list", 16, INK, "bold")
txt(ax, x3 + 0.016, y + h - 0.116, "Maintain latent scores s_i that determine current ranks.", 11.2, MUTED, wrap=True)
box(ax, x3 + 0.016, y + h - 0.270, w3 - 0.032, 0.112, edge=BLUE_TINT, face=BLUE_TINT, radius=0.006, lw=0)
txt(ax, x3 + 0.028, y + h - 0.180, "Match preference probabilities", 10.2, BLUE, "bold")
txt(ax, x3 + 0.028, y + h - 0.219, r"$P_{pair}=\mathrm{sigmoid}(\sigma m_{ij})$", 12.8, INK)
txt(ax, x3 + 0.028, y + h - 0.251, r"$P_{rank}=\mathrm{sigmoid}(\sigma(s_i-s_j))$", 12.8, INK)
txt(ax, x3 + 0.016, y + h - 0.315, "A damped Newton-style step aligns them; every score change is clipped to +/-0.3.", 10.6, MUTED, wrap=True)

# Compact feedback loop
for cy, label, color in [
    (y + 0.145, "PAIR SCORES", GREEN),
    (y + 0.093, "LATENT s", BLUE),
    (y + 0.041, "RANKS + NDCG", ORANGE),
]:
    box(ax, x3 + 0.026, cy, w3 - 0.052, 0.035, edge=color, face=WHITE, radius=0.012, lw=1.1)
    txt(ax, x3 + 0.052, cy + 0.027, label, 9.2, color, "bold")
ax.add_patch(FancyArrowPatch(
    (x3 + w3 - 0.030, y + 0.162), (x3 + w3 - 0.030, y + 0.059),
    transform=ax.transAxes, connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
    mutation_scale=10, linewidth=1.2, color="#7B8791",
))
ax.add_patch(FancyArrowPatch(
    (x3 + 0.023, y + 0.057), (x3 + 0.023, y + 0.158),
    transform=ax.transAxes, connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
    mutation_scale=10, linewidth=1.2, color="#7B8791",
))

# Implementation safeguards strip
txt(
    ax, 0.052, 0.219,
    "IN CODE  |  top-K truncation  |  optional LambdaRank normalization  |  item-weight product  |  log discount / hard preference / L2 controls",
    9.5, MUTED, "bold",
)

# Takeaway band
band = FancyBboxPatch(
    (0.052, 0.060), 0.896, 0.118,
    boxstyle="round,pad=0.008,rounding_size=0.010",
    linewidth=0, facecolor=INK, transform=ax.transAxes,
)
ax.add_patch(band)
txt(ax, 0.071, 0.152, "MANAGER TAKEAWAY", 10.5, "#69D3AC", "bold")
txt(
    ax, 0.071, 0.116,
    "Pairwise learning tells us who should win; NDCG weighting tells us which wins matter most.",
    17.0, WHITE, "bold",
)
txt(
    ax, 0.052, 0.031,
    "Implementation anchors: src/objective/rank_objective.hpp + .cpp; src/io/metadata.cpp; src/io/dataset.cpp",
    8.5, "#707A73",
)

for ext in ("png", "pdf", "svg"):
    kwargs = {"bbox_inches": None, "pad_inches": 0, "facecolor": fig.get_facecolor()}
    if ext == "png":
        kwargs["dpi"] = 180
    fig.savefig(OUT.with_suffix(f".{ext}"), **kwargs)

plt.close(fig)


def write_pptx(path, image_path):
    """Create a standards-based, single-slide PPTX with the render full bleed."""
    parts = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "docProps/core.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Pairwise LambdaRank: Mathematical Theory</dc:title>
  <dc:creator>LightGBM team</dc:creator>
  <cp:lastModifiedBy>LightGBM team</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-07-13T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-07-13T00:00:00Z</dcterms:modified>
</cp:coreProperties>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office PowerPoint</Application><PresentationFormat>Widescreen</PresentationFormat>
  <Slides>1</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips><ScaleCrop>false</ScaleCrop>
  <AppVersion>16.0000</AppVersion>
</Properties>""",
        "ppt/presentation.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""",
        "ppt/_rels/presentation.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>""",
        "ppt/slides/slide1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    <p:pic><p:nvPicPr><p:cNvPr id="2" name="Pairwise LambdaRank theory slide" descr="One-page mathematical overview"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>""",
        "ppt/slides/_rels/slide1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""",
        "ppt/slideLayouts/slideLayout1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""",
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""",
        "ppt/slideMasters/slideMaster1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Theme"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>""",
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""",
        "ppt/theme/theme1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="LightGBM Slide">
  <a:themeElements><a:clrScheme name="LightGBM"><a:dk1><a:srgbClr val="17212B"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="56616B"/></a:dk2><a:lt2><a:srgbClr val="F5F7F4"/></a:lt2><a:accent1><a:srgbClr val="087F5B"/></a:accent1><a:accent2><a:srgbClr val="B54708"/></a:accent2><a:accent3><a:srgbClr val="2563A6"/></a:accent3><a:accent4><a:srgbClr val="69D3AC"/></a:accent4><a:accent5><a:srgbClr val="D6DCD7"/></a:accent5><a:accent6><a:srgbClr val="89928B"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="DejaVu Sans"><a:majorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="LightGBM"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"/></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"/></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>""",
    }
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
        archive.write(image_path, "ppt/media/image1.png")


write_pptx(OUT.with_suffix(".pptx"), OUT.with_suffix(".png"))
