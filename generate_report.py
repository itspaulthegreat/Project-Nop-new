"""
generate_report.py
──────────────────────
Builds a thesis-style PDF report (reports/project_report.pdf) for the project
"Neural Network Weight Optimization as a Constrained NLP" using ReportLab
(Platypus + canvas). No LaTeX is involved.

The report pulls its actual numbers from results/summary.csv and the
per-experiment results/*.json files, and embeds the figures already rendered
in figures/*.png. The 1-hidden-layer network sketch is drawn programmatically
with reportlab.graphics shapes (it is not an image file).

Run:  python generate_report.py
"""

import os
import csv
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, NextPageTemplate, HRFlowable,
                                KeepTogether)
from reportlab.graphics.shapes import Drawing, Circle, Line, String, Rect, Polygon

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(ROOT, 'figures')
REPORTS_DIR = os.path.join(ROOT, 'reports')
OUT_PDF = os.path.join(REPORTS_DIR, 'project_report.pdf')

MARGIN = 2 * cm
GUTTER = 18          # space between the two text columns
FIG_WIDTH = 14 * cm  # max embedded-figure width


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading helpers (read the ACTUAL results, never hard-code numbers)
# ─────────────────────────────────────────────────────────────────────────────

def load_json(name):
    """Load one results/<name>.json as a dict (or {} if missing)."""
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_summary_rows():
    """
    Read results/summary.csv. The csv header was written before the
    `hessian_condition_number` column existed, so the newest rows carry one
    extra trailing value -- but every field used here (name, group, method,
    n_iter, solve_time, train_mse, test_mse) sits *before* that column, so
    DictReader maps them correctly. Duplicate experiment names (re-runs) are
    de-duplicated keeping the most recent row.
    """
    path = os.path.join(RESULTS_DIR, 'summary.csv')
    rows = {}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            if row.get('name'):
                rows[row['name']] = row          # last occurrence wins
    return list(rows.values())


def f5(x):
    try:
        return f'{float(x):.5f}'
    except (TypeError, ValueError):
        return '--'


def f3(x):
    try:
        return f'{float(x):.3f}'
    except (TypeError, ValueError):
        return '--'


def sci(x, nd=2):
    try:
        return f'{float(x):.{nd}e}'
    except (TypeError, ValueError):
        return '--'


# ─────────────────────────────────────────────────────────────────────────────
#  Styles
# ─────────────────────────────────────────────────────────────────────────────

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'ReportTitle', fontName='Helvetica-Bold', fontSize=19, leading=23,
        alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(
        'SubTitle', fontName='Helvetica', fontSize=11, leading=15,
        alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle(
        'AuthorLine', fontName='Helvetica-Bold', fontSize=12, leading=16,
        alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle(
        'AbstractHead', fontName='Helvetica-Bold', fontSize=11, leading=14,
        alignment=TA_CENTER, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(
        'Abstract', fontName='Helvetica', fontSize=9.5, leading=13,
        alignment=TA_JUSTIFY, leftIndent=10, rightIndent=10))
    styles.add(ParagraphStyle(
        'Heading', fontName='Helvetica-Bold', fontSize=12, leading=15,
        spaceBefore=6, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(
        'Body', fontName='Helvetica', fontSize=10, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=5))
    styles.add(ParagraphStyle(
        'Caption', fontName='Helvetica-BoldOblique', fontSize=9, leading=11,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(
        'Reference', fontName='Helvetica', fontSize=9, leading=12,
        alignment=TA_LEFT, leftIndent=16, firstLineIndent=-16, spaceAfter=4))
    return styles


# ─────────────────────────────────────────────────────────────────────────────
#  Self-made network sketch (programmatic reportlab.graphics, NOT an image)
# ─────────────────────────────────────────────────────────────────────────────

def make_network_diagram():
    """
    A small 1-hidden-layer network sketch: one input node, a hidden layer of
    tanh units, one output node, with the weight/bias groups labelled
    'W1, b1' and 'W2, b2'. Drawn with reportlab shapes so it satisfies the
    'at least one self-made sketch' course requirement.
    """
    W, H = 200, 150
    d = Drawing(W, H)
    d.hAlign = 'CENTER'

    x_in, x_hid, x_out = 24, 100, 176
    r = 8
    n_hidden = 5
    y_in = H / 2
    y_out = H / 2
    hidden_ys = [25 + i * (100 / (n_hidden - 1)) for i in range(n_hidden)]

    # connections input -> hidden, hidden -> output (drawn first, behind nodes)
    for hy in hidden_ys:
        d.add(Line(x_in + r, y_in, x_hid - r, hy,
                   strokeColor=colors.HexColor('#9bbad6'), strokeWidth=0.8))
        d.add(Line(x_hid + r, hy, x_out - r, y_out,
                   strokeColor=colors.HexColor('#e0b48a'), strokeWidth=0.8))

    # input node
    d.add(Circle(x_in, y_in, r, fillColor=colors.HexColor('#cfe2f3'),
                 strokeColor=colors.HexColor('#3d6fa5')))
    d.add(String(x_in, y_in - 3, 'x', fontName='Helvetica-Bold', fontSize=9,
                 textAnchor='middle', fillColor=colors.black))

    # hidden nodes
    for hy in hidden_ys:
        d.add(Circle(x_hid, hy, r, fillColor=colors.HexColor('#fde9d3'),
                     strokeColor=colors.HexColor('#c8862f')))

    # output node
    d.add(Circle(x_out, y_out, r, fillColor=colors.HexColor('#d9ead3'),
                 strokeColor=colors.HexColor('#4a7a3a')))
    d.add(String(x_out, y_out - 3, 'y', fontName='Helvetica-Bold', fontSize=9,
                 textAnchor='middle', fillColor=colors.black))

    # weight/bias group labels
    d.add(String((x_in + x_hid) / 2, H - 12, 'W1, b1', fontName='Helvetica-Bold',
                 fontSize=8, textAnchor='middle', fillColor=colors.HexColor('#3d6fa5')))
    d.add(String((x_hid + x_out) / 2, H - 12, 'W2, b2', fontName='Helvetica-Bold',
                 fontSize=8, textAnchor='middle', fillColor=colors.HexColor('#c8862f')))

    # layer captions
    d.add(String(x_in, 4, 'input', fontName='Helvetica', fontSize=7,
                 textAnchor='middle', fillColor=colors.grey))
    d.add(String(x_hid, 4, 'hidden (tanh)', fontName='Helvetica', fontSize=7,
                 textAnchor='middle', fillColor=colors.grey))
    d.add(String(x_out, 4, 'output', fontName='Helvetica', fontSize=7,
                 textAnchor='middle', fillColor=colors.grey))
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  Figure embedding
# ─────────────────────────────────────────────────────────────────────────────

def figure_block(filename, caption, styles):
    """Returns a list of flowables: a centered, width-fitted image plus a
    bold-italic caption, kept together so they never split across a page."""
    path = os.path.join(FIGURES_DIR, filename)
    flow = []
    if os.path.exists(path):
        iw, ih = ImageReader(path).getSize()
        w = FIG_WIDTH
        h = w * ih / iw
        img = Image(path, width=w, height=h)
        img.hAlign = 'CENTER'
        flow.append(img)
    else:
        flow.append(Paragraph(f'[missing figure: {filename}]', styles['Body']))
    flow.append(Paragraph(caption, styles['Caption']))
    return [KeepTogether(flow)]


# ─────────────────────────────────────────────────────────────────────────────
#  Page furniture (footer page numbers)
# ─────────────────────────────────────────────────────────────────────────────

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0] / 2.0, 1.0 * cm, f'Page {doc.page}')
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
#  Document assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_document():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    styles = build_styles()

    page_w, page_h = A4
    usable_w = page_w - 2 * MARGIN
    usable_h = page_h - 2 * MARGIN
    col_w = (usable_w - GUTTER) / 2.0

    one_frame = Frame(MARGIN, MARGIN, usable_w, usable_h, id='one')
    left_frame = Frame(MARGIN, MARGIN, col_w, usable_h, id='left')
    right_frame = Frame(MARGIN + col_w + GUTTER, MARGIN, col_w, usable_h, id='right')

    doc = BaseDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title='Neural Network Weight Optimization as a Constrained NLP',
        author='Arindam')
    doc.addPageTemplates([
        PageTemplate(id='OneCol', frames=[one_frame], onPage=footer),
        PageTemplate(id='TwoCol', frames=[left_frame, right_frame], onPage=footer),
    ])

    story = []
    story += title_page(styles)
    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak())

    story += introduction(styles)
    story += problem_formulation(styles)
    story += methods(styles)

    story.append(NextPageTemplate('OneCol'))
    story.append(PageBreak())
    story += results(styles)

    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak())
    story += discussion(styles)
    story += references(styles)

    doc.build(story)


# ── 1. Title page ────────────────────────────────────────────────────────────

def title_page(styles):
    s = []
    s.append(Spacer(1, 2.0 * cm))
    s.append(Paragraph('Neural Network Weight Optimization<br/>as a Constrained NLP',
                       styles['ReportTitle']))
    s.append(HRFlowable(width='60%', thickness=1, color=colors.grey,
                        spaceBefore=2, spaceAfter=14))
    s.append(Paragraph('Arindam', styles['AuthorLine']))
    s.append(Spacer(1, 0.3 * cm))
    s.append(Paragraph('Numerical Optimization', styles['SubTitle']))
    s.append(Paragraph('Albert-Ludwigs-Universit&auml;t Freiburg', styles['SubTitle']))
    s.append(Paragraph('Summer Term 2026 &middot; Prof. Dr. Moritz Diehl', styles['SubTitle']))
    s.append(Paragraph('July 2026', styles['SubTitle']))

    s.append(Paragraph('Abstract', styles['AbstractHead']))
    abstract = (
        "Neural-network training is almost universally treated as unconstrained "
        "first-order optimization. This report instead casts the training of a "
        "one-hidden-layer tanh network as a constrained nonlinear program (NLP) "
        "over the flattened weights, with the mean-squared error as objective and "
        "three structural constraints: a bilinear Lipschitz bound, a convex "
        "weight norm-ball, and linear symmetry-breaking inequalities. The NLP is "
        "solved exactly with the interior-point solver IPOPT and with sequential "
        "quadratic programming through CasADi, and compared against an "
        "unconstrained Adam baseline and a penalty method. We study how the "
        "Lipschitz bound trades training fit for generalization, read the "
        "constraint's shadow price from the KKT multipliers, and measure how "
        "solve cost scales with network size. The central finding is that hard "
        "constraints act as a controllable, exactly-enforced regularizer that an "
        "unconstrained penalty only approximates.")
    s.append(Paragraph(abstract, styles['Abstract']))
    return s


# ── 2. Introduction ──────────────────────────────────────────────────────────

def introduction(styles):
    s = []
    s.append(Paragraph('1. Introduction', styles['Heading']))
    s.append(Paragraph(
        "Training a neural network is, formally, an optimization problem: choose "
        "the weights that minimize a loss over the data. In practice that problem "
        "is almost always handed to a stochastic first-order method such as Adam, "
        "which is cheap, scalable, and indifferent to the geometry of the loss "
        "surface. From a numerical-optimization standpoint this is a missed "
        "opportunity. The same problem can be written as a nonlinear program "
        "(NLP) and handed to a second-order constrained solver, which lets us add "
        "<i>hard</i> structural requirements -- bounds the network must satisfy "
        "exactly, not merely on average -- and which exposes rich diagnostic "
        "information (Lagrange multipliers, Hessian conditioning, iteration "
        "counts) that gradient descent never produces.", styles['Body']))
    s.append(Paragraph(
        "This is interesting precisely because the resulting NLP is nontrivial: "
        "the most natural capacity constraint, a Lipschitz bound on the network, "
        "is bilinear and therefore nonconvex, so the feasible set is not convex "
        "and multiple local optima can exist. We build the problem symbolically "
        "with CasADi [1], which supplies exact first and second derivatives by "
        "automatic differentiation, and solve it with the interior-point method "
        "IPOPT [2]. The point is not to replace Adam at scale but to understand "
        "what is gained, and at what cost, by enforcing constraints exactly.",
        styles['Body']))
    s.append(Paragraph(
        "Concretely, this report addresses four research questions:", styles['Body']))
    questions = [
        "How do an exact constrained solver (IPOPT), an SQP method, and an "
        "unconstrained baseline (Adam) compare in solution quality and cost on "
        "the identical problem?",
        "How does tightening a hard Lipschitz bound trade training fit against "
        "generalization, and when does the constraint actually bind?",
        "How does the cost of the exact solve scale with the dimension of the "
        "NLP (the number of network weights)?",
        "How does enforcing a constraint exactly compare with the standard "
        "machine-learning practice of adding a soft penalty term to the loss?",
    ]
    for i, q in enumerate(questions, 1):
        s.append(Paragraph(f'<b>RQ{i}.</b> {q}', styles['Body']))
    return s


# ── 3. Problem formulation ───────────────────────────────────────────────────

def problem_formulation(styles):
    s = []
    s.append(Paragraph('2. Problem Formulation', styles['Heading']))
    s.append(Paragraph(
        "The student network is a single hidden layer with tanh activation, "
        "f(x; w) = W<sub>2</sub> tanh(W<sub>1</sub> x + b<sub>1</sub>) + b<sub>2</sub>. "
        "All weights and biases are stacked into a single decision vector "
        "w &isin; R<super>n</super> (the flattened W<sub>1</sub>, b<sub>1</sub>, "
        "W<sub>2</sub>, b<sub>2</sub>); for the default eight-unit network this "
        "is n = 25 variables. The objective is the mean-squared training error",
        styles['Body']))
    s.append(Paragraph(
        "f(w) = (1/N) &Sigma;<sub>i</sub> ( f(x<sub>i</sub>; w) &minus; y<sub>i</sub> )<super>2</super>,",
        ParagraphStyle('eq', parent=styles['Body'], alignment=TA_CENTER,
                       fontName='Helvetica-Oblique', spaceBefore=2, spaceAfter=6)))
    s.append(Paragraph(
        "minimized subject to up to three independently switchable constraints, "
        "described below. Figure&nbsp;1 sketches the network whose weights are "
        "the decision variables.", styles['Body']))

    # self-made sketch
    s.append(Spacer(1, 2))
    s.append(make_network_diagram())
    s.append(Paragraph('Sketch 1. The one-hidden-layer student network. The weight '
                       'and bias groups W1,b1 and W2,b2 form the decision vector w.',
                       styles['Caption']))

    s.append(Paragraph(
        "<b>(i) Lipschitz bound.</b> A tanh unit is 1-Lipschitz, so the network's "
        "Lipschitz constant is bounded by the product of the layer operator norms. "
        "Using the tractable Frobenius proxy gives "
        "||W<sub>1</sub>||<sub>F</sub><super>2</super> &middot; "
        "||W<sub>2</sub>||<sub>F</sub><super>2</super> &le; "
        "L<sub>max</sub><super>2</super>. This is a product of two convex terms "
        "and is therefore <i>bilinear and nonconvex</i>: the feasible set is not "
        "convex, the exact Hessian of the Lagrangian is indefinite, and the "
        "problem can admit several local minima. It is the constraint that makes "
        "this NLP genuinely hard and is the main object of study.", styles['Body']))
    s.append(Paragraph(
        "<b>(ii) Weight norm-ball.</b> A hard L2 ball on all weights, "
        "||w||<sub>2</sub><super>2</super> &le; B<sub>max</sub><super>2</super>. "
        "This is convex (a quadratic ball constraint) and is the exactly-enforced "
        "analogue of L2 weight-decay regularization: rather than penalizing large "
        "weights softly in the loss, it forbids them outright.", styles['Body']))
    s.append(Paragraph(
        "<b>(iii) Symmetry-breaking.</b> The hidden units of a one-layer network "
        "can be permuted without changing the function, so every minimizer has "
        "H! equivalent relabelings. Ordering the hidden biases, "
        "b<sub>1</sub>[0] &le; b<sub>1</sub>[1] &le; &hellip; &le; b<sub>1</sub>[H&minus;1], "
        "adds H&minus;1 linear inequalities that remove this combinatorial "
        "degeneracy without restricting the representable function class, which "
        "helps the solver converge.", styles['Body']))
    return s


# ── 4. Methods ───────────────────────────────────────────────────────────────

def methods(styles):
    s = []
    s.append(Paragraph('3. Methods', styles['Heading']))
    s.append(Paragraph(
        "<b>IPOPT (interior-point).</b> The constrained NLP is solved primarily "
        "with IPOPT [2], a primal-dual interior-point (barrier) method. Inequality "
        "constraints are handled through a logarithmic barrier whose weight is "
        "driven to zero across outer iterations, while each inner step solves a "
        "Newton system built from the exact Hessian of the Lagrangian supplied by "
        "CasADi. This yields fast local convergence and, as a by-product, the KKT "
        "multipliers used later for shadow-price analysis.", styles['Body']))
    s.append(Paragraph(
        "<b>SQP via CasADi.</b> As a second constrained solver we use CasADi's "
        "<i>sqpmethod</i>, a sequential quadratic programming scheme that solves a "
        "convex quadratic subproblem at each iterate. A limited-memory BFGS "
        "approximation keeps those subproblems convex, since the true Hessian is "
        "indefinite under the bilinear Lipschitz constraint. SQP reaches the same "
        "optima but, lacking the exact Hessian, typically needs far more "
        "iterations.", styles['Body']))
    s.append(Paragraph(
        "<b>Adam baseline (unconstrained).</b> As the machine-learning reference "
        "we run plain Adam on the same MSE objective with no constraints. Its "
        "gradient is taken from the identical CasADi expression, so the only "
        "differences are the algorithm and the absence of constraints. A penalty "
        "variant adds a soft hinge on the Lipschitz violation to study the "
        "penalty-versus-hard-constraint question.", styles['Body']))
    s.append(Paragraph(
        "<b>Teacher-student data.</b> All experiments use a synthetic "
        "teacher-student regression task: a fixed random one-hidden-layer network "
        "(the teacher) generates a smooth ground-truth function, sampled on "
        "[&minus;3, 3] and corrupted with Gaussian label noise. The student "
        "network must recover it from 60 noisy training points, with 40 held out "
        "for testing. This gives a known, controllable problem so that observed "
        "behaviour reflects the optimizer and the constraints, not data "
        "artefacts.", styles['Body']))
    return s


# ── 5. Results ───────────────────────────────────────────────────────────────

def method_table(styles):
    rows = [r for r in load_summary_rows() if r.get('group') == 'method_comparison']

    def label(r):
        if r['name'].endswith('spectral'):
            return 'IPOPT (spectral)'
        return r['method'].upper()

    order = {'exp_method_ipopt': 0, 'exp_method_sqp': 1,
             'exp_method_spectral': 2, 'exp_method_adam': 3}
    rows.sort(key=lambda r: order.get(r['name'], 99))

    header = ['Method', 'train MSE', 'test MSE', 'iterations', 'solve time [s]']
    data = [header]
    for r in rows:
        data.append([label(r), f5(r['train_mse']), f5(r['test_mse']),
                     str(r.get('n_iter', '--')), f3(r['solve_time'])])

    tbl = Table(data, colWidths=[4.4 * cm, 2.7 * cm, 2.7 * cm, 2.4 * cm, 2.8 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3d6fa5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tbl


def results(styles):
    s = []
    s.append(Paragraph('4. Results', styles['Heading']))

    # ---- Table + Fig 1: method comparison ----
    ip = load_json('exp_method_ipopt')
    sq = load_json('exp_method_sqp')
    ad = load_json('exp_method_adam')
    s.append(Paragraph(
        "<b>4.1 Optimizer comparison.</b> Table&nbsp;1 reports the three core "
        "optimizers on the identical eight-unit problem (n = 25 variables, nine "
        "active constraint rows for the constrained solvers). IPOPT and SQP reach "
        f"essentially the same fit -- train MSE {f5(ip.get('train_mse'))} and "
        f"{f5(sq.get('train_mse'))} respectively -- but IPOPT needs only "
        f"{ip.get('n_iter','--')} iterations against SQP's {sq.get('n_iter','--')}, "
        "because IPOPT uses the exact Hessian whereas SQP relies on a BFGS "
        "approximation. Adam, solving the easier unconstrained problem, reaches a "
        f"comparable test MSE ({f5(ad.get('test_mse'))}) but its solution is a "
        "different point that does not respect the hard constraints. The numbers "
        "below are read directly from results/summary.csv.", styles['Body']))
    s.append(method_table(styles))
    s.append(Paragraph('Table 1. Method comparison group, read from '
                       'results/summary.csv (IPOPT, SQP, exact-spectral IPOPT, Adam).',
                       styles['Caption']))
    s += figure_block(
        'fig_method_comparison.png',
        "Fig. 1. IPOPT vs SQP vs Adam: train/test MSE and solve time. IPOPT and "
        "SQP solve the same constrained NLP to similar quality; Adam solves a "
        "different (unconstrained) problem.", styles)

    # ---- Fig 2: Lipschitz sweep ----
    Ls = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    sweep = []
    for L in Ls:
        tag = str(L).replace('.', 'p')
        d = load_json(f'exp_lipschitz_L{tag}')
        if d:
            sweep.append(d)
    if sweep:
        lo, hi = sweep[0], sweep[-1]
        s.append(Paragraph(
            "<b>4.2 Lipschitz bound sweep.</b> Sweeping the bound from "
            f"L<sub>max</sub> = {hi['L_max']:g} down to {lo['L_max']:g} steadily "
            f"trades fit for smoothness: training MSE rises from {f5(hi['train_mse'])} "
            f"at the loosest bound to {f5(lo['train_mse'])} at the tightest, as the "
            "network is denied capacity. The held-out error moves the other way -- "
            f"test MSE falls from {f5(hi['test_mse'])} to {f5(lo['test_mse'])} -- so "
            "the constraint behaves as a tunable regularizer. The achieved "
            "Lipschitz value tracks the bound almost exactly at every setting "
            f"(e.g. {f3(lo['lipschitz_estimate'])} at L<sub>max</sub> = {lo['L_max']:g}), "
            "confirming the constraint is active across the whole range.",
            styles['Body']))
    s += figure_block(
        'fig_lipschitz_sweep.png',
        "Fig. 2. Lipschitz bound sweep. As L_max tightens, train MSE rises (less "
        "capacity) while test MSE falls (better generalization). The constraint "
        "is active (binding) across the full range.", styles)

    # ---- Fig 3: KKT analysis ----
    kkt = []
    for L in Ls:
        tag = str(L).replace('.', 'p')
        d = load_json(f'exp_kkt_L{tag}')
        if d:
            kkt.append(d)
    if kkt:
        lam_lo = kkt[0].get('lam_lipschitz')
        lam_hi = kkt[-1].get('lam_lipschitz')
        s.append(Paragraph(
            "<b>4.3 KKT shadow price.</b> IPOPT returns a Lagrange multiplier "
            "(dual variable) for the Lipschitz constraint -- its shadow price, the "
            "marginal objective improvement per unit relaxation of the bound. The "
            f"multiplier is strictly positive throughout, largest at the tightest "
            f"bound (lambda &asymp; {sci(lam_lo)} at L<sub>max</sub> = {kkt[0]['L_max']:g}) "
            f"and smallest at the loosest (lambda &asymp; {sci(lam_hi)} at "
            f"L<sub>max</sub> = {kkt[-1]['L_max']:g}). A positive multiplier with "
            "complementary slackness is the formal certificate that the constraint "
            "is binding: tightening it genuinely costs fit, exactly the trade-off "
            "seen in Section&nbsp;4.2.", styles['Body']))
    s += figure_block(
        'fig_kkt_analysis.png',
        "Fig. 3. KKT dual variable (shadow price) of the Lipschitz constraint vs "
        "L_max. lambda > 0 confirms the constraint is binding; lambda ~ 0 means "
        "it is slack.", styles)

    # ---- Fig 4: size scaling ----
    sizes = []
    for Hn in [4, 8, 16, 32, 64]:
        d = load_json(f'exp_size_H{Hn}')
        if d:
            sizes.append(d)
    if sizes:
        a, b = sizes[0], sizes[-1]
        ratio = (float(b['solve_time']) / float(a['solve_time'])
                 if float(a['solve_time']) else float('nan'))
        s.append(Paragraph(
            "<b>4.4 NLP size scaling.</b> Growing the hidden layer grows the NLP: "
            f"from H = {a['H']} ({a['n_vars']} variables) to H = {b['H']} "
            f"({b['n_vars']} variables) the iteration count stays modest "
            f"({a.get('n_iter','--')} to {b.get('n_iter','--')}), but the solve "
            f"time climbs from {f3(a['solve_time'])} s to {f3(b['solve_time'])} s -- "
            f"about a {ratio:.0f}x increase for a roughly 15x larger variable "
            "count. The growth is superlinear because each interior-point step "
            "factorizes a Hessian/KKT system whose cost scales worse than linearly "
            "in the problem dimension.", styles['Body']))
    s += figure_block(
        'fig_size_scaling.png',
        "Fig. 4. Solve time and IPOPT iteration count vs NLP dimension (hidden "
        "units H). Superlinear growth in time reflects the cost of Hessian "
        "factorization.", styles)

    # ---- Fig 5: noise robustness ----
    noise = {}
    for nz in ['0p0', '0p1', '0p3']:
        noise[(nz, 'ipopt')] = load_json(f'exp_noise_{nz}_ipopt')
        noise[(nz, 'adam')] = load_json(f'exp_noise_{nz}_adam')
    ni, na = noise[('0p3', 'ipopt')], noise[('0p3', 'adam')]
    if ni and na:
        s.append(Paragraph(
            "<b>4.5 Noise robustness.</b> Repeating the constrained IPOPT solve "
            "and the unconstrained Adam run at increasing label noise isolates the "
            "regularization value of the hard constraints. At the highest noise "
            f"level (sigma = {ni['noise_std']:g}) the constrained model attains "
            f"test MSE {f5(ni['test_mse'])} versus Adam's {f5(na['test_mse'])}, and "
            "crucially the unconstrained solution lets the network's Lipschitz "
            f"value blow up to {f3(na['lipschitz_estimate'])} while the constrained "
            f"one is held at {f3(ni['lipschitz_estimate'])}. The hard bound stops "
            "the model from fitting the noise.", styles['Body']))
    s += figure_block(
        'fig_noise_robustness.png',
        "Fig. 5. Train/test MSE vs noise level for constrained IPOPT and "
        "unconstrained Adam. Hard constraints provide a regularization benefit at "
        "higher noise.", styles)

    # ---- Fig 6: penalty vs hard ----
    ref = load_json('exp_penalty_ipopt_ref')
    p0 = load_json('exp_penalty_rho0p0')
    pmid = load_json('exp_penalty_rho0p0001')
    if ref and p0 and pmid:
        s.append(Paragraph(
            "<b>4.6 Penalty versus hard constraint.</b> The standard ML way to "
            "honour a constraint is a soft penalty term. With no penalty the "
            f"Lipschitz constraint is violated by {f3(p0['max_constraint_violation'])}; "
            "raising the penalty weight rho shrinks the violation (to "
            f"{sci(pmid['max_constraint_violation'])} at rho = {pmid.get('rho')}) but "
            "never reaches zero and begins to distort the fit. IPOPT, by contrast, "
            f"enforces the bound exactly -- violation {sci(ref['max_constraint_violation'])} "
            f"-- at test MSE {f5(ref['test_mse'])}. Exact enforcement is the only "
            "way to guarantee feasibility.", styles['Body']))
    s += figure_block(
        'fig_penalty_vs_hard.png',
        "Fig. 6. Penalty Adam vs exact IPOPT enforcement. As rho increases, "
        "penalty Adam reduces constraint violation but at the cost of fit "
        "quality. IPOPT enforces zero violation by construction.", styles)
    return s


# ── 6. Discussion & Conclusion ───────────────────────────────────────────────

def discussion(styles):
    s = []
    s.append(Paragraph('5. Discussion and Conclusion', styles['Heading']))

    # multistart summary, read from the JSONs
    ms = []
    for i in range(20):
        d = load_json(f'exp_multistart_seed{i}')
        if d:
            ms.append(d)
    if ms:
        mses = sorted(float(d['train_mse']) for d in ms)
        spread = mses[-1] - mses[0]
        s.append(Paragraph(
            "<b>What worked.</b> Posing training as a constrained NLP and solving "
            "it with IPOPT worked cleanly and gave information no first-order "
            "method provides: exact constraint satisfaction, shadow prices that "
            "quantify how much each bound costs, and a controllable fit/"
            "generalization dial through L<sub>max</sub>. The CasADi-supplied exact "
            "Hessian made IPOPT markedly more iteration-efficient than BFGS-based "
            "SQP on the same problem.", styles['Body']))
        s.append(Paragraph(
            "<b>Nonconvexity.</b> The bilinear Lipschitz constraint makes the "
            f"problem nonconvex, but the 20-start study was reassuring: across all "
            f"{len(ms)} random initializations the final training MSE ranged only "
            f"from {f5(mses[0])} to {f5(mses[-1])} (a spread of {sci(spread)}). The "
            "landscape has multiple stationary points -- visible as different "
            "iteration counts and basins -- yet they are of nearly equivalent "
            "quality here, so multistart is cheap insurance rather than a "
            "necessity at this scale.", styles['Body']))
    s.append(Paragraph(
        "<b>Limitations and scaling.</b> The main limitation is cost. Section "
        "4.4 shows solve time growing superlinearly with the weight count because "
        "every interior-point step factorizes a dense KKT system. For the small "
        "networks studied here this is irrelevant -- seconds at most -- but it "
        "makes exact second-order constrained solves impractical for the "
        "million-parameter networks of mainstream deep learning, where Adam's "
        "cheap iterations win. The method's niche is small-to-medium models where "
        "guaranteed constraint satisfaction matters more than raw scale, such as "
        "certified-Lipschitz or safety-constrained networks.", styles['Body']))
    s.append(Paragraph(
        "<b>Penalty versus hard.</b> The penalty comparison was the clearest "
        "practical lesson: a soft penalty can be tuned to <i>reduce</i> a "
        "violation but never to eliminate it, and pushing the penalty weight up "
        "starts to corrupt the objective. When a bound must hold exactly -- a "
        "true Lipschitz certificate, a hard resource budget -- only explicit "
        "constraint handling delivers it, and IPOPT does so by construction.",
        styles['Body']))
    s.append(Paragraph(
        "<b>Conclusion.</b> Treating neural-network training as a constrained NLP "
        "turns implicit modelling choices into explicit, exactly-enforced "
        "constraints with interpretable dual information. On a controlled "
        "teacher-student task the constrained solver matched or beat the "
        "unconstrained baseline in generalization, enforced its bounds exactly "
        "where a penalty could not, and revealed -- through shadow prices and "
        "scaling curves -- precisely when each constraint mattered and what it "
        "cost. The approach is a powerful analysis and design tool for the "
        "small-network regime, and a clarifying lens on what regularization "
        "really does.", styles['Body']))
    return s


# ── 7. References ────────────────────────────────────────────────────────────

def references(styles):
    s = []
    s.append(Paragraph('References', styles['Heading']))
    s.append(Paragraph(
        "[1] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, "
        "&ldquo;CasADi: a software framework for nonlinear optimization and optimal "
        "control,&rdquo; <i>Mathematical Programming Computation</i>, vol. 11, "
        "no. 1, pp. 1&ndash;36, 2019.", styles['Reference']))
    s.append(Paragraph(
        "[2] A. W&auml;chter and L. T. Biegler, &ldquo;On the implementation of an "
        "interior-point filter line-search algorithm for large-scale nonlinear "
        "programming,&rdquo; <i>Mathematical Programming</i>, vol. 106, no. 1, "
        "pp. 25&ndash;57, 2006.", styles['Reference']))
    return s


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    build_document()
    size = os.path.getsize(OUT_PDF)
    if size <= 100 * 1024:
        raise SystemExit(f'ERROR: PDF only {size} bytes (expected > 100KB)')
    print(f'Report generated successfully: reports/project_report.pdf '
          f'({size/1024:.0f} KB)')
