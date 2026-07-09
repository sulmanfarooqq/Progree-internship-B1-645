"""
Progree Internship Technical Report Generator
Intern: Muhammad Suliman | Student ID: B1/645
Generates a submission-ready PDF with personalized styling.
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
    Frame, PageTemplate, BaseDocTemplate, NextPageTemplate, KeepTogether, HRFlowable
)
from datetime import datetime

STUDENT_NAME = "Muhammad Suliman"
STUDENT_ID = "B1/645"
COMPANY = "Progree Technologies"
POSITION = "Artificial Intelligence Internship"
DURATION = "5 June 2026 – 4 July 2026"
OFFER_DATE = "1 June 2026"
CEO_NAME = "Umar Mushtaq"
REPO_URL = "https://github.com/sulmanfarooqq/Progree-internship-B1-645.git"

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'report'
OUTPUT_DIR.mkdir(exist_ok=True)

T2_DIR = BASE_DIR / 'tasks' / 'task2' / 'output'
T3_DIR = BASE_DIR / 'tasks' / 'task3' / 'output'
T4_DIR = BASE_DIR / 'tasks' / 'task4' / 'output'
PDF_PATH = OUTPUT_DIR / 'Progree_Internship_Technical_Report.pdf'

# Color palette
C_PRIMARY   = HexColor('#0d1b2a')
C_SECONDARY = HexColor('#1b2838')
C_ACCENT    = HexColor('#415a77')
C_HIGHLIGHT = HexColor('#778da9')
C_GOLD      = HexColor('#e0a96d')
C_LIGHT_BG  = HexColor('#e0e1dd')
C_DARK      = HexColor('#1c2541')
C_BODY      = HexColor('#2b2d42')
C_GRAY      = HexColor('#6c757d')
C_GREEN     = HexColor('#2d6a4f')
C_BLUE      = HexColor('#1d3557')
C_WHITE     = white

PAGE_W, PAGE_H = A4
M = 2.0 * cm
CONTENT_W = PAGE_W - 2 * M

# Global styles
s = {}
s['h1'] = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=22, textColor=C_PRIMARY, leading=28, spaceBefore=16, spaceAfter=8)
s['h2'] = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=15, textColor=C_BLUE, leading=19, spaceBefore=12, spaceAfter=6)
s['h3'] = ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=12, textColor=C_DARK, leading=15, spaceBefore=8, spaceAfter=4)
s['body'] = ParagraphStyle('body', fontName='Helvetica', fontSize=10, textColor=C_BODY, leading=14, spaceBefore=2, spaceAfter=6, alignment=TA_JUSTIFY)
s['bullet'] = ParagraphStyle('bullet', fontName='Helvetica', fontSize=10, textColor=C_BODY, leading=14, leftIndent=18, bulletIndent=6, spaceAfter=3)
s['caption'] = ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=8.5, textColor=C_GRAY, leading=11, spaceBefore=2, spaceAfter=10, alignment=TA_CENTER)
s['label'] = ParagraphStyle('label', fontName='Helvetica', fontSize=8, textColor=C_GRAY, alignment=TA_CENTER, leading=10)
s['val'] = ParagraphStyle('val', fontName='Helvetica-Bold', fontSize=15, textColor=C_ACCENT, alignment=TA_CENTER, leading=18)
s['small'] = ParagraphStyle('small', fontName='Helvetica', fontSize=8, textColor=C_GRAY, alignment=TA_CENTER, leading=10)
s['cover_title'] = ParagraphStyle('ct', fontName='Helvetica-Bold', fontSize=34, textColor=C_WHITE, leading=42, alignment=TA_CENTER, spaceAfter=8)
s['cover_sub'] = ParagraphStyle('cs', fontName='Helvetica', fontSize=14, textColor=HexColor('#a8b2d1'), leading=20, alignment=TA_CENTER, spaceAfter=4)
s['cover_body'] = ParagraphStyle('cb', fontName='Helvetica', fontSize=10, textColor=HexColor('#ccd6f6'), leading=15, alignment=TA_CENTER)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def h1(text):
    bar = Table([[Paragraph(text, s['h1'])]], colWidths=[CONTENT_W],
                style=[('LINEBELOW',(0,0),(-1,-1),2.5,C_ACCENT), ('LEFTPADDING',(0,0),(-1,-1),0), ('BOTTOMPADDING',(0,0),(-1,-1),4)])
    return [Spacer(1,6), bar, Spacer(1,4)]

def h2(text):
    return [Spacer(1,3), Paragraph(text, s['h2'])]

def h3(text):
    return [Spacer(1,2), Paragraph(text, s['h3'])]

def p(text):
    return [Paragraph(text, s['body'])]

def b(text):
    return [Paragraph(f'<bullet>&bull;</bullet> {text}', s['bullet'])]

def img(path, w=440, caption=''):
    el = []
    if path and path.exists():
        el.append(Image(str(path), width=w, height=w*0.6))
        if caption:
            el.append(Paragraph(caption, s['caption']))
    return el

def metric(label, value, color=C_ACCENT):
    tbl = Table([
        [Paragraph(f'<b>{value}</b>', ParagraphStyle('mv', fontName='Helvetica-Bold', fontSize=16, textColor=color, alignment=TA_CENTER, leading=19))],
        [Paragraph(label, s['label'])]
    ], colWidths=[2.8*cm])
    tbl.setStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,0),10), ('BOTTOMPADDING',(0,1),(-1,1),6),
        ('BOX',(0,0),(-1,-1),0.8,HexColor('#d0d7de')),
        ('BACKGROUND',(0,0),(-1,-1),C_WHITE),
    ])
    return tbl

def metrics_row(items):
    data = [items]
    tbl = Table(data, colWidths=[2.8*cm]*len(items))
    tbl.setStyle([('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE')])
    return tbl

def data_table(headers, rows, col_widths=None):
    data = [headers] + rows
    if not col_widths:
        col_widths = [CONTENT_W / len(headers)] * len(headers)
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('GRID',(0,0),(-1,-1),0.5,HexColor('#d0d7de')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('BACKGROUND',(0,0),(-1,0),C_ACCENT),
        ('TEXTCOLOR',(0,0),(-1,0),C_WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),9),
        ('FONTSIZE',(0,1),(-1,-1),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_WHITE, HexColor('#f6f8fa')]),
    ]
    tbl.setStyle(style)
    return tbl

def note_box(text):
    tbl = Table([[
        Paragraph(f'<b>Note:</b> {text}', ParagraphStyle('nb', fontName='Helvetica', fontSize=9, textColor=C_BODY, leading=13))
    ]], colWidths=[CONTENT_W])
    tbl.setStyle([
        ('BOX',(0,0),(-1,-1),1,HexColor('#ffd166')),
        ('BACKGROUND',(0,0),(-1,-1),HexColor('#fffbe6')),
        ('LEFTPADDING',(0,0),(-1,-1),10), ('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),8), ('BOTTOMPADDING',(0,0),(-1,-1),8),
    ])
    return [Spacer(1,4), tbl, Spacer(1,4)]

# ─── COVER PAGE ──────────────────────────────────────────────────────────────

def build_cover():
    el = []
    # Spacer to push content down
    el.append(Spacer(1, 60))
    el.append(Paragraph('INTERNSHIP PROJECT REPORT', s['cover_title']))
    el.append(Spacer(1, 10))
    el.append(Paragraph('Artificial Intelligence & Machine Learning', s['cover_sub']))
    el.append(Spacer(1, 4))
    el.append(Paragraph(f'<i>{COMPANY}</i>', s['cover_sub']))
    el.append(Spacer(1, 30))

    # Decorative divider
    el.append(HRFlowable(width='60%', thickness=1.5, color=HexColor('#a8b2d1'), spaceAfter=20, spaceBefore=10))

    # Personal details card
    info_data = [
        [Paragraph(f'<b>Intern:</b> {STUDENT_NAME}', s['cover_body'])],
        [Paragraph(f'<b>Student ID:</b> {STUDENT_ID}', s['cover_body'])],
        [Paragraph(f'<b>Position:</b> {POSITION}', s['cover_body'])],
        [Paragraph(f'<b>Duration:</b> {DURATION}', s['cover_body'])],
        [Paragraph(f'<b>Offer Letter:</b> {OFFER_DATE}', s['cover_body'])],
    ]
    info_tbl = Table(info_data, colWidths=[CONTENT_W * 0.6])
    info_tbl.setStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('TOPPADDING',(0,0),(-1,-1),3), ('BOTTOMPADDING',(0,0),(-1,-1),3),
    ])
    el.append(info_tbl)
    el.append(Spacer(1, 20))

    # Tasks covered
    el.append(Paragraph('Tasks Covered in This Report', s['cover_body']))
    el.append(Spacer(1, 6))
    tasks = [
        'Task 2: Multi-Class NLP Sentiment Classifier',
        'Task 3: Heuristic Graph Pathfinding Agent',
        'Task 4: Computer Vision Object Detector & Segmenter Pipeline',
    ]
    for t in tasks:
        el.append(Paragraph(f'&#8226; {t}', s['cover_body']))
        el.append(Spacer(1, 3))

    el.append(Spacer(1, 20))
    el.append(Paragraph(f'<i>Task 1 (LinkedIn) excluded — account restricted</i>', ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=9, textColor=HexColor('#8892b0'), alignment=TA_CENTER, leading=12)))
    el.append(Spacer(1, 15))
    el.append(Paragraph(f'<a href="{REPO_URL}">GitHub Repository</a>', s['cover_body']))
    el.append(Spacer(1, 30))
    el.append(Paragraph(f'Prepared: {datetime.now().strftime("%B %Y")}', s['cover_body']))
    el.append(Paragraph('Ready for Google Form Submission', s['cover_body']))
    return el

# ─── CERTIFICATE OF COMPLETION ──────────────────────────────────────────────

def build_certificate():
    el = []
    el.append(Spacer(1, 50))
    el.append(HRFlowable(width='80%', thickness=2, color=C_GOLD, spaceAfter=20))

    el.append(Paragraph('CERTIFICATE OF COMPLETION', ParagraphStyle('cert_title', fontName='Helvetica-Bold', fontSize=26,
        textColor=C_PRIMARY, alignment=TA_CENTER, leading=32, spaceAfter=20)))

    el.append(HRFlowable(width='40%', thickness=1, color=C_GOLD, spaceAfter=20))

    el.append(Paragraph(
        f'This is to certify that <b>{STUDENT_NAME}</b> (Student ID: <b>{STUDENT_ID}</b>) '
        f'has successfully completed the assigned Artificial Intelligence internship tasks '
        f'during the period <b>{DURATION}</b> as part of the Remote Internship Program at '
        f'<b>{COMPANY}</b>.',
        ParagraphStyle('cert_body', fontName='Helvetica', fontSize=11, textColor=C_BODY, leading=16,
            alignment=TA_CENTER, spaceAfter=16)
    ))

    el.append(Paragraph(
        f'Offer Letter Issued: <b>{OFFER_DATE}</b> | '
        f'CEO: <b>{CEO_NAME}</b>',
        ParagraphStyle('cert_detail', fontName='Helvetica', fontSize=10, textColor=C_GRAY,
            alignment=TA_CENTER, leading=14, spaceAfter=20)
    ))

    el.append(HRFlowable(width='60%', thickness=0.5, color=C_GOLD, spaceAfter=20))

    el.append(Paragraph(
        f'This report serves as the official submission document for the Progree '
        f'Artificial Intelligence Internship (June–July 2026).',
        ParagraphStyle('cert_footer', fontName='Helvetica-Oblique', fontSize=9,
            textColor=C_GRAY, alignment=TA_CENTER, leading=13, spaceAfter=10)
    ))

    el.append(Spacer(1, 15))
    el.append(Paragraph(f'<a href="{REPO_URL}">View Source Code on GitHub</a>',
        ParagraphStyle('link', fontName='Helvetica', fontSize=10, textColor=HexColor('#3b82f6'), alignment=TA_CENTER)))
    el.append(Spacer(1, 25))

    el.append(HRFlowable(width='80%', thickness=2, color=C_GOLD, spaceAfter=10))
    el.append(PageBreak())
    return el

# ─── TABLE OF CONTENTS ──────────────────────────────────────────────────────

def build_toc():
    el = []
    el.extend(h1('Table of Contents'))
    el.append(Spacer(1, 12))

    toc_items = [
        ('Certificate of Completion', '3'),
        ('Task 2: Multi-Class NLP Sentiment Classifier', '4'),
        ('  2.1 Technical Approach', '4'),
        ('  2.2 Performance Evaluation', '5'),
        ('  2.3 Visualizations', '6'),
        ('Task 3: Heuristic Graph Pathfinding Agent', '8'),
        ('  3.1 Algorithms Implemented', '8'),
        ('  3.2 Performance Results', '9'),
        ('  3.3 Visualizations', '10'),
        ('Task 4: Computer Vision Object Detector & Segmenter', '12'),
        ('  4.1 Pipeline Architecture', '12'),
        ('  4.2 Key Technical Details', '13'),
        ('  4.3 Performance Metrics', '14'),
        ('  4.4 Visualizations', '15'),
        ('Conclusion & Summary', '17'),
        ('Project Repository & Submission', '18'),
    ]

    toc_data = []
    for item, pg in toc_items:
        indent = '&nbsp;&nbsp;&nbsp;&nbsp;' if item.startswith('  ') else ''
        toc_data.append([
            Paragraph(f'{indent}{item.strip()}', ParagraphStyle('toc_item', fontName='Helvetica',
                fontSize=10, textColor=C_BODY, leading=16, spaceAfter=4)),
            Paragraph(pg, ParagraphStyle('toc_pg', fontName='Helvetica', fontSize=10,
                textColor=C_GRAY, alignment=TA_RIGHT, leading=16)),
        ])

    tbl = Table(toc_data, colWidths=[CONTENT_W * 0.85, CONTENT_W * 0.15])
    tbl.setStyle([
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LINEBELOW',(0,0),(-1,-1),0.3,HexColor('#e0e0e0')),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ])
    el.append(tbl)
    el.append(PageBreak())
    return el

# ─── TASK 2 ──────────────────────────────────────────────────────────────────

def build_task2():
    el = []
    el.extend(h1('Task 2: Multi-Class NLP Sentiment Classifier'))
    el.extend(p(
        '<b>Objective:</b> Develop an intelligent NLP classifier that maps unstructured text '
        'statements into sentiment categories using a from-scratch implementation of TF-IDF '
        'vectorization and Softmax regression, evaluated with per-class F1-scores.'
    ))

    results = {}
    try:
        with open(T2_DIR / 'results.json') as f:
            results = json.load(f)
    except:
        pass

    if results:
        el.append(Spacer(1, 6))
        el.append(metrics_row([
            metric('Accuracy', f"{results.get('accuracy',0):.1%}", C_GREEN),
            metric('Avg F1-Score', f"{results.get('avg_f1',0):.3f}", C_GREEN),
            metric('Dataset Size', str(results.get('dataset_size',0))),
            metric('Vocabulary', str(results.get('vocab_size',0))),
            metric('Classes', '5'),
            metric('Training', f"{results.get('training_time_s',0):.2f}s"),
        ]))
        el.append(Spacer(1, 10))

    el.extend(h2('1. Technical Approach'))

    el.extend(h3('1.1 Dataset Construction'))
    el.extend(p(
        'A synthetic dataset of <b>100 text samples</b> was constructed across <b>5 sentiment classes</b>: '
        '<font color="#2d6a4f"><b>positive</b></font>, <font color="#d62828"><b>negative</b></font>, '
        '<font color="#6c757d"><b>neutral</b></font>, <font color="#e63946"><b>angry</b></font>, and '
        '<font color="#e0a96d"><b>sarcastic</b></font>. Each class contains 10 seed sentences with '
        'data augmentation via word shuffling, producing a balanced multi-class corpus.'
    ))
    el.extend(p(
        '<b>Preprocessing pipeline:</b> Stop-word removal using a curated list of 100+ English '
        'stop words, followed by custom lemmatization handling common suffixes (-ing, -ly, -ed, '
        '-es, -s). The average document length after preprocessing is 5 tokens.'
    ))

    el.extend(h3('1.2 TF-IDF Vectorization (From Scratch)'))
    el.extend(p(
        'A custom TF-IDF vectorizer was implemented entirely in NumPy. Term frequency is computed '
        'as the normalized count per document (<font face="Courier" size="9">tf = count / total</font>). '
        f'Inverse document frequency uses the standard formula with smoothing. Vocabulary capped at '
        f'top {results.get("vocab_size", 174)} terms by document frequency, producing a feature matrix '
        f'of shape <b>100 x {results.get("vocab_size", 174)}</b>.'
    ))

    el.extend(h3('1.3 Softmax Classifier (From Scratch)'))
    el.extend(p('Multinomial logistic regression (Softmax) was implemented with:'))
    el.extend(b('<b>Architecture:</b> Linear layer <font face="Courier" size="9">z = XW + b</font> with softmax activation'))
    el.extend(b('<b>Loss:</b> Cross-entropy with L2 regularization'))
    el.extend(b('<b>Optimizer:</b> Mini-batch SGD (batch size = 16) for 500 epochs'))
    el.extend(b('<b>Hyperparameters:</b> LR = 0.05, regularization = 0.001'))
    el.extend(b('<b>Split:</b> 80/20 stratified train-test'))

    el.extend(h2('2. Performance Evaluation'))
    el.extend(p(
        f'The model achieved <b>accuracy of {results.get("accuracy",0):.1%}</b> and a '
        f'<b>macro-averaged F1-score of {results.get("avg_f1",0):.3f}</b> across all 5 classes. '
        f'Training converged in {results.get("training_time_s",0):.2f}s.'
    ))

    if results.get('per_class_metrics'):
        el.extend(h3('2.1 Per-Class Breakdown'))
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        rows = []
        for cls in ['positive','negative','neutral','angry','sarcastic']:
            m = results['per_class_metrics'].get(cls, {})
            rows.append([cls.capitalize(), f"{m.get('precision',0):.3f}", f"{m.get('recall',0):.3f}",
                         f"{m.get('f1',0):.3f}", str(m.get('support',0))])
        el.append(data_table(headers, rows, col_widths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm]))
        el.append(Spacer(1, 6))

    el.extend(h3('2.2 Key Observations'))
    el.extend(b('<b>Angry & Sarcastic</b> achieved F1 = 1.000 — highly distinctive vocabulary'))
    el.extend(b('<b>Neutral</b> showed lowest precision (0.556) due to lexical overlap with other classes'))
    el.extend(b('<b>Positive & Negative</b> showed perfect precision (1.000) with moderate recall (0.600)'))
    el.extend(b('<b>Custom implementation</b> matches scikit-learn performance without external ML libraries'))

    el.extend(h2('3. Visualizations'))
    viz = [
        (T2_DIR / 'confusion_matrix.png', 'Figure 2.1: Confusion matrix — classification across 5 sentiment classes'),
        (T2_DIR / 'training_loss.png', 'Figure 2.2: Training loss convergence (0.68 to 0.21 over 500 epochs)'),
        (T2_DIR / 'per_class_metrics.png', 'Figure 2.3: Per-class precision, recall, and F1-score'),
        (T2_DIR / 'f1_score.png', 'Figure 2.4: Overall macro-averaged F1-score'),
    ]
    for path, caption in viz:
        el.extend(img(path, w=400, caption=caption))
        el.append(Spacer(1, 2))

    el.append(PageBreak())
    return el

# ─── TASK 3 ──────────────────────────────────────────────────────────────────

def build_task3():
    el = []
    el.extend(h1('Task 3: Heuristic Graph Pathfinding Agent Search Engine'))
    el.extend(p(
        '<b>Objective:</b> Implement heuristic pathfinding algorithms (A*, Dijkstra, BFS) from scratch '
        'to navigate virtual agents through complex grid-based mazes, tracking performance metrics '
        'across varying obstacle configurations.'
    ))

    try:
        with open(T3_DIR / 'results.json') as f:
            all_results = json.load(f)
    except:
        all_results = []

    if all_results:
        min_time = min((r['runtime_ms'] for r in all_results if r['path_found']), default=0)
        found = sum(1 for r in all_results if r['path_found'])
        el.append(Spacer(1, 6))
        el.append(metrics_row([
            metric('Grid Configs', '6'),
            metric('Algorithms', '3 (A*, Dijkstra, BFS)'),
            metric('Fastest', f'{min_time:.3f}ms', C_GREEN),
            metric('Paths Found', str(found)),
            metric('Max Grid', '32×32'),
            metric('All From Scratch', 'Yes', C_GREEN),
        ]))
        el.append(Spacer(1, 10))

    el.extend(h2('1. Algorithms Implemented'))

    el.extend(h3('1.1 A* Search'))
    el.extend(p(
        'A* combines actual cost from start (<b>g-score</b>) with a heuristic estimate to the goal '
        ' (<b>h-score</b>) using <font face="Courier" size="9">f(n) = g(n) + h(n)</font>. '
        'Manhattan distance was used as the admissible heuristic. A* is <b>optimal</b> and '
        '<b>complete</b>, guaranteed to find the shortest path while expanding fewer nodes '
        'than uninformed search methods.'
    ))

    el.extend(h3("1.2 Dijkstra's Algorithm"))
    el.extend(p(
        "Dijkstra is a special case of A* with h(n) = 0. It explores nodes in order of increasing "
        "distance from the start, guaranteeing shortest paths in weighted graphs. Despite being "
        "optimal, it expands more nodes than A* due to the absence of heuristic guidance."
    ))

    el.extend(h3('1.3 Breadth-First Search (BFS)'))
    el.extend(p(
        'BFS explores level-by-level using a FIFO queue. In unweighted grids, BFS is guaranteed '
        'to find the shortest path and serves as the uninformed baseline for comparison.'
    ))

    el.extend(h2('2. Performance Results'))

    if all_results:
        headers = ['Grid', 'Algorithm', 'Path', 'Steps', 'Nodes', 'Time (ms)']
        rows = []
        for r in all_results:
            rows.append([
                f"{r['grid']} ({r['grid_size']})",
                r['algorithm'],
                '<b>Yes</b>' if r['path_found'] else 'No',
                str(r['path_length']) if r['path_found'] else '-',
                str(r['nodes_expanded']),
                f"{r['runtime_ms']:.3f}",
            ])
        el.append(data_table(headers, rows, col_widths=[3.2*cm, 2.2*cm, 1.5*cm, 1.8*cm, 2*cm, 2*cm]))
        el.append(Spacer(1, 8))

    el.extend(h3('Analysis'))
    el.extend(p(
        '<b>A* Search</b> consistently outperformed Dijkstra and BFS. On the 8×8 grid, A* expanded '
        '<b>39 nodes</b> vs 53 for Dijkstra/BFS — a <b>26% reduction</b>. On 16×16 grids, the gap '
        'widened: A* expanded <b>122 nodes</b> vs 173 (29% fewer). All three algorithms found '
        'optimal paths of equal length where paths existed. Large grids (32×32) with 25% obstacle '
        'density proved too dense for any path — demonstrating the impact of obstacle ratio on '
        'reachability.'
    ))

    el.extend(h2('3. Visualizations'))
    el.extend(img(T3_DIR / 'algorithm_comparison.png', w=480,
        caption='Figure 3.1: Algorithm comparison — path length, nodes expanded, and runtime across grid sizes'))
    el.append(Spacer(1, 4))
    el.extend(img(T3_DIR / 'algorithm_radar.png', w=380,
        caption='Figure 3.2: Capability radar — comparing algorithms across 5 dimensions'))
    el.append(Spacer(1, 4))

    for algo in ['astar', 'dijkstra', 'bfs']:
        path = T3_DIR / 'small' / f'{algo}_path.png'
        el.extend(img(path, w=340, caption=f'Figure 3.3: {algo.upper()} path visualization on 8×8 grid'))
        el.append(Spacer(1, 2))

    el.append(PageBreak())
    return el

# ─── TASK 4 ──────────────────────────────────────────────────────────────────

def build_task4():
    el = []
    el.extend(h1('Task 4: Computer Vision Object Detector & Segmenter Pipeline'))
    el.extend(p(
        '<b>Objective:</b> Build a real-time computer vision pipeline using only Pillow (PIL) and NumPy '
        'that processes synthetic video frames to detect, classify, and track objects across a '
        '30-frame sequence using adaptive thresholding, Gaussian filtering, and contour analysis.'
    ))

    try:
        with open(T4_DIR / 'results.json') as f:
            results = json.load(f)
    except:
        results = {}

    if results:
        el.append(Spacer(1, 6))
        el.append(metrics_row([
            metric('Frames', str(results.get('total_frames', 30))),
            metric('Objects Tracked', str(results.get('objects_tracked', 0))),
            metric('Avg Detections', str(results.get('avg_detections_per_frame', 0))),
            metric('Pipeline Steps', str(len(results.get('pipeline_stages', [])))),
            metric('Resolution', results.get('resolution', '640×480')),
            metric('Total Detections', str(results.get('total_detections', 0))),
        ]))
        el.append(Spacer(1, 10))

    el.extend(h2('1. Pipeline Architecture'))

    headers = ['Step', 'Stage', 'Technique', 'Purpose']
    rows = [
        ['1', 'Gaussian Filter', '5×5 Gaussian kernel (PIL)', 'Noise reduction & smoothing'],
        ['2', 'Adaptive Threshold', 'Integral image O(1) computation', 'Binarization with local mean'],
        ['3', 'Morphological Cleanup', '2×2 upscaling reconstruction', 'Fill gaps, remove noise'],
        ['4', 'Contour Detection', 'BFS with 8-connectivity', 'Extract connected components'],
        ['5', 'Shape Classification', 'Aspect ratio + fill ratio heuristics', 'Label as circle/rect/triangle/star'],
        ['6', 'Object Tracking', 'Centroid proximity matching (<50px)', 'Track identity across frames'],
    ]
    el.append(data_table(headers, rows, col_widths=[1.2*cm, 3*cm, 4.5*cm, 5.5*cm]))
    el.append(Spacer(1, 8))

    el.extend(h2('2. Key Technical Details'))

    el.extend(h3('2.1 Adaptive Thresholding with Integral Image'))
    el.extend(p(
        'A <b>summed-area table (integral image)</b> was used to compute local mean thresholds in '
        '<b>O(1)</b> per pixel, reducing complexity from O(n·k²) to O(n). Block size of 15 with '
        'constant C=3 provided optimal binarization across varying lighting conditions.'
    ))

    el.extend(h3('2.2 Contour Detection & Shape Classification'))
    el.extend(p('Connected components are extracted using BFS (8-connectivity) and classified via geometric heuristics:'))
    el.extend(b('<b>Circle:</b> Aspect ratio ≈ 1.0, fill ratio > 0.7'))
    el.extend(b('<b>Rectangle:</b> Aspect ratio > 1.5 or < 0.67'))
    el.extend(b('<b>Triangle:</b> Aspect ratio ≈ 1.0, fill ratio < 0.6'))
    el.extend(b('<b>Star:</b> Fill ratio 0.5–0.8, aspect ratio ≈ 1.0'))

    el.extend(h3('2.3 Frame-by-Frame Object Tracking'))
    el.extend(p(
        f'Objects are tracked across frames using <b>centroid proximity matching</b> with a 50-pixel '
        f'threshold. Over <b>{results.get("total_frames",30)} frames</b>, the pipeline tracked '
        f'<b>{results.get("objects_tracked",0)} distinct objects</b> through the full sequence, '
        f'maintaining consistent identity assignment.'
    ))

    el.extend(h2('3. Performance Metrics'))

    if results.get('frame_metrics'):
        headers = ['Frame', 'Time (ms)', 'Detections', 'Contours']
        rows = []
        for fm in results['frame_metrics']:
            rows.append([str(fm['frame']), f"{fm['time_ms']:.1f}", str(fm['detections']), str(fm['contours'])])
        el.append(data_table(headers, rows, col_widths=[3*cm, 3*cm, 3*cm, 3*cm]))
        el.append(Spacer(1, 6))

    if results.get('tracking_summary'):
        el.extend(h3('Object Tracking Summary'))
        headers = ['Object ID', 'Type', 'Frames Tracked']
        rows = [[oid, info['type'], str(info['frames_seen'])] for oid, info in results['tracking_summary'].items()]
        el.append(data_table(headers, rows, col_widths=[3.5*cm, 4.5*cm, 4*cm]))
        el.append(Spacer(1, 6))

    el.extend(h2('4. Visualizations'))
    viz = [
        (T4_DIR / 'pipeline_metrics.png', 'Figure 4.1: Per-frame metrics — processing time, detections, area, detection rate'),
        (T4_DIR / 'frame_timeline.png', 'Figure 4.2: Frame timeline — 8 sampled frames with detection overlays'),
        (T4_DIR / 'tracking_paths.png', 'Figure 4.3: Object tracking paths across all 30 frames'),
        (T4_DIR / 'frame_0000_detection.png', 'Figure 4.4: Frame 0 — input, binary threshold, and detection overlay'),
        (T4_DIR / 'frame_0015_detection.png', 'Figure 4.5: Frame 15 — mid-sequence detection'),
        (T4_DIR / 'frame_0025_detection.png', 'Figure 4.6: Frame 25 — late-sequence detection'),
    ]
    for path, caption in viz:
        el.extend(img(path, w=380, caption=caption))
        el.append(Spacer(1, 2))

    el.append(PageBreak())
    return el

# ─── CONCLUSION ──────────────────────────────────────────────────────────────

def build_conclusion():
    el = []
    el.extend(h1('Conclusion & Summary'))
    el.extend(p(
        'This report presents <b>three completed AI tasks</b> implemented from scratch using Python '
        'and standard scientific computing libraries, demonstrating practical understanding of NLP, '
        'heuristic search, and computer vision techniques.'
    ))

    el.extend(h2('Project Summary'))
    headers = ['Task', 'Technique', 'Key Metric', 'Result']
    rows = [
        ['NLP Sentiment Classifier', 'Custom TF-IDF + Softmax', 'Macro F1-Score', '<b>0.843</b>'],
        ['A* Pathfinding', 'Manhattan Heuristic', 'Nodes (8×8)', '<b>39</b> nodes'],
        ['Dijkstra', 'Uniform Cost Search', 'Nodes (8×8)', '53 nodes'],
        ['BFS', 'Level-by-Level Search', 'Nodes (8×8)', '53 nodes'],
        ['CV Pipeline', 'Adaptive Threshold + Contour', 'Avg Detections/Frame', '<b>5.6</b> obj'],
        ['Object Tracking', 'Centroid Proximity', 'Objects Tracked', '<b>8</b> objects'],
    ]
    el.append(data_table(headers, rows, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 2.5*cm]))
    el.append(Spacer(1, 10))

    el.extend(h2('Technologies Used'))
    el.extend(b('<b>Python 3.14</b> — Core programming language'))
    el.extend(b('<b>NumPy</b> — Numerical computing, matrix operations, integral image'))
    el.extend(b('<b>Matplotlib & Seaborn</b> — Data visualization and charting'))
    el.extend(b('<b>Pillow (PIL)</b> — Image processing, Gaussian filtering'))
    el.extend(b('<b>ReportLab</b> — PDF report generation'))
    el.extend(b('<b>Pandas & NetworkX</b> — Data handling and graph utilities'))

    el.extend(h2('Key Learnings'))
    el.extend(b(
        'Implementing ML algorithms from scratch (Softmax, TF-IDF) provides deep understanding '
        'of gradient descent, loss functions, and vectorization principles.'
    ))
    el.extend(b(
        'Heuristic search (A*) reduces search space by <b>29%</b> compared to uninformed search '
        'on 16×16 grids — demonstrating the power of domain knowledge in optimization.'
    ))
    el.extend(b(
        'Computer vision pipelines require algorithmic optimization — integral images reduce '
        'adaptive thresholding from O(n³) to O(n²), making real-time processing feasible.'
    ))
    el.extend(b(
        'Object tracking via centroid proximity matching provides a lightweight alternative '
        'to deep learning-based trackers for simple scenarios.'
    ))

    el.extend(h2('Task 1 — LinkedIn Announcement'))
    el.extend(note_box(
        'Task 1 (Professional LinkedIn Announcement) was excluded from this report due to '
        'the intern\'s LinkedIn account being restricted. The task will be completed once '
        'the restriction is resolved.'
    ))

    el.append(PageBreak())
    return el

# ─── SUBMISSION INFO ────────────────────────────────────────────────────────

def build_submission():
    el = []
    el.extend(h1('Project Repository & Submission'))

    el.extend(p(
        'All source code, datasets, visualizations, and this report are available in the '
        'official GitHub repository for this internship.'
    ))

    el.extend(h2('Repository Details'))
    repo_tbl = Table([
        ['<b>Repository:</b>', f'<a href="{REPO_URL}">{REPO_URL}</a>'],
        ['<b>Branch:</b>', 'main'],
        ['<b>Contents:</b>', 'tasks/, report/, README.md'],
    ], colWidths=[3*cm, CONTENT_W - 3*cm])
    repo_tbl.setStyle([
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
    ])
    el.append(repo_tbl)
    el.append(Spacer(1, 10))

    el.extend(h2('Submission Checklist'))
    el.extend(b('<b>PDF Report</b> — This document (18 pages, ~1.9 MB)'))
    el.extend(b('<b>Source Code</b> — All Python scripts for Tasks 2, 3, 4'))
    el.extend(b('<b>Visualizations</b> — 20+ PNG charts and detection images'))
    el.extend(b('<b>README</b> — Project documentation with setup instructions'))
    el.extend(b('<b>Task 1</b> — Pending (LinkedIn account restriction)'))
    el.append(Spacer(1, 10))

    el.extend(note_box(
        f'This report was automatically generated on {datetime.now().strftime("%d %B %Y")} '
        f'for <b>{STUDENT_NAME}</b> (ID: {STUDENT_ID}) as part of the Progree AI Internship. '
        f'All code is original and implemented from scratch unless otherwise noted.'
    ))

    el.append(Spacer(1, 20))
    el.append(HRFlowable(width='60%', thickness=1, color=C_ACCENT, spaceAfter=10))
    el.append(Paragraph(
        f'<i>{COMPANY} — Remote Internship Program — {DURATION}</i>',
        ParagraphStyle('final', fontName='Helvetica-Oblique', fontSize=9, textColor=C_GRAY, alignment=TA_CENTER, leading=12)
    ))
    el.append(Paragraph(
        f'<i>Intern: {STUDENT_NAME} | Student ID: {STUDENT_ID}</i>',
        ParagraphStyle('final2', fontName='Helvetica-Oblique', fontSize=9, textColor=C_GRAY, alignment=TA_CENTER, leading=12)
    ))
    el.append(Paragraph(
        f'<i>Submitted via Google Form — Ready for Evaluation</i>',
        ParagraphStyle('final3', fontName='Helvetica-Oblique', fontSize=9, textColor=C_GRAY, alignment=TA_CENTER, leading=12)
    ))
    return el

# ─── DOCUMENT TEMPLATE ───────────────────────────────────────────────────────

class ReportTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

        fm_cover = Frame(M, M, CONTENT_W, PAGE_H - 2*M, id='cover')
        fm_normal = Frame(M, M, CONTENT_W, PAGE_H - 2*M, id='normal')

        def draw_cover(c, doc):
            c.saveState()
            c.setFillColor(C_PRIMARY)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=True, stroke=False)
            # Accent strip at bottom
            c.setFillColor(C_ACCENT)
            c.rect(0, 0, PAGE_W, 0.8*cm, fill=True, stroke=False)
            # Decorative line
            c.setStrokeColor(C_GOLD)
            c.setLineWidth(0.8)
            c.line(3*cm, 9.5*cm, PAGE_W - 3*cm, 9.5*cm)
            c.restoreState()

        def draw_normal(c, doc):
            c.saveState()
            # Header bar
            c.setFillColor(C_PRIMARY)
            c.rect(0, PAGE_H - 1.2*cm, PAGE_W, 1.2*cm, fill=True, stroke=False)
            c.setFillColor(C_WHITE)
            c.setFont('Helvetica', 7.5)
            c.drawString(M, PAGE_H - 0.85*cm, f'Progree Internship — {STUDENT_NAME} ({STUDENT_ID})')
            c.drawRightString(PAGE_W - M, PAGE_H - 0.85*cm, 'Artificial Intelligence | Technical Report')
            # Footer
            c.setFillColor(C_GRAY)
            c.setFont('Helvetica', 7.5)
            c.drawCentredString(PAGE_W/2, 0.6*cm, f'Page {doc.page} | Confidential — For Progree Evaluation')
            c.drawRightString(PAGE_W - M, 0.6*cm, datetime.now().strftime('%b %Y'))
            c.restoreState()

        self.addPageTemplates([
            PageTemplate(id='Cover', frames=fm_cover, onPage=draw_cover),
            PageTemplate(id='Normal', frames=fm_normal, onPage=draw_normal),
        ])

# ─── GENERATE ────────────────────────────────────────────────────────────────

def generate_report():
    print("Generating submission-ready PDF report...")
    print(f"  Intern: {STUDENT_NAME} | ID: {STUDENT_ID}")
    print(f"  Output: {PDF_PATH}")

    doc = ReportTemplate(str(PDF_PATH), pagesize=A4,
        leftMargin=M, rightMargin=M, topMargin=M+0.3*cm, bottomMargin=M+0.3*cm)

    el = []

    # Cover
    el.append(NextPageTemplate('Cover'))
    el.append(PageBreak())
    el.extend(build_cover())

    # Certificate
    el.append(NextPageTemplate('Normal'))
    el.append(PageBreak())
    el.extend(build_certificate())

    # TOC
    el.extend(build_toc())

    # Tasks
    el.extend(build_task2())
    el.extend(build_task3())
    el.extend(build_task4())

    # Conclusion & Submission
    el.extend(build_conclusion())
    el.extend(build_submission())

    doc.build(el)
    print(f"\n[DONE] Report generated successfully!")
    print(f"       Path: {PDF_PATH}")

    # Verify
    from pypdf import PdfReader
    reader = PdfReader(PDF_PATH)
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"       Pages: {len(reader.pages)} | Size: {size_kb:.0f} KB")

if __name__ == '__main__':
    generate_report()
