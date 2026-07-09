# Progree AI Internship — Technical Report

**Intern:** Muhammad Suliman  
**Student ID:** B1/645  
**Company:** Progree Technologies  
**Period:** 5 June 2026 – 4 July 2026  
**Position:** Artificial Intelligence Internship  

---

## Repository Structure

```
├── tasks/
│   ├── task2/                    # NLP Sentiment Classifier
│   │   ├── sentiment_classifier.py
│   │   └── output/               # Results JSON + visualizations
│   ├── task3/                    # Pathfinding Agent
│   │   ├── pathfinding_agent.py
│   │   └── output/               # Results JSON + grid path visualizations
│   └── task4/                    # Computer Vision Pipeline
│       ├── cv_pipeline.py
│       └── output/               # Results JSON + frame detection images
├── report/
│   ├── generate_report.py        # PDF report generator (ReportLab)
│   └── Progree_Internship_Technical_Report.pdf
├── README.md
└── .gitignore
```

## Tasks Completed

| Task | Description | Technique | Key Result |
|------|-------------|-----------|------------|
| Task 2 | Multi-Class NLP Sentiment Classifier | Custom TF-IDF + Softmax | Macro F1: **0.843** |
| Task 3 | Heuristic Graph Pathfinding Agent | A*, Dijkstra, BFS from scratch | A* 29% fewer nodes than Dijkstra |
| Task 4 | Computer Vision Object Detector & Segmenter | Adaptive Threshold + Contour Detection | 5.6 avg detections/frame |

> **Task 1 (LinkedIn Announcement)** — Excluded due to account restriction.

## How to Run

```bash
# Task 2: NLP Classifier
python tasks/task2/sentiment_classifier.py

# Task 3: Pathfinding Agent
python tasks/task3/pathfinding_agent.py

# Task 4: CV Pipeline
python tasks/task4/cv_pipeline.py

# Generate PDF Report
python report/generate_report.py
```

## Report

The full technical report is available as PDF:  
`report/Progree_Internship_Technical_Report.pdf`

---
Progree Technologies — Remote Internship Program — 2026
