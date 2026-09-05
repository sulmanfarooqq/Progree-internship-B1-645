# Independent Research Portfolio: Empirical Evaluation of AI and Search Algorithms

## Overview

This repository documents independent technical investigation conducted beyond standard degree coursework, building on practical AI work completed during an Artificial Intelligence internship at Progree Technologies.

The work focuses on three connected areas:

1. **Text classification** — implementation and evaluation of a sentiment classifier using TF-IDF features and a Softmax-based model.
2. **Search algorithms** — comparative implementation and evaluation of A*, Dijkstra, and BFS for pathfinding.
3. **Computer vision** — development and evaluation of an image-processing/detection pipeline.

## Research Approach

The investigation follows a reproducible engineering workflow: define a technical problem, implement candidate methods, run experiments, collect quantitative results, compare approaches, document limitations, and identify directions for further investigation.

## Existing Experimental Evidence

- Sentiment classification: Macro F1 = **0.843** on the documented evaluation.
- Pathfinding: A* expanded **29% fewer nodes than Dijkstra** in the documented experiment.
- Computer vision: the documented pipeline averaged **5.6 detections per frame**.

These results are reported from the underlying internship work and are not presented as independently published academic findings.

## Research Questions

- How does feature representation affect classical text-classification performance?
- How do A*, Dijkstra, and BFS differ in search efficiency and pathfinding behaviour?
- How sensitive is a classical computer-vision pipeline to its processing and detection parameters?
- What experimental practices make software-based AI investigations more reproducible and interpretable?

## Limitations

The current portfolio is an independent technical research investigation rather than peer-reviewed academic research. Future work should expand datasets, repeat experiments across multiple configurations, report variance and statistical measures where appropriate, and compare additional baseline methods.

## Related Work

The main internship implementation and technical report are contained in the parent repository. This research section organizes that work around research questions, experimentation, evaluation, and limitations.
